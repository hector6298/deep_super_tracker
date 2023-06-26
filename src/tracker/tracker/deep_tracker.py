import numpy as np
import torch
import motmetrics as mm

from scipy import spatial
from .abstract_tracker import AbstractTracker
from .track import DeepTrack
from ..utils import hungarian_assign

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

INF_WORKAROUND = 999999

class DeepSuperTracker(AbstractTracker):
    """
    A mix of IoU distance-based tracker with CNN latent features.
    """
    def __init__(self, 
                 detector, 
                 featurizer, 
                 weights, 
                 t_missing=1, 
                 det_thresh=0.5, 
                 iou_thresh=0.5,
                 alpha=0.3,
                 beta=0.7):
        
        super().__init__(detector)
        self.t_missing = t_missing
        self.det_thresh = det_thresh
        self.iou_thresh = iou_thresh
        self.t_missing = t_missing

        self.featurizer = featurizer
        self.weights = weights
        self.preprocess = weights.transforms()
        self.alpha = alpha
        self.beta = beta
    
    def cosine_distance(self, feat1, feat2):
        """Compute the cosine distance between 
        two n-dimensional vectors.

        Parameters:
        ===========
        - feat1: np.ndarray
               A 1D numpy array with the features extracted from
               an image patch.
        - feat2: np.ndarray
               A 1D numpy array with the features extracted from
               an image patch.

        Returns:
        ==========
        - dist: float
              The cosine distance. Range [0,1]
        """
        return spatial.distance.cosine(feat1, feat2)

    def get_vector(self, img):
        """ Gets the feature vector of an image.
        Given a CNN feature extractor from torchvision.models
        and a preprocessing function from torchvision.model.EfficientNet_V2_S_Weights.DEFAULT
        it preprocesses the image to be compatible with the input shape of the CNN
        and extracts a 1D feature vector aftewards.

        Parameters:
        ===========
        - img: np.array of shape [C, H, W]
             The image data matrix.

        Returns:
        ==========
        - feature_vector: np.ndarray
            The latent feature vector
        """
        # Transforms to the required shape and adds a new dimension 
        t_img = self.preprocess(img.unsqueeze(0))
        return self.featurizer(t_img.to(device)).detach().cpu().numpy()

    def cosine_matrix(self, track_features, patches_features):
        """Compute the pair-wise cosine distance matrix
        between two lists of latent feature vectors. This function
        calls cosine_distance() for every possible pair of vectors.

        Parameters:
        ===========
        - track_features: list[np.ndarray]
            A list of latent feature vectors.
        - patches_features: list[np.ndarray]
            A list of latent feature vectors.

        Returns:
        - cosine_distance: 2D np.array of shape [N,N]
            The distance matrix.
        """
        cosine_distance = []
        # Iterates over each possible pair and computes cosine_distance()
        for track_feature in track_features:
            track_boxes_d = []
            for patch_features in patches_features:
                track_boxes_d.append(self.cosine_distance(track_feature[0], patch_features[0]))
            cosine_distance.append(track_boxes_d)
        return np.array(cosine_distance)
        
    def get_patches_features(self, boxes, img):
        """Get the image patches for each bounding box in the image.

        Parameters:
        ===========
        - boxes: np.array of shape [n_boxes, x1, y1, x2, y2]
            The bounding boxes of the detections in the image.
        - img: np.array of shape [C, H, W]
            The image matrix.

        Returns:
        ==========
        - patches_features: list[np.ndarray]
            A list with the feature vectors for all of the detections. 
        """
        patches_features = []
        for box in boxes.numpy().astype(int):
            patch = img[:, box[1]:box[3]+1, box[0]:box[2]+1]
            patches_features.append(self.get_vector(patch))
        return patches_features

    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them.
        
        Parameters:
        ===========
        - new_boxes: list
            A list of bounding boxes of the form [x1, y1, x2, y2]
        - new_scores: list
            A list of objectness scores.
        - new_features: list
            A list of CNN latent features obtained from image patches.
        
        Returns:
        ===========
        - None
        """
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(DeepTrack(
                new_boxes[i],
                new_scores[i],
                new_features[i],
                self.track_num + i
            ))
        self.track_num += num_new

    def data_association(self, boxes, scores, img):
        """The association function. It receives the detections of the current
        of the image in the current time step, computes a distance matrix
        and assigns those detections to the corresponding track that has the minimum
        ditances. The distance matrix is the result of the following weighted sum:

        distance = alpha*cosine_distance + beta*iou_distance

        Where the cosine_distance uses every pair of latent feature vectors
        from the vectors stored in the current tracks and the vectors from the detections
        in the current time step.

        Parameters:
        ===========
        - boxes: np.array of shape [n_boxes, x1, y1, x2, y2]
            The bounding boxes of the detections in the image.
        - img: np.array of shape [1, C, H, W]
            The image matrix with a batch dimension at the beggining.
            This batch dimension will be squeezed, as this function only handles one 
            image at a time.
        - scores: np.array of shape [n_boxes, score]
            The objectness score or whatever, I don't use this jaja.
        
        Returns:
        ==========
        - None
        """
        # Keep only boxes with enough confidence
        boxes = boxes[scores > self.det_thresh]
        
        # Squeeze the image
        img = img[0]
        
        patches_features = self.get_patches_features(boxes, img)
        
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            track_features = np.stack([t.track_features for t in self.tracks], axis=0)

            # Compute compound distance (alpha*cosine + beta*iou)
            cosine_distance = self.cosine_matrix(track_features, patches_features)
            iou_distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)
            
            iou_distance = np.where(iou_distance <= self.iou_thresh, iou_distance, iou_distance*np.nan)
            distance = self.alpha*cosine_distance + self.beta*iou_distance
            
            remove_track_ids = []
            
            d_copy = distance.copy()
            
            # Iterate over each track and the corresponding distance vector
            for t, dist in zip(self.tracks, d_copy):
               
                # If no affinity in any of the bounding boxes for the current frame
                # then check if max is reached for missing -> remove track
                if np.isnan(dist).all():
                    if t.miss == self.t_missing:
                        remove_track_ids.append(t.id)
                    else:
                        t.miss += 1

                # Otherwise get the id whose distance is minimum
                else:
                    match_id = np.nanargmin(dist)
                    
                    # Tracks already assgined will be deleted
                    d_copy[:,match_id] = np.nan
                    
                    t.box = boxes[match_id]
                    t.miss = 0
                    t.track_features = patches_features[match_id]
                    
            # Keep only the active tracks in internal container
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]

            new_boxes = []
            new_scores = []
            new_features = []
            
            # transpose distances to get all tracks for each box in a row
            for i, dist in enumerate(np.transpose(distance)):
                # If the current box has no affinity for any track, it means that it is a new track
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
                    new_features.append(patches_features[i])
                    
            # add new tracks
            self.add(new_boxes, new_scores, new_features)

        else:
            self.add(boxes, scores, patches_features)


class DeepHungarianTracker(DeepSuperTracker):
    """
    The same DeepSuperTracker, but using the Hungarian Algorithm
    instead of np.nanargmin. 
    You could say that this is the DeepSuperHyperTracker (?),
    but that is more ridiculous than DeepHungarianTracker.
    """

    def data_association(self, boxes, scores, img):
        """The association function. It receives the detections of the current
        of the image in the current time step, computes a distance matrix
        and performs linear assignment.
        The distance matrix is the result of the following weighted sum:

        distance = alpha*cosine_distance + beta*iou_distance

        Where the cosine_distance uses every pair of latent feature vectors
        from the vectors stored in the current tracks and the vectors from the detections
        in the current time step.

        Parameters:
        ===========
        - boxes: np.array of shape [n_boxes, x1, y1, x2, y2]
            The bounding boxes of the detections in the image.
        - img: np.array of shape [1, C, H, W]
            The image matrix with a batch dimension at the beggining.
            This batch dimension will be squeezed, as this function only handles one 
            image at a time.
        - scores: np.array of shape [n_boxes, score]
            The objectness score or whatever, I don't use this jaja.
        
        Returns:
        ==========
        - None
        """
        # Keep only boxes with enough confidence
        boxes = boxes[scores > self.det_thresh]

        # Squeeze the image
        img = img[0]

        patches_features = self.get_patches_features(boxes, img)

        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            track_features = np.stack([t.track_features for t in self.tracks], axis=0)

            # Compute compound distance (alpha*cosine + beta*iou)
            cosine_distance = self.cosine_matrix(track_features, patches_features)
            iou_distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            iou_distance = np.where(iou_distance <= self.iou_thresh, iou_distance, INF_WORKAROUND)
            distance = self.alpha * cosine_distance + self.beta * iou_distance
            
            # Replace NaN values with a large value
            distance[np.isnan(distance)] = INF_WORKAROUND

            remove_track_ids = []

            # Perform linar assignment
            assigned_rows, assigned_columns = hungarian_assign(distance)

            for row, col in zip(assigned_rows, assigned_columns):
                t = self.tracks[row]
                t.box = boxes[col]
                t.miss = 0
                t.track_features = patches_features[col]

            for t in self.tracks:
                if t.id not in assigned_rows:
                    if t.miss == self.t_missing:
                        remove_track_ids.append(t.id)
                    else:
                        t.miss += 1

            self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

            new_boxes = []
            new_scores = []
            new_features = []

            for i in range(len(boxes)):
                if i not in assigned_columns:
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
                    new_features.append(patches_features[i])

            self.add(new_boxes, new_scores, new_features)
        else:
            self.add(boxes, scores, patches_features)