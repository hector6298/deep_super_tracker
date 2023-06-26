import numpy as np
import torch
import motmetrics as mm

from scipy import spatial
from .abstract_tracker import AbstractTracker
from .track import Track
from ..utils import hungarian_assign

INF_WORKAROUND = 999999

class TrackerIoUAssignment(AbstractTracker):
    """
    Simple IoU based tracker with t_missing feature.
    """
    def __init__(self, detector, t_missing=1, det_thresh=0.5, iou_thresh=0.5):
      """TrackerIoUAssignment constructor"""
      super().__init__(detector)
      self.t_missing = t_missing
      self.det_thresh = det_thresh
      self.iou_thresh = iou_thresh

    def add(self, new_boxes, new_scores):
        """Initializes new Track objects and saves them.
        
        Parameters:
        ===========
        - new_boxes: list
            A list of bounding boxes of the form [x1, y1, x2, y2]
        - new_scores: list
            A list of objectness scores.
        
        Returns:
        ===========
        - None
        """
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(
                new_boxes[i],
                new_scores[i],
                self.track_num + i
            ))
        self.track_num += num_new
                
    def data_association(self, boxes, scores, img):
        """The association function. It receives the detections of the current
        of the image in the current time step, computes a distance matrix
        and assigns those detections to the corresponding track that has the minimum
        ditances. The distance matrix is corresponds to the IoU distance.

        Parameters:
        ===========
        - boxes: np.array of shape [n_boxes, x1, y1, x2, y2]
            The bounding boxes of the detections in the image.
        - img: np.array of shape [1, C, H, W]
            Ignored.
        - scores: np.array of shape [n_boxes, score]
            The objectness score or whatever, I don't use this jaja.
        
        Returns:
        ==========
        - None
        """
        # Keep only boxes with enough confidence
        boxes = boxes[scores > self.det_thresh]
        
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
        
            # compute distance based on IoU (distance=1-IoU)
            distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)
            
            # Keep only distances with at most iou_thresh
            distance = np.where(distance <= self.iou_thresh, distance, distance*np.nan)
            
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

            # Keep only the active tracks in internal container
            self.tracks = [t for t in self.tracks
                           if t.id not in remove_track_ids]

            new_boxes = []
            new_scores = []

            # transpose distances to get all tracks for each box in a row
            for i, dist in enumerate(np.transpose(distance)):
                # If the current box has no affinity for any track, it means that it is a new track
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])

            # add new tracks
            self.add(new_boxes, new_scores)

        else:
            self.add(boxes, scores)


class TrackerLinearAssignment(TrackerIoUAssignment):
    """
    The same TrackerIoUAssignment, but replacing np.nanargmin in
    the assignment, for the Hungarian algorithm implemented as linear_assignment
    in scipy.optimize.
    """
    def data_association(self, boxes, scores, img):
        """The association function. It receives the detections of the current
        of the image in the current time step, computes a distance matrix
        and performs linear assignment. 
        The distance matrix is corresponds to the IoU distance.

        Parameters:
        ===========
        - boxes: np.array of shape [n_boxes, x1, y1, x2, y2]
            The bounding boxes of the detections in the image.
        - img: np.array of shape [1, C, H, W]
            Ignored.
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

        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)

            distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)
            distance = np.where(distance <= self.iou_thresh, distance, INF_WORKAROUND)

            remove_track_ids = []

            # Replace NaN values with a large value
            distance[np.isnan(distance)] = INF_WORKAROUND
            
            # Perform linear assignment
            assigned_rows, assigned_columns = hungarian_assign(distance)

            for row, col in zip(assigned_rows, assigned_columns):
                t = self.tracks[row]
                t.box = boxes[col]
                t.miss = 0

            for t in self.tracks:
                if t.id not in assigned_rows:
                    if t.miss == self.t_missing:
                        remove_track_ids.append(t.id)
                    else:
                        t.miss += 1

            self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

            new_boxes = []
            new_scores = []

            for i in range(len(boxes)):
                if i not in assigned_columns:
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])

            self.add(new_boxes, new_scores)
        else:
            self.add(boxes, scores)