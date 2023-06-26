import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class EfficientNetV2SFeaturizer:
    """
    Converts an image into a feature vector.
    """
    def __init__(self):
        """ Constructor for the EfficientNetV2SFeaturizer class.
        """
        self.weights = EfficientNet_V2_S_Weights.DEFAULT
        self.model = efficientnet_v2_s(weights=self.weights)

        # Set the model to output embeddings, instead of prob distributions
        self.model.to(device)
        self.model.classifier = self.model.classifier[:-1] 
        self.model.eval()

        # Get the transformation function for images
        self.preprocess = self.weights.transforms()
    
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
        return self.model(t_img.to(device)).detach().cpu().numpy()