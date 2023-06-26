import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms


class AbstractTracker:
	"""
    An abstract Tracker class. Must be subclassed
    with specific logic and should override:
    - add()
    - data_association()
    """

	def __init__(self, obj_detect):
		"""Constructor for the Abstract class"""
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		"""Resets the record of tracklets, specifying whether to reset ids to 0 or not.
	
        Parameters:
	    ===========
	    - hard: bool
            Whether to reset the ids or not.
	    
	    Returns:
	    ===========
	    - None
        
	    """
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def get_pos(self):
		"""Get the positions of all active tracks.

		Parameters:
		===========
		- None
		
		Returns:
		===========
		- box: torch.tensor
            The bounding boxes of all active tracks in the format [x1, y1, x2, y2]
        """
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box
	
	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		
		Parameters:
		===========
		- frame: np.array of shape [C, H, W]
            The image matrix of the current time step.
	    
	    Returns:
	    ===========
	    - None
		"""
		# object detection
		boxes, scores = self.obj_detect.detect(frame['img'])

		self.data_association(boxes, scores, frame['img'])

		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1

	def get_results(self):
		"""Returns the tracker results.
		
		Parameters:
		===========
		- None
		
		Returns:
		===========
		- results: Dict
            The results.
        """
		return self.results
	

	def add(self):
		"""Initializes new Track objects and saves them."""
		raise NotImplementedError("You must override this function to work!")


	def data_association(self, boxes, scores):
		"""Perform association of detections with the tracks."""
		raise NotImplementedError("You must override this function to work!")




