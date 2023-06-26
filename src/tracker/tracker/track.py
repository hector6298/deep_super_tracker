class Track(object):
  """Simple class to define a tracklet."""
  def __init__(self, box, score, track_id):
    self.id = track_id
    self.box = box
    self.score = score
    self.miss = 0

class DeepTrack(object):
  """Class to define tracklets with CNN features."""
  def __init__(self, box, score, feature_vector, track_id):
    self.id = track_id
    self.box = box
    self.score = score
    self.track_features = feature_vector
    self.miss = 0