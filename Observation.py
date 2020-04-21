class Observation:
  def __init__(self, classification, features):
    self._classification = classification
    self._features = features


  def get_classification(self):
    return self._classification


  def get_features(self):
    return self._features


  def get_feature(self, index):
    return self._features[index]