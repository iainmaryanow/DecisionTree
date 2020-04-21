class DecisionTree:
  def __init__(self, variable=None, comparision_value=None, classification=None, false_group=None, true_group=None):
    self._variable = variable                    # Feature used for the current node
    self._comparision_value = comparision_value  # Value to compare new observations against using the variable
    self._classification = classification
    self._true_group = true_group
    self._false_group = false_group

    self._true_tree = None
    self._false_tree = None


  def get_true_group(self):
    return self._true_group


  def get_false_group(self):
    return self._false_group


  def get_classification(self):
    return self._classification


  def set_true_tree(self, tree):
    self._true_tree = tree


  def set_false_tree(self, tree):
    self._false_tree = tree


  def is_leaf(self):
    return self._classification is not None


  @staticmethod
  def build(observations, max_depth=3):
    classifications = set(observation.get_classification() for observation in observations)
    if len(classifications) <= 1:
      return DecisionTree(classification=observations[0].get_classification())

    best_variable = None
    best_comparison_value = None
    best_groups = None
    minimum_gini_score = float('inf')

    for index in range(len(observations[0].get_features())):
      for observation in observations:
        groups = DecisionTree.split_observations_by_feature(index, observation.get_feature(index), observations)
        gini_score = DecisionTree.calculate_gini_score(groups, classifications)

        if gini_score < minimum_gini_score:
          best_variable = index
          best_comparison_value = observation.get_feature(index)
          best_groups = groups
          minimum_gini_score = gini_score

    tree = DecisionTree(best_variable, best_comparison_value, true_group=best_groups[0], false_group=best_groups[1])
    DecisionTree.split_tree(tree, max_depth)
    return tree


  @staticmethod
  def split_observations_by_feature(index, value, observations):
    left, right = [], []
    for observation in observations:
      if observation.get_feature(index) < value:
        left.append(observation)
      else:
        right.append(observation)
    return left, right


  @staticmethod
  def calculate_gini_score(groups, classifications):
    total_number_of_individuals = sum([len(group) for group in groups])
    gini_score = 0

    for group in groups:
      size = len(group)
      if size > 0:
        group_score = 0
        for classification in classifications:
          classification_score = [observation.get_classification() for observation in group].count(classification) / size
          group_score += classification_score * classification_score

        gini_score += (1 - group_score) * (size / total_number_of_individuals)

    return gini_score


  @staticmethod
  def split_tree(tree, max_depth, depth=1):
    true_group = tree.get_true_group()
    false_group = tree.get_false_group()

    if not true_group or not false_group:
      majority_classification = get_majority_class(true_group + false_group)
      tree.set_true_tree(DecisionTree(classification=majority_classification))
      tree.set_false_tree(DecisionTree(classification=majority_classification))
    
    elif depth >= max_depth:
      tree.set_true_tree(DecisionTree(classification=get_majority_class(true_group)))
      tree.set_false_tree(DecisionTree(classification=get_majority_class(false_group)))

    else:
      true_group_split_tree = DecisionTree.build(true_group)
      tree.set_true_tree(true_group_split_tree)
      if not true_group_split_tree.is_leaf():
        DecisionTree.split_tree(true_group_split_tree, max_depth, depth + 1)

      false_group_split_tree = DecisionTree.build(false_group)
      tree.set_false_tree(false_group_split_tree)
      if not false_group_split_tree.is_leaf():
        DecisionTree.split_tree(false_group_split_tree, max_depth, depth + 1)


  @staticmethod
  def get_majority_class(observations):
    classifications = [observation.get_classification() for observation in observations]
    return max(set(classifications), key=classifications.count)


  def predict(self, observation):
    if observation.get_feature(self._variable) < self._comparision_value:
      if self._true_tree.is_leaf():
        return self._true_tree.get_classification()
      return self._true_tree.predict(observation)

    if self._false_tree.is_leaf():
      return self._false_tree.get_classification()
    return self._false_tree.predict(observation)


  def print_tree(self, indentation=0, prefix=''):
    if self.is_leaf():
      print(indentation*'  ', prefix, self._classification)
    else:
      print(indentation*'  ', prefix, 'Is X' + str(self._variable), 'less than', self._comparision_value, '?')
      self._true_tree.print_tree(indentation + 2, '✓')
      self._false_tree.print_tree(indentation + 2, '✗')