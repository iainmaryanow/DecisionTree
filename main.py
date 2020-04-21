from DecisionTree import DecisionTree
from Observation import Observation

if __name__ == '__main__':
  observations = [
    Observation(0, [2.77, 1.78]),
    Observation(0, [1.72, 1.16]),
    Observation(0, [3.67, 2.81]),
    Observation(0, [3.96, 2.61]),
    Observation(0, [2.99, 2.20]),
    Observation(0, [10.12, 3.23]),
    Observation(1, [7.49, 3.16]),
    Observation(1, [9.00, 3.33]),
    Observation(1, [7.44, 0.47]),
    Observation(1, [6.64, 3.31])
  ]

  tree = DecisionTree.build(observations)
  tree.print_tree()
  print('\nPredicted class: %d' % tree.predict(Observation(None, [6.89, 2.95])))