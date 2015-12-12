import unittest

from vector_utils import dot

class Perceptron(object):

  def __init__(self, dimensions, onIteration=None):
    self.dimensions = dimensions
    self.weights = [0.0 for _ in xrange(dimensions)]
    self.bias = 0.0
    self.onIteration = onIteration

  def train(self, data, num_iters):
    for iteration_count in xrange(num_iters):
      num_errors = 0
      for vector, label in data:
        activation = dot(self.weights, vector) + self.bias
        if label * activation <= 0:
          # Update weights
          for i in xrange(self.dimensions):
            self.weights[i] += label * vector[i]
          self.bias += label
          num_errors += 1
      if self.onIteration is not None:
        shouldContinue = self.onIteration(iteration_count,
                                            self.weights,
                                            self.bias,
                                            num_errors,
                                            len(data))
      if shouldContinue is False:
        break
    return self.weights, self.bias


# TESTS --------------------------------

class PerceptronTests(unittest.TestCase):

  def test_SmallSingleStepExample(self):
    data = [
      [(1, 0), 1],
      [(0, 1), -1],
      [(1, 1), 1],
    ]
    p = Perceptron(2)
    trained_weights, trained_bias = p.train(data, 1)
    self.assertEqual((trained_weights, trained_bias),
                     ([2.0, 0.0], 1.0))


if __name__ == "__main__":
    unittest.main()