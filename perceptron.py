import unittest

from vector_utils import array_subtract, dot, euclidean_distance, normalize, scale_array

class Perceptron(object):

  def __init__(self, dimensions, onIteration=None, initial_weights=None):
    self.dimensions = dimensions
    if initial_weights is not None:
      self.weights = initial_weights
    self.weights = normalize([0.000001 for _ in xrange(dimensions)])
    self.bias = 0.0
    self.onIteration = onIteration

  def train(self, data, num_iters):
    shouldContinue = True
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


class AveragedPerceptron(object):

  def __init__(self, dimensions, onIteration=None, initial_weights=None):
    self.dimensions = dimensions
    if initial_weights is not None:
      self.weights = initial_weights
    self.weights = normalize([0.000001 for _ in xrange(dimensions)])
    self.bias = 0.0
    self.cached_weights = normalize([0.000001 for _ in xrange(dimensions)])
    self.cached_bias = 0.0
    self.counter = 1.0
    self.onIteration = onIteration

  def train(self, data, num_iters):
    shouldContinue = True
    for iteration_count in xrange(num_iters):
      num_errors = 0
      for vector, label in data:
        activation = dot(self.weights, vector) + self.bias
        if label * activation <= 0:
          # Update weights
          for i in xrange(self.dimensions):
            self.weights[i] += label * vector[i]
            self.cached_weights += label * vector[i] * self.counter
          self.bias += label
          self.cached_bias += label * self.counter
          num_errors += 1
        self.counter += 1
      if self.onIteration is not None:
        shouldContinue = self.onIteration(iteration_count,
                                          self.weights,
                                          self.bias,
                                          num_errors,
                                          len(data))
      if shouldContinue is False:
        break
    return (
      array_subtract(self.weights, array_scale(self.cached_weights, 1 / self.counter)),
      self.bias - self.cached_bias / self.counter
    )

# TESTS --------------------------------

class PerceptronTests(unittest.TestCase):

  def test_SmallSingleStepExample(self):
    data = [
      [(1, 0), 1],
      [(0, 1), -1],
      [(1, 1), 1],
    ]
    p = Perceptron(2, initial_weights=[0.0001, 0.0001])
    trained_weights, trained_bias = p.train(data, 1)
    self.assertEqual((trained_weights, trained_bias),
                     ([1.7071067811865475, 0.7071067811865476], 0.0))



if __name__ == "__main__":
    unittest.main()