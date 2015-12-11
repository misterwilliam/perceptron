import random
import unittest

from vector_utils import dot, normalize, sign

class Perceptron(object):

  def __init__(self, dimensions):
    self.dimensions = dimensions
    self.weights = [0.0 for _ in xrange(dimensions)]
    self.bias = 0.0

  def train(self, data, num_iters):
    for _ in xrange(num_iters):
      num_errors = 0
      for vector, label in data:
        activation = dot(self.weights, vector) + self.bias
        if label * activation <= 0:
          # Update weights
          for i in xrange(self.dimensions):
            self.weights[i] += label * vector[i]
          self.bias += label
          num_errors += 1
      print num_errors
    return self.weights, self.bias

def gen_random_training_data(num_examples, dimensions):
  weights = [random.uniform(-1, 1) for _ in xrange(dimensions)]
  bias = random.uniform(-1, 1)
  training_data = []
  for _ in xrange(num_examples):
    feature_vector = [random.uniform(-1, 1) for _ in xrange(dimensions)]
    classification = classify(feature_vector, weights, bias)
    training_data.append((feature_vector, classification))
  return training_data, weights, bias

def train(data, dimensions, max_iter):
  weights = [0.0 for _ in xrange(dimensions)]
  bias = 0.0
  for _ in xrange(max_iter):
    num_errors = 0
    for vector, label in data:
      activation = dot(weights, vector) + bias
      if label * activation <= 0:
        # Update weights
        for i in xrange(dimensions):
          weights[i] += label * vector[i]
        bias += label
        num_errors += 1
  return weights, bias

def classify(vector, weights, bias):
  activation = dot(vector, weights) + bias
  return sign(activation)

# BENCHMARK ---------------------------------------

class BenchmarkTests(unittest.TestCase):

  def test_BenchmarkAccuracy(self):
    training_data, true_weights, true_bias = gen_random_training_data(10000, 15)
    p = Perceptron(15)
    p.train(training_data, 30)
    #trained_weights, trained_bias = train(training_data, 15, 30)
    #print normalize(true_weights, true_bias), normalize(trained_weights, trained_bias)


class TrainTests(unittest.TestCase):

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