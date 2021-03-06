import argparse
import cProfile
import math
import numpy
import pstats
import random
import time

from vector_utils import dot, normalize, sign
from perceptron import Perceptron, AveragedPerceptron

def classify(vector, weights, bias):
  activation = dot(vector, weights) + bias
  return sign(activation)

def gen_random_training_data(num_examples, dimensions,
                             initial_weights=None, initial_bias=None):
  if initial_weights is None:
    weights = normalize([random.uniform(-1, 1) for _ in xrange(dimensions)])
  else:
    weights = initial_weights
  if initial_bias is None:
    bias = random.uniform(-1, 1)
  else:
    bias = initial_bias
  training_data = []
  for _ in xrange(num_examples):
    feature_vector = [random.uniform(-1, 1) for _ in xrange(dimensions)]
    classification = classify(feature_vector, weights, bias)
    training_data.append((feature_vector, classification))
  return training_data, weights, bias


class Benchmarker(object):

  def __init__(self, perceptron, num_dimensions, num_training_examples):
    # |perceptron| is a perceptron constructor
    self.num_iterations_till_good_enough_log = []
    self.perceptron = perceptron
    self.num_dimensions = num_dimensions
    self.num_training_examples = num_training_examples
    self.test_correct = 0
    self.test_case = 0

  def benchmark(self, num_trials):
    for _ in xrange(num_trials):
      training_data, true_weights, true_bias = gen_random_training_data(
        self.num_training_examples, self.num_dimensions)
      p = self.perceptron(self.num_dimensions, onIteration=self.handle_iteration)
      p.train(training_data, 9999999999)
      # Eval
      testing_data, true_weights, true_bias = gen_random_training_data(
        500, self.num_dimensions, initial_weights=true_weights,
        initial_bias=true_bias)
      for vector, true_label in testing_data:
        label = p.classify(vector)
        if label == true_label:
          self.test_correct += 1
        self.test_case += 1
    # If this assertion fails then p.train(training_data, 999999999) is the problem.
    # The number 9999999999 is not big enough.
    assert(len(self.num_iterations_till_good_enough_log) == num_trials)

  def handle_iteration(self, iteration_count, weights, bias, num_errors,
                       num_training_examples):
    if num_errors / float(num_training_examples) < 0.01:
      self.num_iterations_till_good_enough_log.append(iteration_count)
      return False

  def print_stats(self):
    print("Benchmarking: num training examples: {} num dimensions: {}".format(
      self.num_training_examples, self.num_dimensions))
    print("Average iterations till good enough: {:f} std_dev: {:f} num trials: {}".format(
      numpy.mean(self.num_iterations_till_good_enough_log),
      numpy.std(self.num_iterations_till_good_enough_log),
      len(self.num_iterations_till_good_enough_log)))
    print("Test accuracy: {:f} num trials: {}".format(
      self.test_correct / float(self.test_case), self.test_case))

  def reset(self):
    self.num_iterations_till_good_enough_log = []

def main():
  parser = argparse.ArgumentParser(description='Benchmark perceptron.py')
  parser.add_argument("--profile", action="store_true", help="Do profiling")
  args = parser.parse_args()

  if args.profile:
    pr = cProfile.Profile()
    pr.enable()

  print("------- Perceptron -------")
  start = time.clock()
  benchmarker = Benchmarker(Perceptron, 10, 5000)
  benchmarker.benchmark(40)
  end = time.clock()
  print("Elapsed time: %.03f sec" % (end - start))
  benchmarker.print_stats()

  print("--- Averaged Perceptron ---")
  start = time.clock()
  benchmarker = Benchmarker(AveragedPerceptron, 10, 5000)
  benchmarker.benchmark(40)
  end = time.clock()
  print("Elapsed time: %.03f sec" % (end - start))
  benchmarker.print_stats()


  if args.profile:
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats()


if __name__ == "__main__":
  main()