import random
from vector_utils import dot, sign
from perceptron import Perceptron

def classify(vector, weights, bias):
  activation = dot(vector, weights) + bias
  return sign(activation)

def gen_random_training_data(num_examples, dimensions):
  weights = [random.uniform(-1, 1) for _ in xrange(dimensions)]
  bias = random.uniform(-1, 1)
  training_data = []
  for _ in xrange(num_examples):
    feature_vector = [random.uniform(-1, 1) for _ in xrange(dimensions)]
    classification = classify(feature_vector, weights, bias)
    training_data.append((feature_vector, classification))
  return training_data, weights, bias


class Benchmarker(object):

  def __init__(self):
    self.num_iterations_till_good_enough_log = []

  def handle_iteration(self, iteration_count, weights, bias, num_errors,
                       num_training_examples):
    if num_errors / float(num_training_examples) < 0.01:
      self.num_iterations_till_good_enough_log.append(iteration_count)
      return False

  def get_avg_iterations_required(self):
    print self.num_iterations_till_good_enough_log
    return (float(sum(self.num_iterations_till_good_enough_log)) /
              len(self.num_iterations_till_good_enough_log))

  def reset(self):
    self.num_iterations_till_good_enough_log = []

def main():
  benchmarker = Benchmarker()
  for _ in xrange(10):
    training_data, true_weights, true_bias = gen_random_training_data(10000, 15)
    p = Perceptron(15, onIteration=benchmarker.handle_iteration)
    p.train(training_data, 100)
  print "Average iterations till good enough: %f" % benchmarker.get_avg_iterations_required()


if __name__ == "__main__":
  main()