import random
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


def main():
  good_enough_iteration = []
  def handle_iteration(iteration_count, weights, bias, num_errors, num_training_examples):
    if num_errors / float(num_training_examples) < 0.01:
      good_enough_iteration.append(iteration_count)
  for _ in xrange(10):
    training_data, true_weights, true_bias = gen_random_training_data(10000, 15)
    p = Perceptron(15, onIteration=handle_iteration)
    p.train(training_data, 30)
  print "Average iterations till good enough: %f" % (
    sum(good_enough_iteration) / len(good_enough_iteration))

if __name__ == "__main__":
  main()