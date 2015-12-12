import itertools
import math
import numpy
import unittest

def dot(a, b):
  # Believe it or not this code has been benchmarked against numpy.dot and a few other
  # alternatives and so far this seems the fastest.
  acc = 0
  for i in xrange(len(a)):
    acc += a[i] * b[i]
  return acc

def sign(value):
  if value > 0:
    return 1
  elif value == 0:
    return 0
  else:  # value < 0
    return -1

def normalize(vector):
  length = euclidean_distance(vector)
  if length == 0.0:
    length = 0.0000000000001
  return [weight / length for weight in vector]

def euclidean_distance(vector):
  sum_squares = sum(math.pow(e, 2) for e in vector)
  return math.pow(sum_squares, 0.5)

def cosine_similarity(a, b):
  return dot(a, b) / (euclidean_distance(a) * euclidean_distance(b))

def scale_array(array, scalar):
  return [scalar * element for element in array]

def array_subtract(a, b):
  return [a_i - b_i for a_i, b_i in itertools.izip(a, b)]

# TESTS -------------------------------------------

class DotTests(unittest.TestCase):

  def test_SimpleExample(self):
    a = [1, 2, 3]
    b = [-1, 4, 5]
    self.assertEqual(dot(a, b), -1 + 8 + 15)


class SignTests(unittest.TestCase):

  def test_SimpleExample(self):
    self.assertEqual(sign(4), 1)
    self.assertEqual(sign(0), 0)
    self.assertEqual(sign(-4), -1)

if __name__ == "__main__":
  unittest.main()