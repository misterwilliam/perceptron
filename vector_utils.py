import math
import unittest

def dot(a, b):
  acc = 0
  for i in range(len(a)):
    acc += a[i] * b[i]
  return acc

def sign(value):
  if value > 0:
    return 1
  elif value == 0:
    return 0
  else:  # value < 0
    return -1

def normalize(vector, bias):
  total = sum(vector) + bias
  return [weight / total for weight in vector], bias / total

def euclidean_distance(vector):
  sum_squares = sum(math.pow(e, 2) for e in vector)
  return math.pow(sum_squares, 0.5)

def cosine_similarity(a, b):
  return dot(a, b) / (euclidean_distance(a) * euclidean_distance(b))

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