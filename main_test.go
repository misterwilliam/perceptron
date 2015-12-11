package main

import (
  "reflect"
  "testing"
)

func TestVectorAdd(t *testing.T) {
  a := []float64{1.0, 2.0}
  b := []float64{3.0, 0.5}
  sum := VectorAdd(a, b)
  if !reflect.DeepEqual(sum, []float64{4.0, 2.5}) {
    t.Fatalf("Wrong sum. actual %s", sum)
  }
}

func TestVectorSubtract(t *testing.T) {
  a := []float64{1.0, 2.0}
  b := []float64{3.0, 0.5}
  answer := VectorSubtract(a, b)
  if !reflect.DeepEqual(answer, []float64{-2.0, 1.5}) {
    t.Fatal("Wrong answer.", answer)
  }
}

func TestScore(t *testing.T) {
  a := []float64{1.0, 2.0}
  b := []float64{3.0, 0.5}
  score := Score(a, b)
  if score != 4.0 {
    t.Fatal("Wrong answer", score)
  }
}

func TestNormalizeVector(t *testing.T) {
  a := []float64{-1.0, 2.0}
  normalized_a := NormalizeVector(a)
  if !reflect.DeepEqual(normalized_a, []float64{-0.2, 0.4}) {
    t.Fatal("Wrong answer", normalized_a)
  }
}