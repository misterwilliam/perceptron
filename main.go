package main

import (
  "fmt"
  "math"
  "math/rand"
)

type LabeledVector struct {
  featureVector []float64
  label bool
}

func Dot(a []float64, b[]float64) float64 {
  retVal := 0.0
  for i, val := range a {
    retVal += val * b[i]
  }
  return retVal
}

func VectorAdd(a []float64, b []float64) []float64 {
  retVal := make([]float64, len(a))
  for i, val := range(a) {
    retVal[i] = val + b[i]
  }
  return retVal
}

func VectorSubtract(a []float64, b []float64) []float64 {
  retVal := make([]float64, len(a))
  for i, val := range(a) {
    retVal[i] = val - b[i]
  }
  return retVal
}

func Score(featureVector []float64, model []float64, threshold float64) float64 {
  return Dot(featureVector, model) - threshold
}

func Classify(featureVector []float64, model []float64, threshold float64) bool {
  return Score(featureVector, model, threshold) > 0
}

func Train(trainingData []LabeledVector) ([]float64, float64) {
  model := make([]float64, len(trainingData[0].featureVector))
  threshold := 0.0
  for i := 0; i < 200; i++ {
    for _, trainingSample := range trainingData {
      modelLabel := Classify(trainingSample.featureVector, model, threshold)
      if modelLabel == trainingSample.label {
        continue
      } else if modelLabel {
        // modelLabel is true trainingSample label is false
        threshold--
        model = VectorAdd(model, trainingSample.featureVector)
      } else if !modelLabel {
        // modelLabel if false and trainingSample label is true
        threshold++
        model = VectorSubtract(model, trainingSample.featureVector)
      }
    }
  }
  return model, threshold
}

func GenTrainingSamples(model []float64,
                        threshold float64,
                        numSamples int) []LabeledVector {
  // make feature vector
  stdDev := 1.0
  vectorSize := len(model)
  samples := make([]LabeledVector, numSamples)
  for i := 0; i < numSamples; i++ {
    featureVector := make([]float64, vectorSize)
    for j := 0; j < vectorSize; j++ {
      featureVector[j] = rand.NormFloat64() * stdDev
    }
    score := Score(featureVector, model, threshold)
    samples[i] = LabeledVector{
                   featureVector,
                   rand.NormFloat64() * stdDev + score > 0.0,
                 }
  }
  return samples
}

func Normalize(model []float64, threshold float64) ([]float64, float64) {
  absoluteValue := math.Abs(threshold)
  for _, val := range model {
    absoluteValue += math.Abs(val)
  }
  normalizedModel := make([]float64, len(model))
  for i := range(model) {
    normalizedModel[i] = model[i] / absoluteValue
  }
  return normalizedModel, threshold / absoluteValue
}

func main() {
  trainingData := GenTrainingSamples([]float64{0.2, 0.3, -0.2}, 0.3, 2000)

  model, threshold := Train(trainingData)
  fmt.Println(Normalize(model, threshold))
}