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
  for i := 0; i < 100; i++ {
    numRight := 0
    numWrong := 0
    for _, trainingSample := range trainingData {
      modelLabel := Classify(trainingSample.featureVector, model, threshold)
      if modelLabel == trainingSample.label {
        numRight++
        continue
      } else if trainingSample.label {
        numWrong++
        // trainingSample label is true, lower threshold and add training sample feature
        // vector to model
        threshold--
        model = VectorAdd(model, trainingSample.featureVector)
      } else if !trainingSample.label {
        numWrong++
        // trainingSample label is false, increase threshold and subtract training sample
        // feature vector to model
        threshold++
        model = VectorSubtract(model, trainingSample.featureVector)
      }
    }
    model, threshold = Normalize(model, threshold)
    fmt.Println(model, threshold, float32(numRight) / float32(numWrong + numRight))
  }
  return model, threshold
}

func GenTrainingSamples(model []float64,
                        threshold float64,
                        numSamples int) []LabeledVector {
  // make feature vector
  stdDev := 10.5
  vectorSize := len(model)
  samples := make([]LabeledVector, numSamples)
  for i := 0; i < numSamples; i++ {
    // Make random weights
    featureVector := make([]float64, vectorSize)
    for j := 0; j < vectorSize; j++ {
      featureVector[j] = rand.Float64()
    }
    // Assign a label according to model
    score := Score(featureVector, model, threshold)
    samples[i] = LabeledVector{
                   featureVector,
                   rand.NormFloat64() * stdDev + score > 0.0,
                 }
  }
  return samples
}

func Normalize(model []float64, threshold float64) ([]float64, float64) {
  if threshold == 0.0 {
    return model, threshold
  }
  absoluteThreshold := math.Abs(threshold)
  normalizedModel := make([]float64, len(model))
  for i := range(model) {
    normalizedModel[i] = model[i] / absoluteThreshold
  }
  return normalizedModel, threshold / absoluteThreshold
}

func main() {
  trainingData := GenTrainingSamples([]float64{0.3, -0.1, 0.1, -0.1, 0.4}, 1.0, 100)
  model, threshold := Train(trainingData)
  fmt.Println(Normalize(model, threshold))
}