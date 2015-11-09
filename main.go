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

func Score(featureVector []float64, model []float64) float64 {
  return Dot(featureVector, model)
}

func Classify(featureVector []float64, model []float64,) bool {
  return Score(featureVector, model) > 0
}

func Train(trainingData []LabeledVector) []float64 {
  model := make([]float64, len(trainingData[0].featureVector))
  for i := 0; i < 10000; i++ {
    numRight := 0
    numWrong := 0
    for _, trainingSample := range trainingData {
      modelLabel := Classify(trainingSample.featureVector, model)
      if modelLabel == trainingSample.label {
        numRight++
        continue
      } else if trainingSample.label {
        numWrong++
        // trainingSample label is true, add training sample feature vector to model
        model = VectorAdd(model, trainingSample.featureVector)
      } else if !trainingSample.label {
        numWrong++
        // trainingSample label is false, subtract training sample feature vector to model
        model = VectorSubtract(model, trainingSample.featureVector)
      }
    }
    model = NormalizeVector(model)
  }
  return model
}

func GenTrainingSamples(model []float64,
                        numSamples int) []LabeledVector {
  // make feature vector
  stdDev := 0.0
  vectorSize := len(model)
  samples := make([]LabeledVector, numSamples)
  for i := 0; i < numSamples; i++ {
    // Make random normalized weights
    featureVector := make([]float64, vectorSize)
    for j := 0; j < vectorSize; j++ {
      featureVector[j] = rand.Float64()
    }
    featureVector = NormalizeVector(featureVector)
    // Assign a label according to model
    score := Score(featureVector, model)
    samples[i] = LabeledVector{
                   featureVector,
                   rand.NormFloat64() * stdDev + score > 0.0,
                 }
  }
  return samples
}

// Normalize vector so that Euclidean distance is 1
func NormalizeVector(vector []float64) []float64 {
  euclideanDistance := math.Sqrt(Dot(vector, vector))
  retVal := make([]float64, len(vector))
  for i, val := range vector {
    retVal[i] = val / euclideanDistance
  }
  return retVal
}

func main() {
  trueModel := NormalizeVector([]float64{0.3, -0.1, 0.1, -0.1, 0.4})
  trainingData := GenTrainingSamples(trueModel, 50000)
  model := Train(trainingData)
  fmt.Println(model)
  fmt.Println("Actual model", trueModel)
}