package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
	"github.com/sjwhitworth/golearn/evaluation"
	"math/rand"
	"time"
)

func runmultipleknn(data base.FixedDataGrid, iterations int) ([]float64) {

	acc := []float64{}

	for i := 0; i < iterations; i++ {

		rand.Seed(time.Now().UTC().UnixNano())

		// Split the data up in Train en Test set. The divide param is the size of the Test set.
		shuffledData := base.Shuffle(data) // First shuffle the set so
		trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.3)

		// Create new Classifier
		cls := knn.NewKnnClassifier("euclidean", 5)
		// Fit the data to the classifier
		cls.Fit(trainData)

		// Predictions made on basis of the fitted data
		predictions := cls.Predict(testData)

		// Now compare our actual test data to the predicted data.
		confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
		if err != nil {
			panic(err)
		}

		// See the accuracy of our model.
		//fmt.Println(evaluation.GetSummary(confusionMat))
		//fmt.Println(evaluation.GetAccuracy(confusionMat))
		acc = append(acc, evaluation.GetAccuracy(confusionMat))
	}
	return acc
}

func main() {

	rawData, err := base.ParseCSVToInstances("fakenames3k_slimcollumns_numeric.csv", true)
	if err != nil {
		panic(err)
	}

	// Set random seed for randomisation.
	rand.Seed(time.Now().UTC().UnixNano())

	// Split the data up in Train en Test set. The divide param is the size of the Test set.
	shuffledData := base.Shuffle(rawData) // First shuffle the set so
	trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.3)

	// Create new Classifier
	cls := knn.NewKnnClassifier("euclidean", 5)
	// Fit the data to the classifier
	cls.Fit(trainData)

	// Predictions made on basis of the fitted data
	predictions := cls.Predict(testData)

	// Now compare our actual test data to the predicted data.
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(err)
	}

	// See the accuracy of our model.
	fmt.Println(evaluation.GetSummary(confusionMat))

	// Run it multiple times to see if it's correct. (cross validation)
	fmt.Println(runmultipleknn(rawData, 10))


}