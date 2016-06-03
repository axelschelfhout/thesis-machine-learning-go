package main

import (
	"fmt"
	"time"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
)

func main()  {

	fmt.Println("Starting..")

	rawData, err := base.ParseCSVToInstances("fakenames3k_slimcollumns_numeric.csv", true)
	if err != nil {
		panic(err)
	}

	// Set random seed for randomisation.
	rand.Seed(time.Now().UTC().UnixNano())

	// Split the data up in Train en Test set. The divide param is the size of the Test set.
	shuffledData := base.Shuffle(rawData) // First shuffle the set so
	trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.25)

	// Create new randomforest
	forest := ensemble.NewRandomForest(70, 5)
	err = forest.Fit(trainData)
	if err != nil {
		panic(err)
	}

	// Predict what the testData should be classified
	predictions, err := forest.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Get a ConfusionMatrix where the real labels are put against the predictions
	matrix, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(err)
	}
	// Show the matrix
	fmt.Println(evaluation.GetSummary(matrix))

}