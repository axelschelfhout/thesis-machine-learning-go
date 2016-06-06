package main

import (
	"fmt"
	"time"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/gonum/floats"
)


func runmultiplerandomtrees(data base.FixedDataGrid, iterations int) ([]float64) {

	acc := []float64{}

	for i := 0; i < iterations; i++ {

		// Set random seed for randomisation.
		rand.Seed(time.Now().UTC().UnixNano())

		// Split the data up in Train en Test set. The divide param is the size of the Test set.
		shuffledData := base.Shuffle(data) // First shuffle the set so
		trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.25)

		fmt.Println("Random Forest")
		// Create new randomforest
		forest := ensemble.NewRandomForest(70, 4)
		forest.Fit(trainData)

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

		// See the accuracy of our model.
		//fmt.Println(evaluation.GetSummary(confusionMat))
		//fmt.Println(evaluation.GetAccuracy(confusionMat))
		acc = append(acc, evaluation.GetAccuracy(matrix))
	}
	return acc
}


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

	fmt.Println("Random Forest")

	iterations := 10
	cross_val := runmultiplerandomtrees(trainData, iterations)
	fmt.Println(cross_val)
	fmt.Println(floats.Sum(cross_val)/ float64(iterations))

	panic(0)

	// Create new randomforest
	forest := ensemble.NewRandomForest(70, 4)
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