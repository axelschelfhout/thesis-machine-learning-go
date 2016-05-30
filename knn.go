package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
	"github.com/sjwhitworth/golearn/evaluation"
	"math/rand"
)

func main() {

	rawData, err := base.ParseCSVToInstances("fakenames3k_slimcollumns_numeric.csv", true)
	if err != nil {
		panic(err)
	}

	// Set random seed for randomasation.
	rand.Seed(1337)

	// Split the data up in Train en Test set. The divide param is the size of the Test set.
	shuffledData := base.Shuffle(rawData) // First shuffle the set so
	trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.21)

	// Create new Classifier
	cls := knn.NewKnnClassifier("euclidean", 5)

	// Fit the data to the classifier
	cls.Fit(trainData)

	// Predictions made on basis of the fitted data
	predictions := cls.Predict(testData)

	// Now compare our actual test data to the predictioned data.
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(err)
	}

	// See the accuracy of our model.
	fmt.Println(evaluation.GetSummary(confusionMat))

	/**
	Eval: (3 neighbours)
	This changes with the random seed. But when I dont set a seed the accuracy is as follows:
		train/test -> accuracy
		70/30 -> 0.8184
		60/40 -> 0.8154
		65/35 -> 0.8247
		64/36 -> 0.8245
		63/37 -> 0.8225
		66/34 -> 0.8226
		67/33 -> 0.8210
		80/20 -> 0.8231
		75/25 -> 0.8252
		77/23 -> 0.8227
		79/21 -> 0.8255 #
		78/22 -> 0.8248
	 */

}