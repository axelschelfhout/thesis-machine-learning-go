package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/knn"
)

func main() {

	rawData, err := base.ParseCSVToInstances("fakenames3k.csv", false)
	if err != nil {
		panic(err)
	}

	cls := knn.NewKnnClassifier("euclidean", 2)

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	cls.Fit(trainData)

	predictions := cls.Predict(testData)
	fmt.Println(predictions)


}