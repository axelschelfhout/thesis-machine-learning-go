package main

import (
	"fmt"

	//"github.com/cdipaolo/goml/cluster"
	"github.com/sjwhitworth/golearn/base"

	//"reflect"
	//"github.com/sjwhitworth/golearn/knn"
	//"github.com/sjwhitworth/golearn/evaluation"

)


func main() {

	// https://github.com/cdipaolo/goml

	fmt.Println("Starting..")

	rawData, err := base.ParseCSVToInstances("fakenames3k.csv", true)
	if err != nil {
		panic(err)
	}

	as := base.ResolveAttributes(rawData, rawData.AllAttributes())
	fmt.Println(as)

	//trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)





	//
	//cls := knn.NewKnnClassifier("euclidean", 2)
	//cls.Fit(trainData)

	//
	//predictions := cls.Predict(testData)
	//fmt.Println(predictions)

	//confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	//if err != nil {
	//	panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	//}
	//fmt.Println(evaluation.GetSummary(confusionMat))

	//rawData := base.ParseCSVGetAttributes("fakenames3k.csv", true)

	//model := cluster.NewTriangleKMeans(2, 30, rawData)
	//if model.Learn() != nil {
	//	panic("OH NO")
	//}


}
