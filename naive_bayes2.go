package main

import(
	"fmt"
	"time"
	"math/rand"

	"github.com/sjwhitworth/golearn/base"

	"github.com/sjwhitworth/golearn/naive"
	"github.com/sjwhitworth/golearn/filters"
	"github.com/sjwhitworth/golearn/evaluation"

	"github.com/gonum/floats"
)


func main() {

	fmt.Println("Starting")

	rawData, err := base.ParseCSVToInstances("fakenames3k_slimcollumns_numeric.csv", true)
	if err != nil {
		panic(err)
	}

	// Set random seed for randomisation.
	rand.Seed(time.Now().UTC().UnixNano())

	// Split the data up in Train en Test set. The divide param is the size of the Test set.
	shuffledData := base.Shuffle(rawData) // First shuffle the set so its random everytime we run the algoritm.
	trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.3)

	nb := naive.NewBernoulliNBClassifier()
	nb.Fit(convertFixedDataGridToBinary(trainData))


	predict := nb.Predict(convertFixedDataGridToBinary(testData))
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predict)

	fmt.Println(evaluation.GetSummary(confusionMat))

	iterations := 10
	cross_val := runmultiplenaivebayes(rawData, iterations)

	fmt.Println(cross_val)
	fmt.Println(floats.Sum(cross_val)/ float64(iterations))

}

func runmultiplenaivebayes(data base.FixedDataGrid, iterations int) ([]float64) {

	acc := []float64{}

	for i := 0; i < iterations; i++ {

		// Set random seed for randomisation.
		rand.Seed(time.Now().UTC().UnixNano())

		// Split the data up in Train en Test set. The divide param is the size of the Test set.
		shuffledData := base.Shuffle(data) // First shuffle the set so
		trainData, testData := base.InstancesTrainTestSplit(shuffledData, 0.25)

		nb := naive.NewBernoulliNBClassifier()

		nb.Fit(convertFixedDataGridToBinary(trainData))
		predict := nb.Predict(convertFixedDataGridToBinary(testData))

		// Get a ConfusionMatrix where the real labels are put against the predictions
		matrix, err := evaluation.GetConfusionMatrix(testData, predict)
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


// Convert Fixed Data Grid to all values to be binary to use in the NB classifier.
func convertFixedDataGridToBinary(data base.FixedDataGrid) base.FixedDataGrid {
	filter := filters.NewBinaryConvertFilter()
	attributes := base.NonClassAttributes(data)
	for _, a := range attributes {
		filter.AddAttribute(a)
	}
	filter.Train()
	return base.NewLazilyFilteredInstances(data, filter)
}