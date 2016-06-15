package main

import(
	"fmt"
	"time"
	"math/rand"

	gomlbase 		"github.com/cdipaolo/goml/base"
	gomlcluster 	"github.com/cdipaolo/goml/cluster"

	golearnbase 	"github.com/sjwhitworth/golearn/base"
	golearnknn 		"github.com/sjwhitworth/golearn/knn"
	golearneval 	"github.com/sjwhitworth/golearn/evaluation"
)

func main() {

	fmt.Println("Starting...")

	scriptTimeStart := time.Now()

	kmeansTimeStart := time.Now()
	runKmeans()
	kmeansElapsedTime := time.Since(kmeansTimeStart)

	knnTimeStart := time.Now()
	runKnn()
	knnElapsedTime := time.Since(knnTimeStart)

	fmt.Println("The kMeans algorithm took the script: ", kmeansElapsedTime, " to complete.")
	fmt.Println("The kNN algorithm took the script: ", knnElapsedTime, " to complete.")
	fmt.Println("The full script took ", time.Since(scriptTimeStart), " to complete.")

}

// Run the kMeans algorithm
func runKmeans() {

	fmt.Println("Starting kMeans")

	// Load data
	data, _, err := gomlbase.LoadDataFromCSV("fakenames50k_slimcollumns_noheader_10k.csv")
	if err != nil {
		panic(err)
	}

	// Set the random seed to use random shuffling
	rand.Seed(time.Now().UTC().UnixNano())

	// Shuffle the data
	for i := range data {
		j := rand.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}

	// Normalize data.
	gomlbase.Normalize(data)

	// Start the kMeans algorithm with 65 clusters (based on the cluster calculation in Python).
	kmeans := gomlcluster.NewKMeans(65, 20, data)
	if kmeans.Learn() != nil {
		panic("The model didn't learn.. there is something wrong!")
	}

	// Save the clustered data to a file. The cluster label is appended.
	err = kmeans.SaveClusteredData("clustered.csv")
	if err != nil {
		panic(err)
	}

	fmt.Println("Finished kMeans, the clustered data is saved to 'clustered.csv'")

}

// Run the kNN algorithm
func runKnn() {

	fmt.Println("Starting kNN")
	starttime := time.Now()

	// Get the clustered data.
	labeledData, err := golearnbase.ParseCSVToInstances("clustered.csv", false)
	if err != nil {
		panic(err)
	}

	// Set random seed for randomisation
	rand.Seed(time.Now().UTC().UnixNano())

	// Split the data up in Train en Test set. The divide param is the size of the Test set.
	shuffledData := golearnbase.Shuffle(labeledData) // First shuffle the set so its random everytime we run the algoritm.
	trainData, testData := golearnbase.InstancesTrainTestSplit(shuffledData, 0.3)

	// Create the knn algorithm
	knnClassifier := golearnknn.NewKnnClassifier("euclidean", 5)

	// Fit the trainings data to the algorithm to let it learn the labels in combination with the data.
	knnClassifier.Fit(trainData)

	// Predictions made on basis of the fitted data
	predictions := knnClassifier.Predict(testData)

	// Now compare our actual test data to the predicted data.
	confusionMat, err := golearneval.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(err)
	}
	// See the accuracy of our model.
	fmt.Println(golearneval.GetSummary(confusionMat))
	fmt.Println("The predicted accuracy of this algorithm: ", golearneval.GetAccuracy(confusionMat))
	fmt.Println("kNN Learned in: ", time.Since(starttime))

	// Load the unlabeled data
	unlabeledData, err := golearnbase.ParseCSVToInstances("fakenames50k_slimcollumns_noheader_40k.csv", false)
	if err != nil {
		panic(err)
	}

	// Predict the labels over the unlabeled data.
	predictUnlabeledData := knnClassifier.Predict(unlabeledData)

	fmt.Println(predictUnlabeledData)

}
