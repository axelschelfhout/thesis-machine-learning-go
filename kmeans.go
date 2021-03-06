package main

import (
	"fmt"
	"reflect"
	"math/rand"
	"time"

	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/cluster"
)


func main() {

	fmt.Println("Starting..")

	// Load the data
	data, _, err := base.LoadDataFromCSV("fakenames3k_slimcollumns_numeric_noheader.csv")
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

	// Start KMeans. (with k amount of clusters, x amount of iterations to run, and the dataset
	kmeans := cluster.NewKMeans(6, 20, data)

	// Let the model learn the data and determine the clusters
	if kmeans.Learn() != nil {
		panic("The model hasn't learned.. there is something wrong")
	}

	// Get the clustering result from the data.
	fmt.Println(kmeans.Guesses())

	// Concat the clusters to the actual data -> can be used in plots or in supervised algorithms
	err = kmeans.SaveClusteredData("clustered.csv")
	if err != nil {
		panic(err)
	}

}
