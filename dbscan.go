package main

import(
	"fmt"
	"time"
	"math/rand"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/clustering"
	"github.com/sjwhitworth/golearn/metrics/pairwise"

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
	trainData, _ := base.InstancesTrainTestSplit(shuffledData, 0.3)

	dbs_parameters := clustering.DBSCANParameters{
		clustering.ClusterParameters{
			trainData.AllAttributes(),
			pairwise.NewEuclidean(),
		},
		0.3,
		10,
	}

	//dbs_parameters := clustering.DBSCANParameters{ Eps:1, MinCount:3 }
	dbscan, err := clustering.DBSCAN(trainData, dbs_parameters)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(dbscan)



}
