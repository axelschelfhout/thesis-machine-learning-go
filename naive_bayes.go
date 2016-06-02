package main

import (
	"fmt"

	"github.com/jbrukh/bayesian"
)

const(
	Good bayesian.Class = "Good"
	Bad bayesian.Class = "Bad"
)

func main() {

	fmt.Println("Starting..")

	// Naive Bayesian Classification
	// Great for classifing texts with predefined good or bad classifiers

	// Create the bayesian classifier with Good and Bad classification
	classifier := bayesian.NewClassifier(Good, Bad)

	goodThings := []string{"big","cheap","pretty"}
	badThings := []string{"small","expensive","ugly"}

	// Learn the model that the strings in the goodThings array are classified as Good
	classifier.Learn(goodThings, Good)
	// Learn the model that the strings in the badThings array are classified as Bad
	classifier.Learn(badThings, Bad)

	//
	scores, likely, _ := classifier.ProbScores(
			[]string{"big", "house", "with", "ugly", "fence", "and", "small", "lawn"},
		)

	fmt.Println(scores)
	fmt.Println(likely)

}
