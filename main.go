//Author: Nathaniel Cantwell
//Neural network which classifies the iris data set

package main

import "fmt"

//Neuron struct which acts as a node in the network
type Neuron struct {
	//Fill in with stuff
}

//Network struct which holds the neurons and other data
type Network struct {
	//The layers of neurons in the network
	inputLayer  []Neuron
	hiddenLayer []Neuron
	outputLayer []Neuron
}

func main() {
	fmt.Println("We're making a neural network!")

	//Initialize the hard coded data
	data := initData()
	fmt.Println(data)

	//Initialize the network
	var myNetwork Network
	myNetwork.initNetwork(4, 7, 3)
	fmt.Println(myNetwork)
}

//initNetwork initializes and populates the network with neurons
// whose weights are set randomly
func (net *Network) initNetwork(numInputs, numHidden, numOutput int) {
	net.inputLayer = make([]Neuron, numInputs+1)
	net.hiddenLayer = make([]Neuron, numHidden+1)
	net.outputLayer = make([]Neuron, numOutput+1)
}

//initData returns a [][]float64 containing the hard coded iris data
func initData() [][]float64 {
	var allData [][]float64

	// sepal length, width, petal length, width
	// Iris setosa = 0 0 1
	// Iris versicolor = 0 1 0
	// Iris virginica = 1 0 0

	allData = append(allData, []float64{5.1, 3.5, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.9, 3.0, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.7, 3.2, 1.3, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.6, 3.1, 1.5, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.6, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.4, 3.9, 1.7, 0.4, 0, 0, 1})
	allData = append(allData, []float64{4.6, 3.4, 1.4, 0.3, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.4, 1.5, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.4, 2.9, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.9, 3.1, 1.5, 0.1, 0, 0, 1})

	allData = append(allData, []float64{5.4, 3.7, 1.5, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.8, 3.4, 1.6, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.8, 3.0, 1.4, 0.1, 0, 0, 1})
	allData = append(allData, []float64{4.3, 3.0, 1.1, 0.1, 0, 0, 1})
	allData = append(allData, []float64{5.8, 4.0, 1.2, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.7, 4.4, 1.5, 0.4, 0, 0, 1})
	allData = append(allData, []float64{5.4, 3.9, 1.3, 0.4, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.5, 1.4, 0.3, 0, 0, 1})
	allData = append(allData, []float64{5.7, 3.8, 1.7, 0.3, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.8, 1.5, 0.3, 0, 0, 1})

	allData = append(allData, []float64{5.4, 3.4, 1.7, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.7, 1.5, 0.4, 0, 0, 1})
	allData = append(allData, []float64{4.6, 3.6, 1.0, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.3, 1.7, 0.5, 0, 0, 1})
	allData = append(allData, []float64{4.8, 3.4, 1.9, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.0, 1.6, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.4, 1.6, 0.4, 0, 0, 1})
	allData = append(allData, []float64{5.2, 3.5, 1.5, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.2, 3.4, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.7, 3.2, 1.6, 0.2, 0, 0, 1})

	allData = append(allData, []float64{4.8, 3.1, 1.6, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.4, 3.4, 1.5, 0.4, 0, 0, 1})
	allData = append(allData, []float64{5.2, 4.1, 1.5, 0.1, 0, 0, 1})
	allData = append(allData, []float64{5.5, 4.2, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.9, 3.1, 1.5, 0.1, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.2, 1.2, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.5, 3.5, 1.3, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.9, 3.1, 1.5, 0.1, 0, 0, 1})
	allData = append(allData, []float64{4.4, 3.0, 1.3, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.4, 1.5, 0.2, 0, 0, 1})

	allData = append(allData, []float64{5.0, 3.5, 1.3, 0.3, 0, 0, 1})
	allData = append(allData, []float64{4.5, 2.3, 1.3, 0.3, 0, 0, 1})
	allData = append(allData, []float64{4.4, 3.2, 1.3, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.5, 1.6, 0.6, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.8, 1.9, 0.4, 0, 0, 1})
	allData = append(allData, []float64{4.8, 3.0, 1.4, 0.3, 0, 0, 1})
	allData = append(allData, []float64{5.1, 3.8, 1.6, 0.2, 0, 0, 1})
	allData = append(allData, []float64{4.6, 3.2, 1.4, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.3, 3.7, 1.5, 0.2, 0, 0, 1})
	allData = append(allData, []float64{5.0, 3.3, 1.4, 0.2, 0, 0, 1})

	allData = append(allData, []float64{7.0, 3.2, 4.7, 1.4, 0, 1, 0})
	allData = append(allData, []float64{6.4, 3.2, 4.5, 1.5, 0, 1, 0})
	allData = append(allData, []float64{6.9, 3.1, 4.9, 1.5, 0, 1, 0})
	allData = append(allData, []float64{5.5, 2.3, 4.0, 1.3, 0, 1, 0})
	allData = append(allData, []float64{6.5, 2.8, 4.6, 1.5, 0, 1, 0})
	allData = append(allData, []float64{5.7, 2.8, 4.5, 1.3, 0, 1, 0})
	allData = append(allData, []float64{6.3, 3.3, 4.7, 1.6, 0, 1, 0})
	allData = append(allData, []float64{4.9, 2.4, 3.3, 1.0, 0, 1, 0})
	allData = append(allData, []float64{6.6, 2.9, 4.6, 1.3, 0, 1, 0})
	allData = append(allData, []float64{5.2, 2.7, 3.9, 1.4, 0, 1, 0})

	allData = append(allData, []float64{5.0, 2.0, 3.5, 1.0, 0, 1, 0})
	allData = append(allData, []float64{5.9, 3.0, 4.2, 1.5, 0, 1, 0})
	allData = append(allData, []float64{6.0, 2.2, 4.0, 1.0, 0, 1, 0})
	allData = append(allData, []float64{6.1, 2.9, 4.7, 1.4, 0, 1, 0})
	allData = append(allData, []float64{5.6, 2.9, 3.6, 1.3, 0, 1, 0})
	allData = append(allData, []float64{6.7, 3.1, 4.4, 1.4, 0, 1, 0})
	allData = append(allData, []float64{5.6, 3.0, 4.5, 1.5, 0, 1, 0})
	allData = append(allData, []float64{5.8, 2.7, 4.1, 1.0, 0, 1, 0})
	allData = append(allData, []float64{6.2, 2.2, 4.5, 1.5, 0, 1, 0})
	allData = append(allData, []float64{5.6, 2.5, 3.9, 1.1, 0, 1, 0})

	allData = append(allData, []float64{5.9, 3.2, 4.8, 1.8, 0, 1, 0})
	allData = append(allData, []float64{6.1, 2.8, 4.0, 1.3, 0, 1, 0})
	allData = append(allData, []float64{6.3, 2.5, 4.9, 1.5, 0, 1, 0})
	allData = append(allData, []float64{6.1, 2.8, 4.7, 1.2, 0, 1, 0})
	allData = append(allData, []float64{6.4, 2.9, 4.3, 1.3, 0, 1, 0})
	allData = append(allData, []float64{6.6, 3.0, 4.4, 1.4, 0, 1, 0})
	allData = append(allData, []float64{6.8, 2.8, 4.8, 1.4, 0, 1, 0})
	allData = append(allData, []float64{6.7, 3.0, 5.0, 1.7, 0, 1, 0})
	allData = append(allData, []float64{6.0, 2.9, 4.5, 1.5, 0, 1, 0})
	allData = append(allData, []float64{5.7, 2.6, 3.5, 1.0, 0, 1, 0})

	allData = append(allData, []float64{5.5, 2.4, 3.8, 1.1, 0, 1, 0})
	allData = append(allData, []float64{5.5, 2.4, 3.7, 1.0, 0, 1, 0})
	allData = append(allData, []float64{5.8, 2.7, 3.9, 1.2, 0, 1, 0})
	allData = append(allData, []float64{6.0, 2.7, 5.1, 1.6, 0, 1, 0})
	allData = append(allData, []float64{5.4, 3.0, 4.5, 1.5, 0, 1, 0})
	allData = append(allData, []float64{6.0, 3.4, 4.5, 1.6, 0, 1, 0})
	allData = append(allData, []float64{6.7, 3.1, 4.7, 1.5, 0, 1, 0})
	allData = append(allData, []float64{6.3, 2.3, 4.4, 1.3, 0, 1, 0})
	allData = append(allData, []float64{5.6, 3.0, 4.1, 1.3, 0, 1, 0})
	allData = append(allData, []float64{5.5, 2.5, 4.0, 1.3, 0, 1, 0})

	allData = append(allData, []float64{5.5, 2.6, 4.4, 1.2, 0, 1, 0})
	allData = append(allData, []float64{6.1, 3.0, 4.6, 1.4, 0, 1, 0})
	allData = append(allData, []float64{5.8, 2.6, 4.0, 1.2, 0, 1, 0})
	allData = append(allData, []float64{5.0, 2.3, 3.3, 1.0, 0, 1, 0})
	allData = append(allData, []float64{5.6, 2.7, 4.2, 1.3, 0, 1, 0})
	allData = append(allData, []float64{5.7, 3.0, 4.2, 1.2, 0, 1, 0})
	allData = append(allData, []float64{5.7, 2.9, 4.2, 1.3, 0, 1, 0})
	allData = append(allData, []float64{6.2, 2.9, 4.3, 1.3, 0, 1, 0})
	allData = append(allData, []float64{5.1, 2.5, 3.0, 1.1, 0, 1, 0})
	allData = append(allData, []float64{5.7, 2.8, 4.1, 1.3, 0, 1, 0})

	allData = append(allData, []float64{6.3, 3.3, 6.0, 2.5, 1, 0, 0})
	allData = append(allData, []float64{5.8, 2.7, 5.1, 1.9, 1, 0, 0})
	allData = append(allData, []float64{7.1, 3.0, 5.9, 2.1, 1, 0, 0})
	allData = append(allData, []float64{6.3, 2.9, 5.6, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.5, 3.0, 5.8, 2.2, 1, 0, 0})
	allData = append(allData, []float64{7.6, 3.0, 6.6, 2.1, 1, 0, 0})
	allData = append(allData, []float64{4.9, 2.5, 4.5, 1.7, 1, 0, 0})
	allData = append(allData, []float64{7.3, 2.9, 6.3, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.7, 2.5, 5.8, 1.8, 1, 0, 0})
	allData = append(allData, []float64{7.2, 3.6, 6.1, 2.5, 1, 0, 0})

	allData = append(allData, []float64{6.5, 3.2, 5.1, 2.0, 1, 0, 0})
	allData = append(allData, []float64{6.4, 2.7, 5.3, 1.9, 1, 0, 0})
	allData = append(allData, []float64{6.8, 3.0, 5.5, 2.1, 1, 0, 0})
	allData = append(allData, []float64{5.7, 2.5, 5.0, 2.0, 1, 0, 0})
	allData = append(allData, []float64{5.8, 2.8, 5.1, 2.4, 1, 0, 0})
	allData = append(allData, []float64{6.4, 3.2, 5.3, 2.3, 1, 0, 0})
	allData = append(allData, []float64{6.5, 3.0, 5.5, 1.8, 1, 0, 0})
	allData = append(allData, []float64{7.7, 3.8, 6.7, 2.2, 1, 0, 0})
	allData = append(allData, []float64{7.7, 2.6, 6.9, 2.3, 1, 0, 0})
	allData = append(allData, []float64{6.0, 2.2, 5.0, 1.5, 1, 0, 0})

	allData = append(allData, []float64{6.9, 3.2, 5.7, 2.3, 1, 0, 0})
	allData = append(allData, []float64{5.6, 2.8, 4.9, 2.0, 1, 0, 0})
	allData = append(allData, []float64{7.7, 2.8, 6.7, 2.0, 1, 0, 0})
	allData = append(allData, []float64{6.3, 2.7, 4.9, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.7, 3.3, 5.7, 2.1, 1, 0, 0})
	allData = append(allData, []float64{7.2, 3.2, 6.0, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.2, 2.8, 4.8, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.1, 3.0, 4.9, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.4, 2.8, 5.6, 2.1, 1, 0, 0})
	allData = append(allData, []float64{7.2, 3.0, 5.8, 1.6, 1, 0, 0})

	allData = append(allData, []float64{7.4, 2.8, 6.1, 1.9, 1, 0, 0})
	allData = append(allData, []float64{7.9, 3.8, 6.4, 2.0, 1, 0, 0})
	allData = append(allData, []float64{6.4, 2.8, 5.6, 2.2, 1, 0, 0})
	allData = append(allData, []float64{6.3, 2.8, 5.1, 1.5, 1, 0, 0})
	allData = append(allData, []float64{6.1, 2.6, 5.6, 1.4, 1, 0, 0})
	allData = append(allData, []float64{7.7, 3.0, 6.1, 2.3, 1, 0, 0})
	allData = append(allData, []float64{6.3, 3.4, 5.6, 2.4, 1, 0, 0})
	allData = append(allData, []float64{6.4, 3.1, 5.5, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.0, 3.0, 4.8, 1.8, 1, 0, 0})
	allData = append(allData, []float64{6.9, 3.1, 5.4, 2.1, 1, 0, 0})

	allData = append(allData, []float64{6.7, 3.1, 5.6, 2.4, 1, 0, 0})
	allData = append(allData, []float64{6.9, 3.1, 5.1, 2.3, 1, 0, 0})
	allData = append(allData, []float64{5.8, 2.7, 5.1, 1.9, 1, 0, 0})
	allData = append(allData, []float64{6.8, 3.2, 5.9, 2.3, 1, 0, 0})
	allData = append(allData, []float64{6.7, 3.3, 5.7, 2.5, 1, 0, 0})
	allData = append(allData, []float64{6.7, 3.0, 5.2, 2.3, 1, 0, 0})
	allData = append(allData, []float64{6.3, 2.5, 5.0, 1.9, 1, 0, 0})
	allData = append(allData, []float64{6.5, 3.0, 5.2, 2.0, 1, 0, 0})
	allData = append(allData, []float64{6.2, 3.4, 5.4, 2.3, 1, 0, 0})
	allData = append(allData, []float64{5.9, 3.0, 5.1, 1.8, 1, 0, 0})

	return allData
}
