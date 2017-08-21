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
	var myNetwork Network
	myNetwork.initNetwork(4, 7, 3)
}

//initNetwork initializes and populates the network with neurons
// whose weights are set randomly
func (net *Network) initNetwork(numInputs, numHidden, numOutput int) {
	//Fill in details here
}
