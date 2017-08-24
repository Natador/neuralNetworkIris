//Author: Nathaniel Cantwell
//Neural network which classifies the iris data set

package main

import "fmt"
import "math"
import "math/rand"
import "time"

//Neuron struct which acts as a node in the network
type Neuron struct {
	outgoVal     float64   //Outgoing signal value
	outgoWeights []float64 //Outgoing weight values
	outgoDeltas  []float64 //Change in outgoing weight values
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
	data := loadData()
	//trainData, testData := prepData(data)
	trainData, _ := prepData(data)

	//Initialize the network
	rand.Seed(time.Now().Unix())
	var myNetwork Network
	myNetwork.initNetwork(4, 7, 3)

	learningRate := 0.01
	momentum := 0.05
	maxEpochs := 1
	myNetwork.Train(trainData, maxEpochs, learningRate, momentum)
	//testAccuracy := myNetwork.Test(testData)
	//fmt.Println("Test accuracy:", testAccuracy)
}

//****** Network functions ******//

//initNetwork initializes and populates the network with neurons
// whose weights are set randomly
func (net *Network) initNetwork(numInputs, numHidden, numOutput int) {
	//Allocate memory for neurons
	// + 1 for the bias neurons except for the output
	net.inputLayer = make([]Neuron, numInputs+1)
	net.hiddenLayer = make([]Neuron, numHidden+1)
	net.outputLayer = make([]Neuron, numOutput)

	//Initialize input layer
	for i := range net.inputLayer {
		net.inputLayer[i].outgoWeights = make([]float64, numHidden+1)

		//Set each weight to a random number in [0.001, 0.01]
		for j := range net.inputLayer[i].outgoWeights {
			//If the neuron is a bias neuron, set its output to 1.0
			if j == len(net.inputLayer[i].outgoWeights) {
				net.inputLayer[i].outgoVal = 1.0
			}
			net.inputLayer[i].outgoWeights[j] = rand.Float64()*(0.01-0.001) + 0.001
		}

		//Declare array for change in weights
		net.inputLayer[i].outgoDeltas = make([]float64, numHidden+1)

		//Set the input layer's outgoVal to zero
		net.inputLayer[i].outgoVal = 0
	}

	//Initialize hidden layer
	for i := range net.hiddenLayer {
		net.hiddenLayer[i].outgoWeights = make([]float64, numOutput+1)

		//Set each weight to a random number in [0.001, 0.01]
		for j := range net.hiddenLayer[i].outgoWeights {
			//If the neuron is a bias neuron, set its output to 1.0
			if j == len(net.hiddenLayer[i].outgoWeights) {
				net.hiddenLayer[i].outgoVal = 1.0
			}
			net.hiddenLayer[i].outgoWeights[j] = rand.Float64()*(0.01-0.001) + 0.001
		}
		//Declare array for change in weights
		net.hiddenLayer[i].outgoDeltas = make([]float64, numOutput+1)

		//Set hidden layer's outputs to zero
		net.hiddenLayer[i].outgoVal = 0
	}

	//Output layer has no outgoing weights, so they are set to nil
	for i := range net.outputLayer {
		net.outputLayer[i].outgoVal = 0
		net.outputLayer[i].outgoWeights = nil
		net.outputLayer[i].outgoDeltas = nil
	}
}

//train trains the network using the input data
func (net *Network) Train(trainData [][]float64, maxEpochs int, learnRate, momentum float64) {
	//Indices for randomly looping through the training data
	indices := initIndices(len(trainData))

	//Main training loop
	for epoch := 0; epoch < maxEpochs; epoch++ {
		//	if net.calculateError() > 0.005 {
		//		break
		//	}

		//Shuffle the array of indices
		shuffleIndices(indices)

		//Loop through the data randomly and compute the feedForward output, then print the output
		for _, i := range indices {
			//inputData to hold the measurements, targetData to hold the classification
			inputData := trainData[i][:4]
			//targetData := trainData[i][4:]

			//Compute the output by feeding-forward the input data
			net.feedForward(inputData)
			fmt.Println()
			for j := range net.outputLayer {
				fmt.Println(net.outputLayer[j].outgoVal)
			}

			//net.backProp(targetData, learnRate, momentum)
		}
	}

	//	backProp(targetData, learnRate, momentum) updates the weights
}

//feedForward computes the output for each neuron in each layer
func (net *Network) feedForward(inputs []float64) {
	//Error checking the dimension of the input vector and the number of input neurons
	if len(inputs) != len(net.inputLayer)-1 {
		fmt.Println("Error with input vector dimensions!")
	} else {
		//Set outgoing values of input layer to the input data, excluding the bias neuron
		for i := 0; i < len(net.inputLayer)-1; i++ {
			net.inputLayer[i].outgoVal = inputs[i]
		}

		//compute hidden layer values

		//Compute the weighted sum of input values (excludes bias neuron)
		for i := 0; i < len(net.hiddenLayer)-1; i++ {
			var sum float64 = 0
			for j := range net.inputLayer {
				//Weights accessed by i becuase i determines which connection is made to the next layer
				//	Includes the weighted sum of the bias neuron's output
				sum += net.inputLayer[j].outgoVal * net.inputLayer[j].outgoWeights[i]
			}

			//Apply the activation function to the weighted sum
			net.hiddenLayer[i].outgoVal = activationFunction(sum)
		}

		//Compute output values

		//Compute the weighted sum of hidden layer outputs and apply activation function.
		//	No bias neurons in the output layer, so we can loop directly through it
		for i := range net.outputLayer {
			var sum float64 = 0
			for j := range net.hiddenLayer {
				sum += net.hiddenLayer[j].outgoVal * net.hiddenLayer[j].outgoWeights[i]
			}

			//Apply the activation function to the weighted sum
			net.outputLayer[i].outgoVal = activationFunction(sum)
		}
	}
}

func (net *Network) backProp(targetData []float64, learnRate, momentum float64) {
	//Calculate output layer gradients

	//Calculate hidden layer gradients

	//Update weights for all connections

	//Update weight deltas for all connections
}

//Wrapper function for changes or optimization
func activationFunction(num float64) float64 {
	return math.Tanh(num)
}

//Derivative of the activation function used in backpropagation
func activationDerivatiev(num float64) float64 {
	j := math.Tanh(num)
	return 1.0 - j*j
}

//****** Data functions ******//

//initIndices creates a slice of the given length with values stored as their indices
func initIndices(length int) []int {
	indices := make([]int, length)
	for i := range indices {
		indices[i] = i
	}

	return indices
}

//shuffleIndices shuffles a given array of indices
func shuffleIndices(indices []int) []int {
	//Shuffle the array using the Fisher shuffle algorithm
	for i, j := len(indices)-1, 0; i > 0; i-- {
		//random integer in interval [0, i]
		j = rand.Intn(i)

		swap := indices[i]
		indices[i] = indices[j]
		indices[j] = swap
	}

	return indices
}

//prepData splits the training data into 80% for training and 20% for testing
func prepData(data [][]float64) (trainData, testData [][]float64) {
	trainSize := int(float64(len(data)) * 0.8)
	trainData = data[:trainSize]
	testData = data[trainSize:]
	return trainData, testData
}

//initData returns a [][]float64 containing the hard coded iris data
func loadData() [][]float64 {
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
