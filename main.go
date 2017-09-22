//Author: Nathaniel Cantwell
//Neural network which classifies the famous iris data set

package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

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

const numInputNeuron = 4
const numHiddenNeuron = 7
const numOutputNeuron = 3

func main() {
	fmt.Println("We're making a neural network!")
	rand.Seed(42)

	//Initialize the hard coded data
	data := LoadData()
	trainData, testData := prepData(data)

	//Initialize the network
	var myNetwork Network
	myNetwork.initNetwork(numInputNeuron, numHiddenNeuron, numOutputNeuron)

	learningRate := 0.05
	momentum := 0.05
	maxEpochs := 10000
	maxError := 0.005
	epochsRun := myNetwork.Train(trainData, maxEpochs, maxError, learningRate, momentum)
	trainAccuracy := myNetwork.Test(trainData)
	testAccuracy := myNetwork.Test(testData)
	fmt.Println("Total epochs run:", epochsRun)
	fmt.Printf("\nTraining accuracy: %.2f%%\n", trainAccuracy*100.0)
	fmt.Printf("Testing accuracy: %.2f%%\n\n", testAccuracy*100.0)
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
			net.inputLayer[i].outgoWeights[j] = rand.Float64()*(0.01-0.001) + 0.001
		}

		//Declare array for change in weights
		net.inputLayer[i].outgoDeltas = make([]float64, numHidden+1)

		//Set bias neuron's output to 1.0, and all other outputs to zero
		if i == 0 {
			net.inputLayer[i].outgoVal = 1.0
		} else {
			net.inputLayer[i].outgoVal = 0.0
		}
	}

	//Initialize hidden layer
	for i := range net.hiddenLayer {
		net.hiddenLayer[i].outgoWeights = make([]float64, numOutput+1)

		//Set each weight to a random number in [0.001, 0.01]
		for j := range net.hiddenLayer[i].outgoWeights {
			net.hiddenLayer[i].outgoWeights[j] = rand.Float64()*(0.01-0.001) + 0.001
		}
		//Declare array for change in weights
		net.hiddenLayer[i].outgoDeltas = make([]float64, numOutput+1)

		//Set hidden layer's bias neuron output to 1.0. Set all other outputs to zero.
		if i == 0 {
			net.hiddenLayer[i].outgoVal = 1.0
		} else {
			net.hiddenLayer[i].outgoVal = 0.0
		}
	}

	//Output layer has no outgoing weights, so they are set to nil
	for i := range net.outputLayer {
		net.outputLayer[i].outgoVal = 0
		net.outputLayer[i].outgoWeights = nil
		net.outputLayer[i].outgoDeltas = nil
	}
}

//Train trains the network using the input data
func (net *Network) Train(trainData [][]float64, maxEpochs int, maxError, learnRate, momentum float64) int {
	//Indices for randomly looping through the training data
	indices := initIndices(len(trainData))

	//Main training loop
	var epoch int
	for epoch = 0; epoch < maxEpochs; epoch++ {
		globalError := net.calculateGlobalError(trainData)
		if globalError < maxError {
			break
		}

		//Shuffle the array of indices
		shuffleIndices(indices)

		//Loop through the data randomly and compute the feedForward output, then print the output
		for _, i := range indices {
			//inputData to hold the measurements, targetData to hold the classification
			inputData := trainData[i][:numInputNeuron]
			targetData := trainData[i][numInputNeuron:]

			//Compute the output by feeding-forward the input data
			net.feedForward(inputData)

			//Adjust the weights based on the output
			net.backProp(targetData, learnRate, momentum)
		}
	}
	return epoch
}

//Test compares the output from the neural network to the target output in the test data
func (net *Network) Test(data [][]float64) float64 {
	var numCorrect int

	//Loop through the dataset
	for _, datum := range data {
		//Feed the inputs through the network
		net.feedForward(datum[:numInputNeuron])

		//Compare the outputs to the actual data
		if net.isCorrect(datum[numInputNeuron:]) {
			numCorrect++
		}
	}
	return float64(numCorrect) / float64(len(data))
}

//isCorrect compares the output of the network to the actual value and returns a boolean
func (net *Network) isCorrect(targetData []float64) bool {
	if len(net.outputLayer) != len(targetData) {
		fmt.Println("Error with targetData length!")
		return false
	}

	var maxVal = net.outputLayer[0].outgoVal
	var maxIndex int
	for i := range net.outputLayer {
		if net.outputLayer[i].outgoVal > maxVal {
			maxVal = net.outputLayer[i].outgoVal
			maxIndex = i
		}
	}

	if targetData[maxIndex] == 1.0 {
		return true
	}
	return false
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
			var sum float64
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
			var sum float64
			for j := range net.hiddenLayer {
				sum += net.hiddenLayer[j].outgoVal * net.hiddenLayer[j].outgoWeights[i]
			}

			//Apply the activation function to the weighted sum
			net.outputLayer[i].outgoVal = activationFunction(sum)
		}
	}
}

func (net *Network) backProp(targetData []float64, learnRate, momentum float64) {
	//Calculate output layer error signal
	outputErrorSignal := make([]float64, len(net.outputLayer))
	for i := range net.outputLayer {
		outputErrorSignal[i] = (targetData[i] - net.outputLayer[i].outgoVal) * activationDerivative(net.outputLayer[i].outgoVal)
	}

	//Update hidden layer outgoing weights and deltas
	for i := range net.outputLayer {
		for j := range net.hiddenLayer {
			//Calculate the change in weight
			weightChange := learnRate*outputErrorSignal[i]*net.hiddenLayer[j].outgoVal + momentum*net.hiddenLayer[j].outgoDeltas[i]

			//Apply the change to the hidden weight. Note +=
			net.hiddenLayer[j].outgoWeights[i] += weightChange

			//Update the weight delta for each connection. Note =
			net.hiddenLayer[j].outgoDeltas[i] = weightChange
		}
	}

	//Calculate hidden layer gradients
	hiddenErrorSignal := make([]float64, len(net.hiddenLayer))

	//Compute the error signal for each neuron in the hidden layer
	for j := range net.hiddenLayer {
		var weightSum float64
		for i := range net.outputLayer {
			weightSum += net.hiddenLayer[j].outgoWeights[i] * outputErrorSignal[i]
		}
		hiddenErrorSignal[j] = activationDerivative(net.hiddenLayer[j].outgoVal) * weightSum
	}

	//Update outgoing weights and deltas for input neurons
	for j := range net.hiddenLayer {
		for k := range net.inputLayer {
			//Calculate the change in weight
			weightChange := learnRate*hiddenErrorSignal[j]*net.inputLayer[k].outgoVal + momentum*net.inputLayer[k].outgoDeltas[j]

			//Apply the change to the input weight. Note the +=
			net.inputLayer[k].outgoWeights[j] += weightChange

			//Update the weight delta for each connection. Note =
			net.inputLayer[k].outgoDeltas[j] = weightChange
		}
	}
}

//Wrapper function for changes or optimization
func activationFunction(num float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1.0*num))
}

//Derivative of the activation function used in backpropagation
//	Derivative of sigmoid = sigmoid*(1-sigmoid)
//	We assume this receives the output of the sigmoid already, so it does not apply it again
func activationDerivative(output float64) float64 {
	return output * (1 - output)
}

//calculateGobalError calculates the mean squared error of the network for the given dataset
func (net *Network) calculateGlobalError(data [][]float64) float64 {
	var errSum float64

	for i := range data {
		trainData := data[i][:numInputNeuron] // Change to accomodate different data-length
		targetData := data[i][numInputNeuron:]
		net.feedForward(trainData)

		for j := range net.outputLayer {
			diff := targetData[j] - net.outputLayer[j].outgoVal
			errSum += diff * diff
		}
	}

	return errSum / float64((len(net.outputLayer) * len(data)))
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

//prepData randomizes and splits the iris data into 80% for training and 20% for testing.
func prepData(data [][]float64) (trainData, testData [][]float64) {
	trainSize := int(float64(len(data)) * 0.8)
	testSize := len(data) - trainSize
	trainData = make([][]float64, trainSize)
	testData = make([][]float64, testSize)

	randomIndices := initIndices(len(data))
	randomIndices = shuffleIndices(randomIndices)

	//Randomly index the data to split into training and testing data
	for i := 0; i < trainSize; i++ {
		trainData[i] = data[randomIndices[i]]
	}

	for i := 0; i < testSize; i++ {
		testData[i] = data[randomIndices[i+trainSize]]
	}

	return trainData, testData
}

//LoadData loads and parses the iris data from a csv file in ./data
func LoadData() [][]float64 {
	irisCsv, err := os.Open("data/iris_dataset.csv")
	if err != nil {
		log.Fatal(err)
	}

	dataReader := csv.NewReader(bufio.NewReader(irisCsv))

	var data [][]float64

	for {
		line, err := dataReader.Read()
		//If at the end of the file, break
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		//Ignore 0th entry of the line, as it contains the line number

		//Convert next 4 entries to float64 and store in temporary array
		tempData := make([]float64, 7)
		for i := 1; i < 5; i++ {
			var strErr error
			tempData[i-1], strErr = strconv.ParseFloat(line[i], 64)
			if strErr != nil {
				log.Fatal(strErr)
			}
		}

		//Switch the strings labeling the target data to an array of one-hot-state float64's
		switch line[5] {
		case "setosa":
			tempData[4] = 1.0
			tempData[5] = 0.0
			tempData[6] = 0.0
		case "versicolor":
			tempData[4] = 0.0
			tempData[5] = 1.0
			tempData[6] = 0.0
		case "virginica":
			tempData[4] = 0.0
			tempData[5] = 0.0
			tempData[6] = 1.0
		default:
			fmt.Println("I don't know about", line[5])
		}
		data = append(data, tempData)
	}
	return data
}
