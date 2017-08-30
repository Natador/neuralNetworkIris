// Program structure based on program in this link:
// 	https://www.thepolyglotdeveloper.com/2017/03/parse-csv-data-go-programming-language/
package data

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"fmt"
	"os"
	"strconv"
)

func loadData() [][]float64 {
	irisCsv, err := os.Open("iris_dataset.csv")
	if err != nil {
		log.Fatal(err)
	}

	dataReader := csv.NewReader(bufio.NewReader(irisCsv))

	var data [][]float64

	for index := 0; ; index++ {
		line, err := reader.Read()
		//If at the end of the file, break
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		//Ignore 0th entry of the line, as it contains the line number

		//Convert next 4 entries to float64 and store in temporary array
		tempInputData := make([]float64, 4)
		for i := 1; i < 5; i++ {
			tempInputData[i-1], strErr = strconv.ParseFloat(line[i], 64)
			if strErr != nil {
				log.Fatal(strErr)
			}
		}

		//Switch the strings labeling the target data to an array of one-hot-state float64's
		tempTargetData := make([]float64, 3)
		switch line[5] {
		case "setosa":
			tempTargetData[0] = 1.0
			tempTargetData[1] = 0.0
			tempTargetData[2] = 0.0
		case "versicolor":
			tempTargetData[0] = 0.0
			tempTargetData[1] = 1.0
			tempTargetData[2] = 0.0
		case "virginica":
			tempTargetData[0] = 0.0
			tempTargetData[1] = 0.0
			tempTargetData[2] = 1.0
		default:
			fmt.Println("I don't know about", line[5])
		}
		data = append(data, tempInputData..., tempTargetData...)
	}
}
