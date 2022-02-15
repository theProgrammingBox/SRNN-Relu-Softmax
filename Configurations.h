#ifndef CONFIGURATIONS_H_
#define CONFIGURATIONS_H_

#include "RandomAndTime.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>

using std::ios;
using std::exp;
using std::log;
using std::cin;
using std::setw;
using std::cout;
using std::endl;
using std::left;
using std::right;
using std::fixed;
using std::string;
using std::ifstream;
using std::ofstream;
using std::setprecision;
using std::ostringstream;

/*Formating Parameters*/
const int DECIMAL_SIZE = 8;
const int FLOAT_SIZE = 14;

/*Training Parameters*/
const float LEARNING_RATE = 0.001;
const int BATCH_SIZE = 10;
const int SEQUENCE_SIZE = 3;

/*Network Parameters*/
const float STARTING_PARAMETER_RANGE = 0.1;	// defines the +- range of starting parameters
const int INPUT_ARRAY_SIZE = 2;			// used for both x and output (option probabilities), (x >= 2) cuz softmax needs at least 2 options
const int MEMORY_ARRAY_SIZE = 10;			// number of memory values 'passed' to next iteration, (x >= 1)
const int INTERFACE_ARRAY_SIZE = MEMORY_ARRAY_SIZE + INPUT_ARRAY_SIZE;	// x and output size of network
const int HIDDEN_ARRAY_SIZE = 10;			// nodes in each hidden layer, (x >= 1)
const int HIDDEN_LAYERS = 4;				// (x >= 2)

const float LeakyRELU(float x) // Leaky Rectified Linear Unit
{
	if (x < 0)
	{
		return x * 0.1;
	}
	return x;
}

const float LeakyRELUGradient(float x)
{
	if (x < 0)
	{
		return 0.1;
	}
	return 1;
}

const float LLU(float x) // Logrithmic Linear Unit
{
	if (x < 0)
	{
		return -log(1 - x);
	}
	return x;
}

const float LLUGradient(float x)
{
	if (x < 0)
	{
		return 1 / (1 - x);
	}
	return 1;
}

const void Softmax(float* input, float* output)
{
	int parentNode;
	float largestValue = input[MEMORY_ARRAY_SIZE];
	float total = 0;

	for (parentNode = MEMORY_ARRAY_SIZE + 1; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
	{
		if (largestValue < input[parentNode])
		{
			largestValue = input[parentNode];
		}
	}

	for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] = exp(input[parentNode + MEMORY_ARRAY_SIZE] - largestValue);
		total += output[parentNode];
	}

	for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] /= total;
	}
}

const void SoftmaxGradient(float* input, float* output)
{
	int parentNode, childNode;
	float numerator[INPUT_ARRAY_SIZE]{};
	float largestValue = input[MEMORY_ARRAY_SIZE];
	float total = 0;

	for (parentNode = MEMORY_ARRAY_SIZE + 1; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
	{
		if (largestValue < input[parentNode])
		{
			largestValue = input[parentNode];
		}
	}

	for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] = input[parentNode + MEMORY_ARRAY_SIZE] - largestValue;
		total += exp(output[parentNode]);
	}

	total *= total;

	for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
	{
		for (childNode = 0; childNode < INPUT_ARRAY_SIZE; childNode++)
		{
			if (parentNode != childNode) // optimize
			{
				numerator[parentNode] += exp(output[parentNode] + output[childNode]);
			}
		}
	}

	for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
	{
		output[parentNode] = numerator[parentNode] / total;
	}
}

#endif
