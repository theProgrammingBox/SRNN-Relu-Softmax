#ifndef NETWORKPARAMETERGRADIENTS_H_
#define NETWORKPARAMETERGRADIENTS_H_

#include "NetworkValueGradients.h"

class  NetworkParameterGradients
{
public:
	NetworkValueGradients* networkValueGradients;
	float initialMemoryGradient[MEMORY_ARRAY_SIZE]{};
	float inputWeightGradient[HIDDEN_ARRAY_SIZE][INTERFACE_ARRAY_SIZE]{};
	float hiddenWeightGradient[HIDDEN_LAYERS - 1][HIDDEN_ARRAY_SIZE][HIDDEN_ARRAY_SIZE]{};
	float outputWeightGradient[INTERFACE_ARRAY_SIZE][HIDDEN_ARRAY_SIZE]{};
	float hiddenBiasGradient[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float outputBiasGradient[INTERFACE_ARRAY_SIZE]{};

	void Initialize(NetworkValueGradients* networkValueGradient)
	{
		networkValueGradients = networkValueGradient;
	}

	void ForwardPropagate() // apply changes to parameter
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkValueGradients->networkValues->networkParameters->initialMemory[parentNode] += initialMemoryGradient[parentNode] * LEARNING_RATE / BATCH_SIZE; // optimize
			initialMemoryGradient[parentNode] = 0;
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				networkValueGradients->networkValues->networkParameters->inputWeight[parentNode][childNode] += inputWeightGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				inputWeightGradient[parentNode][childNode] = 0;
			}
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					networkValueGradients->networkValues->networkParameters->hiddenWeight[hiddenLayer - 1][parentNode][childNode] += hiddenWeightGradient[hiddenLayer - 1][parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
					hiddenWeightGradient[hiddenLayer - 1][parentNode][childNode] = 0;
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				networkValueGradients->networkValues->networkParameters->outputWeight[parentNode][childNode] += outputWeightGradient[parentNode][childNode] * LEARNING_RATE / BATCH_SIZE;
				outputWeightGradient[parentNode][childNode] = 0;
			}
		}

		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				networkValueGradients->networkValues->networkParameters->hiddenBias[hiddenLayer][parentNode] += hiddenBiasGradient[hiddenLayer][parentNode] * LEARNING_RATE / BATCH_SIZE;
				hiddenBiasGradient[hiddenLayer][parentNode] = 0;
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			networkValueGradients->networkValues->networkParameters->outputBias[parentNode] += outputBiasGradient[parentNode] * LEARNING_RATE / BATCH_SIZE;
			outputBiasGradient[parentNode] = 0;
		}
	}

	void BackPropagate(float* input) // compound changes
	{
		int hiddenLayer, parentNode, childNode;

		for (childNode = 0; childNode < MEMORY_ARRAY_SIZE; childNode++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				initialMemoryGradient[childNode] += networkValueGradients->hiddenSumGradient[0][parentNode] * networkValueGradients->networkValues->networkParameters->inputWeight[parentNode][childNode];
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputBiasGradient[parentNode] += networkValueGradients->outputSumGradient[parentNode];

			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				outputWeightGradient[parentNode][childNode] += networkValueGradients->outputSumGradient[parentNode] * networkValueGradients->networkValues->hiddenRelu[HIDDEN_LAYERS - 1][childNode];
			}
		}

		for (hiddenLayer = HIDDEN_LAYERS - 1; hiddenLayer > 0; hiddenLayer--)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				hiddenBiasGradient[hiddenLayer][parentNode] += networkValueGradients->hiddenSumGradient[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenWeightGradient[hiddenLayer - 1][parentNode][childNode] += networkValueGradients->hiddenSumGradient[hiddenLayer][parentNode] * networkValueGradients->networkValues->hiddenRelu[hiddenLayer - 1][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			hiddenBiasGradient[0][parentNode] += networkValueGradients->hiddenSumGradient[0][parentNode];

			for (childNode = 0; childNode < MEMORY_ARRAY_SIZE; childNode++)
			{
				inputWeightGradient[parentNode][childNode] += networkValueGradients->hiddenSumGradient[0][parentNode] * networkValueGradients->networkValues->memory[childNode];
			}

			for (childNode = MEMORY_ARRAY_SIZE; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				inputWeightGradient[parentNode][childNode] += networkValueGradients->hiddenSumGradient[0][parentNode] * input[childNode - MEMORY_ARRAY_SIZE];
			}
		}
	}
};


#endif
