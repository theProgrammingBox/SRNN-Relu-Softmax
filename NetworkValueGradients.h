#ifndef NETWORKVALUEGRADIENTS_H_
#define NETWORKVALUEGRADIENTS_H_


#include "NetworkValues.h"

class NetworkValueGradients
{
public:
	NetworkValues* networkValues;
	float memoryGradient[MEMORY_ARRAY_SIZE]{};
	float hiddenSumGradient[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float hiddenReluGradient[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float outputSumGradient[INTERFACE_ARRAY_SIZE]{};
	float outputReluGradient[INTERFACE_ARRAY_SIZE]{};
	float outputSoftMaxGradient[INPUT_ARRAY_SIZE] {};

	void Initialize(NetworkValues* networkValue)
	{
		networkValues = networkValue;
	}

	const void BackPropagate(float* output)
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
		{
			output[parentNode] = 2 * (output[parentNode] - networkValues->outputSoftMax[parentNode]);
		}

		SoftmaxGradient(networkValues->outputRelu, outputSoftMaxGradient);

		for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
		{
			outputReluGradient[parentNode + MEMORY_ARRAY_SIZE] = output[parentNode] * outputSoftMaxGradient[parentNode];
		}

		for (parentNode = MEMORY_ARRAY_SIZE; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputSumGradient[parentNode] = outputReluGradient[parentNode] * LeakyRELUGradient(networkValues->outputSum[parentNode]);
		}

		for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
		{
			hiddenReluGradient[HIDDEN_LAYERS - 1][childNode] = 0;

			for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
			{
				hiddenReluGradient[HIDDEN_LAYERS - 1][childNode] += outputSumGradient[parentNode] * networkValues->networkParameters->outputWeight[parentNode][childNode];
			}
			hiddenSumGradient[HIDDEN_LAYERS - 1][childNode] = hiddenReluGradient[HIDDEN_LAYERS - 1][childNode] * LeakyRELUGradient(networkValues->hiddenSum[HIDDEN_LAYERS - 1][childNode]);
		}

		for (hiddenLayer = HIDDEN_LAYERS - 2; hiddenLayer >= 0; hiddenLayer--)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				hiddenReluGradient[hiddenLayer][childNode] = 0;

				for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
				{
					hiddenReluGradient[hiddenLayer][childNode] += hiddenSumGradient[hiddenLayer + 1][parentNode] * networkValues->networkParameters->hiddenWeight[hiddenLayer][parentNode][childNode];
				}
				hiddenSumGradient[hiddenLayer][childNode] = hiddenReluGradient[hiddenLayer][childNode] * LeakyRELUGradient(networkValues->hiddenSum[hiddenLayer][childNode]);
			}
		}

		for (childNode = 0; childNode < MEMORY_ARRAY_SIZE; childNode++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				memoryGradient[childNode] += hiddenSumGradient[0][parentNode] * networkValues->networkParameters->inputWeight[parentNode][childNode];
			}
		}

		for (childNode = 0; childNode < MEMORY_ARRAY_SIZE; childNode++)
		{
			outputReluGradient[childNode] = memoryGradient[childNode];
		}
	}
};

#endif
