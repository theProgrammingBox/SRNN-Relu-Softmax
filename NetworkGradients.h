#ifndef NETWORKGRADIENTS_H_
#define NETWORKGRADIENTS_H_

#include "NetworkValues.h"

class NetworkGradients
{
public:
	NetworkValues* networkValues;
	float memoryGradient[MEMORY_ARRAY_SIZE]{};
	float hiddenSumGradient[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float hiddenReluGradient[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float outputSumGradient[INTERFACE_ARRAY_SIZE]{};
	float outputReluGradient[INTERFACE_ARRAY_SIZE]{};
	float outputSoftMaxGradient[INPUT_ARRAY_SIZE] {};

	void Initialize(NetworkValues* networkValues_)
	{
		networkValues = networkValues_;
	}

	const void BackPropagate(float* input, float* output)
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

		for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
		{
			outputReluGradient[parentNode + MEMORY_ARRAY_SIZE] = output[parentNode] * outputSoftMaxGradient[parentNode];
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputSumGradient[parentNode] = outputReluGradient[parentNode] * LeakyRELUGradient(networkValues->outputSum[parentNode]);

			hiddenReluGradient[HIDDEN_LAYERS - 1][parentNode] = 0; // fix, resets the gradient on each parent node
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				hiddenReluGradient[HIDDEN_LAYERS - 1][parentNode] += outputSumGradient[parentNode] * networkValues->networkParameters->outputWeight[parentNode][childNode];
			}
		}

		for (hiddenLayer = HIDDEN_LAYERS - 1; hiddenLayer > 0; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
			{
				hiddenSumGradient[hiddenLayer][parentNode] = hiddenReluGradient[hiddenLayer][parentNode] * LeakyRELUGradient(networkValues->hiddenSum[hiddenLayer][parentNode]);
				hiddenReluGradient[hiddenLayer - 1][parentNode] = 0; // fix, resets the gradient on each parent node
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenReluGradient[hiddenLayer - 1][parentNode] += hiddenSumGradient[hiddenLayer][parentNode] * networkValues->networkParameters->hiddenWeight[hiddenLayer - 1][parentNode][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			memoryGradient[parentNode] = hiddenReluGradient[0][parentNode] * LeakyRELUGradient(networkValues->outputSum[parentNode]);
		}
	}
};

#endif
