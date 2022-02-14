#ifndef NETWORKPARAMETERGRADIENTS_H_
#define NETWORKPARAMETERGRADIENTS_H_

#include "NetworkValueGradients.h"

class  NetworkParameterGradients
{
public:
	NetworkValueGradients* networkValueGradients;
	float inputWeightGradient[HIDDEN_ARRAY_SIZE][INTERFACE_ARRAY_SIZE]{};
	float hiddenWeightGradient[HIDDEN_LAYERS - 1][HIDDEN_ARRAY_SIZE][HIDDEN_ARRAY_SIZE]{};
	float outputWeightGradient[INTERFACE_ARRAY_SIZE][HIDDEN_ARRAY_SIZE]{};
	float hiddenBiasGradient[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float outputBiasGradient[INTERFACE_ARRAY_SIZE]{};

	void ForwardPropagate() // apply changes to parameter
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			networkValueGradients->networkValues->networkParameters->initialMemory[parentNode] += networkValueGradients->memoryGradient[parentNode];
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				networkValueGradients->networkValues->networkParameters->inputWeight[parentNode][childNode] += inputWeightGradient[parentNode][childNode];
			}
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					networkValueGradients->networkValues->networkParameters->hiddenWeight[hiddenLayer - 1][parentNode][childNode] += hiddenWeightGradient[hiddenLayer - 1][parentNode][childNode];
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				networkValueGradients->networkValues->networkParameters->outputWeight[parentNode][childNode] += outputWeightGradient[parentNode][childNode];
			}
		}

		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				networkValueGradients->networkValues->networkParameters->hiddenBias[hiddenLayer][parentNode] += hiddenBiasGradient[hiddenLayer][parentNode];
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			networkValueGradients->networkValues->networkParameters->outputBias[parentNode] += outputBiasGradient[parentNode];
		}
	}

	void BackPropagate() // compound changes
	{
		//
	}
};


#endif
