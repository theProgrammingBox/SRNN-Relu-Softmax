#ifndef NETWORKVALUES_H_
#define NETWORKVALUES_H_

#include "NetworkParameters.h"

class NetworkValues
{
public:
	NetworkParameters* networkParameters;
	float memory[MEMORY_ARRAY_SIZE]{};
	float hiddenSum[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float hiddenRelu[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float outputSum[INTERFACE_ARRAY_SIZE]{};
	float outputRelu[INTERFACE_ARRAY_SIZE]{};
	float outputSoftMax[INPUT_ARRAY_SIZE]{};

	void Initialize(NetworkParameters* networkParameter)
	{
		networkParameters = networkParameter;
		ResetMemory();
	}

	void ResetMemory()
	{
		int parentNode;

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			memory[parentNode] = networkParameters->initialMemory[parentNode];
		}
	}

	void ForwardPropagate(float* input, float* output)
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			hiddenSum[0][parentNode] = networkParameters->hiddenBias[0][parentNode];

			for (childNode = 0; childNode < MEMORY_ARRAY_SIZE; childNode++)
			{
				hiddenSum[0][parentNode] += networkParameters->inputWeight[parentNode][childNode] * memory[childNode];
			}

			for (childNode = MEMORY_ARRAY_SIZE; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				hiddenSum[0][parentNode] += networkParameters->inputWeight[parentNode][childNode] * input[childNode - MEMORY_ARRAY_SIZE];
			}

			hiddenRelu[0][parentNode] = LeakyRELU(hiddenSum[0][parentNode]);
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				hiddenSum[hiddenLayer][parentNode] = networkParameters->hiddenBias[hiddenLayer][parentNode];

				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenSum[hiddenLayer][parentNode] += networkParameters->hiddenWeight[hiddenLayer - 1][parentNode][childNode] * hiddenRelu[hiddenLayer - 1][childNode];
				}

				hiddenRelu[hiddenLayer][parentNode] = LeakyRELU(hiddenSum[hiddenLayer][parentNode]);
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputSum[parentNode] = networkParameters->outputBias[parentNode];

			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				outputSum[parentNode] += networkParameters->outputWeight[parentNode][childNode] * hiddenRelu[HIDDEN_LAYERS - 1][childNode];
			}

			outputRelu[parentNode] = LeakyRELU(outputSum[parentNode]);
		}

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			memory[parentNode] += outputRelu[parentNode];
		}

		Softmax(outputRelu, outputSoftMax); // double check correct shifting

		for (parentNode = 0; parentNode < INPUT_ARRAY_SIZE; parentNode++)
		{
			output[parentNode] = outputSoftMax[parentNode];
		}
	}

	string ExportNetworkParameters()
	{
		return networkParameters->ExportNetworkParemeters();
	}

	string ExportNetworkValues()
	{
		ostringstream netOut;
		int hiddenLayer, node;

		netOut << "NETWORK VALUES\n\n";
		netOut << setprecision(DECIMAL_SIZE) << fixed;

		netOut << "memory:\n";
		for (node = 0; node < MEMORY_ARRAY_SIZE; node++)
		{
			netOut << setw(FLOAT_SIZE) << memory[node];
		}
		netOut << "\n\n";

		netOut << "hiddenSum:\n";
		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (node = 0; node < HIDDEN_ARRAY_SIZE; node++)
			{
				netOut << setw(FLOAT_SIZE) << hiddenSum[hiddenLayer][node];
			}
			netOut << endl;
		}
		netOut << endl;

		netOut << "hiddenRelu:\n";
		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (node = 0; node < HIDDEN_ARRAY_SIZE; node++)
			{
				netOut << setw(FLOAT_SIZE) << hiddenRelu[hiddenLayer][node];
			}
			netOut << endl;
		}
		netOut << endl;

		netOut << "outputSum:\n";
		for (node = 0; node < INTERFACE_ARRAY_SIZE; node++)
		{
			netOut << setw(FLOAT_SIZE) << outputSum[node];
		}
		netOut << "\n\n";

		netOut << "outputRelu:\n";
		for (node = 0; node < INTERFACE_ARRAY_SIZE; node++)
		{
			netOut << setw(FLOAT_SIZE) << outputRelu[node];
		}
		netOut << "\n\n";

		netOut << "outputSoftMax:\n";
		for (node = 0; node < INPUT_ARRAY_SIZE; node++)
		{
			netOut << setw(FLOAT_SIZE) << outputSoftMax[node];
		}
		netOut << "\n\n\n\n";

		netOut << setprecision(6);
		netOut.unsetf(ios::fixed);

		return netOut.str();
	}
};

#endif
