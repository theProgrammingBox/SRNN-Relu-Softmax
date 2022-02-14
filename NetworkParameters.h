#ifndef NETWORKPARAMETERS_H_
#define NETWORKPARAMETERS_H_

#include "Configurations.h"


class NetworkParameters
{
public:
	float initialMemory[MEMORY_ARRAY_SIZE]{};
	float inputWeight[HIDDEN_ARRAY_SIZE][INTERFACE_ARRAY_SIZE]{};
	float hiddenWeight[HIDDEN_LAYERS - 1][HIDDEN_ARRAY_SIZE][HIDDEN_ARRAY_SIZE]{};
	float outputWeight[INTERFACE_ARRAY_SIZE][HIDDEN_ARRAY_SIZE]{};
	float hiddenBias[HIDDEN_LAYERS][HIDDEN_ARRAY_SIZE]{};
	float outputBias[INTERFACE_ARRAY_SIZE]{};

	void Initialize()
	{
		Reset();
		AddRandomizedMatrix();
		MakeSymmetric();
		AddIdentityMatrix();
		Normalize(1.0 / GetMagnitude());
	}

	void Reset()
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			initialMemory[parentNode] = 0;
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				inputWeight[parentNode][childNode] = 0;
			}
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenWeight[hiddenLayer - 1][parentNode][childNode] = 0;
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				outputWeight[parentNode][childNode] = 0;
			}
		}

		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				hiddenBias[hiddenLayer][parentNode] = 0;
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputBias[parentNode] = 0;
		}
	}

	void AddRandomizedMatrix()
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			initialMemory[parentNode] += (DoubleRand3() * 2 - 1) * HIDDEN_ARRAY_SIZE * STARTING_PARAMETER_RANGE;
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				inputWeight[parentNode][childNode] += (DoubleRand3() * 2 - 1) * HIDDEN_ARRAY_SIZE * STARTING_PARAMETER_RANGE;
			}
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenWeight[hiddenLayer - 1][parentNode][childNode] += (DoubleRand3() * 2 - 1) * HIDDEN_ARRAY_SIZE * STARTING_PARAMETER_RANGE;
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				outputWeight[parentNode][childNode] += (DoubleRand3() * 2 - 1) * HIDDEN_ARRAY_SIZE * STARTING_PARAMETER_RANGE;
			}
		}

		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				hiddenBias[hiddenLayer][parentNode] += (DoubleRand3() * 2 - 1) * HIDDEN_ARRAY_SIZE * STARTING_PARAMETER_RANGE;
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputBias[parentNode] += (DoubleRand3() * 2 - 1) * HIDDEN_ARRAY_SIZE * STARTING_PARAMETER_RANGE;
		}
	}

	void MakeSymmetric()
	{
		int hiddenLayer, parentNode, childNode;
		float average;

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					average = 0.5 * (hiddenWeight[hiddenLayer - 1][parentNode][childNode] + hiddenWeight[hiddenLayer - 1][childNode][parentNode]);
					hiddenWeight[hiddenLayer - 1][parentNode][childNode] = average;
					hiddenWeight[hiddenLayer - 1][childNode][parentNode] = average;
				}
			}
		}
	}

	void AddIdentityMatrix()
	{
		int hiddenLayer, parentNode, childNode;

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenWeight[hiddenLayer - 1][parentNode][childNode] += HIDDEN_ARRAY_SIZE * (parentNode == childNode);
				}
			}
		}
	}

	float GetMagnitude()
	{
		int hiddenLayer, parentNode, childNode;
		float magnitude = initialMemory[0];

		for (parentNode = 1; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			if (initialMemory[parentNode] > magnitude)
			{
				magnitude = initialMemory[parentNode];
			}
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				if (inputWeight[parentNode][childNode] > magnitude)
				{
					magnitude = inputWeight[parentNode][childNode];
				}
			}
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					if (hiddenWeight[hiddenLayer - 1][parentNode][childNode] > magnitude)
					{
						magnitude = hiddenWeight[hiddenLayer - 1][parentNode][childNode];
					}
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				if (outputWeight[parentNode][childNode] > magnitude)
				{
					magnitude = outputWeight[parentNode][childNode];
				}
			}
		}

		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				if (hiddenBias[hiddenLayer][parentNode] > magnitude)
				{
					magnitude = hiddenBias[hiddenLayer][parentNode];
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			if (outputBias[parentNode] > magnitude)
			{
				magnitude = outputBias[parentNode];
			}
		}

		return magnitude;
	}

	void Normalize(float inverseMagnitude)
	{
		int hiddenLayer, parentNode, childNode;

		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			initialMemory[parentNode] *= inverseMagnitude;
		}

		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				inputWeight[parentNode][childNode] *= inverseMagnitude;
			}
		}

		for (hiddenLayer = 1; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					hiddenWeight[hiddenLayer - 1][parentNode][childNode] *= inverseMagnitude;
				}
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				outputWeight[parentNode][childNode] *= inverseMagnitude;
			}
		}

		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				hiddenBias[hiddenLayer][parentNode] *= inverseMagnitude;
			}
		}

		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			outputBias[parentNode] *= inverseMagnitude;
		}
	}

	void ImportNetworkParemeters()
	{
		ifstream netIn("Network.txt");
		string label;
		int interfaceVectorSize, hiddenVectorSize, hiddenLayers;
		int hiddenLayer, parentNode, childNode;

		getline(netIn, label);
		getline(netIn, label);

		getline(netIn, label);
		netIn >> interfaceVectorSize;
		getline(netIn, label);
		getline(netIn, label);
		if (interfaceVectorSize != INTERFACE_ARRAY_SIZE) {
			cout << "interfaceVectorSize mismatch. Press enter to continue.\n";
			cin.get();
			Initialize();
			return;
		}

		getline(netIn, label);
		netIn >> hiddenVectorSize;
		getline(netIn, label);
		getline(netIn, label);
		if (hiddenVectorSize != HIDDEN_ARRAY_SIZE) {
			cout << "hiddenVectorSize mismatch. Press enter to continue.\n";
			cin.get();
			Initialize();
			return;
		}

		getline(netIn, label);
		netIn >> hiddenLayers;
		getline(netIn, label);
		getline(netIn, label);
		if (hiddenLayers != HIDDEN_LAYERS) {
			cout << "hiddenLayers mismatch. Press enter to continue.\n";
			cin.get();
			Initialize();
			return;
		}

		getline(netIn, label);
		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			netIn >> initialMemory[parentNode];
		}
		getline(netIn, label);
		getline(netIn, label);

		getline(netIn, label);
		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				netIn >> inputWeight[parentNode][childNode];
			}
		}
		getline(netIn, label);
		getline(netIn, label);

		getline(netIn, label);
		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS - 1; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					netIn >> hiddenWeight[hiddenLayer][parentNode][childNode];
				}
			}
		}
		getline(netIn, label);
		getline(netIn, label);

		getline(netIn, label);
		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				netIn >> outputWeight[parentNode][childNode];
			}
		}
		getline(netIn, label);
		getline(netIn, label);

		getline(netIn, label);
		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				netIn >> hiddenBias[hiddenLayer][parentNode];
			}
		}
		getline(netIn, label);
		getline(netIn, label);

		getline(netIn, label);
		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			netIn >> outputBias[parentNode];
		}
	}

	string ExportNetworkParemeters()
	{
		ostringstream netOut;
		int hiddenLayer, parentNode, childNode;

		netOut << setprecision(DECIMAL_SIZE) << fixed;
		netOut << "NETWORK PARAMETERS\n\n";

		netOut << "INTERFACE_ARRAY_SIZE:\n";
		netOut << INTERFACE_ARRAY_SIZE << "\n\n";

		netOut << "HIDDEN_ARRAY_SIZE:\n";
		netOut << HIDDEN_ARRAY_SIZE << "\n\n";

		netOut << "HIDDEN_LAYERS:\n";
		netOut << HIDDEN_LAYERS << "\n\n";

		netOut << "initialMemory:\n";
		for (parentNode = 0; parentNode < MEMORY_ARRAY_SIZE; parentNode++)
		{
			netOut  << setw(FLOAT_SIZE) << initialMemory[parentNode];
		}
		netOut << "\n\n";

		netOut << "inputWeights:\n";
		for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < INTERFACE_ARRAY_SIZE; childNode++)
			{
				netOut << setw(FLOAT_SIZE) << inputWeight[parentNode][childNode];
			}
			netOut << endl;
		}
		netOut << endl;

		netOut << "hiddenWeights:\n";
		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS - 1; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
				{
					netOut  << setw(FLOAT_SIZE) << hiddenWeight[hiddenLayer][parentNode][childNode];
				}
				netOut << endl;
			}
			netOut << endl;
		}

		netOut << "outputWeights:\n";
		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			for (childNode = 0; childNode < HIDDEN_ARRAY_SIZE; childNode++)
			{
				netOut  << setw(FLOAT_SIZE) << outputWeight[parentNode][childNode];
			}
			netOut << endl;
		}
		netOut << endl;

		netOut << "hiddenBiases:\n";
		for (hiddenLayer = 0; hiddenLayer < HIDDEN_LAYERS; hiddenLayer++)
		{
			for (parentNode = 0; parentNode < HIDDEN_ARRAY_SIZE; parentNode++)
			{
				netOut  << setw(FLOAT_SIZE) << hiddenBias[hiddenLayer][parentNode];
			}
			netOut << endl;
		}
		netOut << endl;

		netOut << "outputBiases:\n";
		for (parentNode = 0; parentNode < INTERFACE_ARRAY_SIZE; parentNode++)
		{
			netOut  << setw(FLOAT_SIZE) << outputBias[parentNode];
		}
		netOut << "\n\n\n\n";

		netOut << setprecision(6);
		netOut.unsetf(ios::fixed);

		return netOut.str();
	}
};

#endif
