#include "NetworkTrainer.h"

// 1. Define a NetworkParameters
// 2. Define a NetworkValues
// 2. Define an Environment
// 3. Initialize or Import the Network parameters
// 4. Initialize NetworkValues with the Network parameters
// 5. Initialize the Environment
// 6. ForwardPropagate the Environment
// 7. ForwardPropagate the NetworkValues with the output of the Environment

int main()
{
	ofstream fout;
	NetworkParameters networkParameters;
	NetworkTrainer networkTrainer;

	fout.open("Network.txt");
	networkParameters.Initialize();
	networkTrainer.Initialize(&networkParameters);
	networkTrainer.Train(10000);
	fout << networkParameters.ExportNetworkParemeters();

	while(true)
	{
		networkTrainer.Train(100000);
		fout << networkParameters.ExportNetworkParemeters();
	}

	return 0;
}
