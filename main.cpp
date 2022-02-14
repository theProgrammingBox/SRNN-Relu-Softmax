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
//	ofstream fout;
	NetworkParameters networkParameters;
	NetworkTrainer networkTrainer;

//	fout.open("Network.txt");
	networkParameters.Initialize();
	networkTrainer.Initialize(&networkParameters);
	networkTrainer.Train(100);







//	float input[3] { 0.2, 0.4, -0.2 };
//	float inter[2] {};
//	float output[2] {};
//
//	for (int j = 0; j < 10000; j++)
//	{
//		Softmax(input, inter);
//		SoftmaxGradient(input, output);
//
//		for (int i = 0; i < 2; i++)
//		{
//			input[i + 1] += 2 * ((i == 1) - inter[i]) * output[i];
//		}
//	}
//	cout << endl;
//	for (int i = 0; i < 3; i++)
//	{
//		cout << input[i] << ' ';
//	}
//	cout << endl;
//	for (int i = 0; i < 2; i++)
//	{
//		cout << (i == 1) - inter[i] << ' ';
//	}
//	cout << endl;
//	for (int i = 0; i < 2; i++)
//	{
//		cout << inter[i] << ' ';
//	}

	return 0;
}
