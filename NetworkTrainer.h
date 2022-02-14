#ifndef NETWORKTRAINER_H_
#define NETWORKTRAINER_H_

#include "Environment.h"
#include "NetworkGradients.h"

class NetworkTrainer
{
public:
	NetworkParameters* networkParameters;
	NetworkValues networkValues[SEQUENCE_SIZE];
	NetworkGradients networkGradients[SEQUENCE_SIZE];
	Environment environment;
	float input[INPUT_ARRAY_SIZE]{};
	float output[INPUT_ARRAY_SIZE]{};

	void Initialize(NetworkParameters* networkParameter)
	{
		networkParameters = networkParameter;
		environment.Initialize();
	}

	void Train(int iterations)
	{
		for (int iteration = 0; iteration < iterations; iteration++)
		{
			NetworkValues networkValue;
			networkValue.Initialize(networkParameters);

			for (int i = 0; i < SEQUENCE_SIZE; i++)
			{
				environment.GetInput(input);
				networkValue.ForwardPropagate(input, output);
				environment.ForwardPropagate();
				networkValues[networkValue] = networkValue;
			}
			environment.GetOutput(output);
		}
	}
};

#endif
