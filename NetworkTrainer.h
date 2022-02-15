#ifndef NETWORKTRAINER_H_
#define NETWORKTRAINER_H_

#include "Environment.h"
#include "NetworkParameterGradients.h"

class NetworkTrainer
{
public:
	NetworkParameters* networkParameters;
	NetworkValues networkValuesArray[SEQUENCE_SIZE];
	float input[SEQUENCE_SIZE][INPUT_ARRAY_SIZE]{};
	float output[SEQUENCE_SIZE][INPUT_ARRAY_SIZE]{};
	float outputSee[SEQUENCE_SIZE][INPUT_ARRAY_SIZE]{};
	float networkOutput[SEQUENCE_SIZE][INPUT_ARRAY_SIZE]{};			// not used in training, only a placeholder for the function

	void Initialize(NetworkParameters* networkParameter)
	{
		networkParameters = networkParameter;
	}

	void Train(int iterations)
	{
		NetworkParameterGradients networkParameterGradients;
		int iteration, batch, sequence;

		for (iteration = 0; iteration < iterations; iteration++)
		{
			for (batch = 0; batch < BATCH_SIZE; batch++)
			{
				Environment environment;
				NetworkValues networkValues;
				NetworkValueGradients networkValueGradients;

				environment.Initialize();
				networkValues.Initialize(networkParameters);
				networkParameterGradients.Initialize(&networkValueGradients);

				for (sequence = 0; sequence < SEQUENCE_SIZE; sequence++)
				{
					environment.GetInput(input[sequence]);
					environment.GetOutput(output[sequence]);
					environment.GetOutput(outputSee[sequence]);
					networkValues.ForwardPropagate(input[sequence], networkOutput[sequence]);
					networkValuesArray[sequence] = networkValues;
					environment.ForwardPropagate();
				}

				for (sequence = SEQUENCE_SIZE - 1; sequence >= 0; sequence--)
				{
					networkValueGradients.Initialize(&networkValuesArray[sequence]);
					networkValueGradients.BackPropagate(output[sequence]);
					networkParameterGradients.BackPropagate(input[sequence]);
				}
			}

			networkParameterGradients.ForwardPropagate();
		}

		cout << left;
		cout << setw(20) << "Input" << setw(20) << "Network Output" << setw(20) << "Output";
		cout << endl;
		for (sequence = 0; sequence < SEQUENCE_SIZE; sequence++)
		{
			for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
			{
				cout << setw(10) << input[sequence][i];
			}
			for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
			{
				cout << setw(10) << networkOutput[sequence][i];
			}
			for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
			{
				cout << setw(10) << outputSee[sequence][i] << ' ';
			}
			cout << endl;
		}
		cout << endl;
		cout << right;
	}
};

#endif
