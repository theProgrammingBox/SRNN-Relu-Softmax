#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "Configurations.h"

class Environment
{
public:
	bool state;

	void Initialize()
	{
		state = UIntRand3();
	}

	void ForwardPropagate()
	{
		state ^= 1;
	}

	void GetInput(float* input)
	{
		int iteration;

		for (iteration = 0; iteration < INPUT_ARRAY_SIZE; iteration++)
		{
			input[iteration] = state == iteration;
		}
	}

	void GetOutput(float* output)
	{
		int iteration;

		for (iteration = 0; iteration < INPUT_ARRAY_SIZE; iteration++)
		{
			output[iteration] = state != iteration;
		}
	}
};

#endif
