#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "Configurations.h"

class Environment
{
public:
	const int memoryLag = 2;
	bool memory[2]{};
	int memoryIndex;

	void Initialize()
	{
		memoryIndex = 0;
		ForwardPropagate();
	}

	void ForwardPropagate() // will use network output if interactive
	{
		bool randVal = UIntRand3() % 2;

		memoryIndex = memoryIndex + 1 == memoryLag? 0 : memoryIndex + 1;
		memory[memoryIndex] = randVal;
	}

	void GetInput(float* input)
	{
		for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
		{
			input[i] = memory[memoryIndex] == i;
		}
	}

	void GetOutput(float* output)
	{
		for (int i = 0; i < INPUT_ARRAY_SIZE; i++)
		{
			output[i] = memory[memoryIndex + 1 == memoryLag? 0 : memoryIndex + 1] == i;
		}
	}
};

#endif
