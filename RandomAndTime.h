#ifndef RANDOMANDTIME_H_
#define RANDOMANDTIME_H_

#include <chrono>

using std::chrono::seconds;
using std::chrono::milliseconds;
using std::chrono::microseconds;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

const unsigned int second = duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count();
const unsigned int millisecond = duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
const unsigned int microsecond = duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
const unsigned int nanosecond = duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();

int m_z = second ^ nanosecond;
int m_w = millisecond ^ microsecond;
int state = m_z ^ m_w;
bool randOption = false;

auto start = high_resolution_clock::now();

const unsigned int UIntRand1()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);

	return (m_z << 16) + m_w;
}

const int IntRand1()
{
	return UIntRand1();
}

const double DoubleRand1()	// 0 through 1
{
	return UIntRand1() * 2.328306435454494e-10;
}

const unsigned int UIntRand2()
{
	state = (state ^ 2747636419u) * 2654435769u;
	state = (state ^ (state >> 16u)) * 2654435769u;
	state = (state ^ (state >> 16u)) * 2654435769u;

	return state;
}

const int IntRand2()
{
	return UIntRand2();
}

const double DoubleRand2()	// 0 through 1
{
	return UIntRand2() / 4294967295.0;
}

const unsigned int UIntRand3()
{
	if (randOption ^= 0)
	{
		return UIntRand1();
	}
	return UIntRand2();
}

const int IntRand3()
{
	return UIntRand3();
}

const double DoubleRand3()	// 0 through 1
{
	if (randOption ^= 0)
	{
		return DoubleRand1();
	}
	return DoubleRand2();
}

const void StartTimer()
{
	start = high_resolution_clock::now();
}

const double StopTimer()
{
	return duration_cast<microseconds>(high_resolution_clock::now() - start).count() * 0.000001;
}

#endif
