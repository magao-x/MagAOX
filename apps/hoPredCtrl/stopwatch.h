#include <chrono>

namespace DDSPC
{

class Timer{
private:
	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::high_resolution_clock::time_point t2;
public:
	Timer(){};

	void tick();
	void tock();
	void elapsed(int, bool in_us=false);
	double get_elapsed();
};

}