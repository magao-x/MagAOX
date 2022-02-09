#include "stopwatch.h"
#include <iostream>

namespace DDSPC
{

void Timer::tick(){
	t1 = std::chrono::high_resolution_clock::now();
}

void Timer::tock(){
	t2 = std::chrono::high_resolution_clock::now();
}

void Timer::elapsed(int num_frames, bool in_us){
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	if(in_us){
		std::cout << "It took me " << time_span.count() * 1000000 / num_frames << " us on average." <<std::endl;
	}else{
		std::cout << "It took me " << time_span.count() * 1000 / num_frames << " ms on average." <<std::endl;
	}
}

double Timer::get_elapsed(){
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	return time_span.count() * 1000;
}

}