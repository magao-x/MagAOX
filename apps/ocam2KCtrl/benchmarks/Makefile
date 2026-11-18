## \file Makefile
## \brief Build the hardware-free OCAM descramble benchmark independently of the application.

CXX ?= c++
CC ?= cc
CXXFLAGS ?= -O3 -march=native -ffast-math -std=c++20 -Wall -Wextra
CFLAGS ?= -O3 -march=native -ffast-math
SDK_DEFINES = -DOCAMSDK_VERSION=\"2\" -DOCAMSDK_BUILD=\"0\"
THREAD_FLAGS = -pthread
TEST_CPUS ?= 2 4

.PHONY: all check clean
all: ocamDescramble_bench

ocamDescramble_bench: ocamDescramble_bench.cpp ocamDescrambleWorker.hpp ../ocamDescrambleParallel.hpp ../ocamDescrambler.hpp ../../../libMagAOX/common/streamCopy.hpp ocam2_sdk.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) -DOCAM_BENCH_CXXFLAGS='"$(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS)"' -DOCAM_BENCH_CFLAGS='"$(CFLAGS)"' $< ocam2_sdk.o $(LDFLAGS) $(LDLIBS) -o $@

ocamDescrambleWorker_test: ocamDescrambleWorker_test.cpp ocamDescrambleWorker.hpp ../ocamDescrambleParallel.hpp ../ocamDescrambler.hpp ../tests/ocamDescrambleFixture.hpp ocam2_sdk.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< ocam2_sdk.o $(LDFLAGS) $(LDLIBS) -o $@

ocamDescrambleParallel_test: ocamDescrambleParallel_test.cpp ../ocamDescrambleParallel.hpp ../ocamDescrambler.hpp ../tests/ocamDescrambleFixture.hpp ocam2_sdk.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< ocam2_sdk.o $(LDFLAGS) $(LDLIBS) -o $@

check: ocamDescrambleWorker_test ocamDescrambleParallel_test
	./ocamDescrambleWorker_test $(TEST_CPUS)
	./ocamDescrambleParallel_test

ocam2_sdk.o: ../fli/ocam2_sdk.c ../fli/ocam2_sdk.h ../fli/ocam2_pvt.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDK_DEFINES) -c $< -o $@

clean:
	rm -f ocamDescramble_bench ocamDescrambleWorker_test ocamDescrambleParallel_test ocam2_sdk.o
