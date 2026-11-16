## \file Makefile
## \brief Build the hardware-free OCAM descramble benchmark independently of the application.

CXX ?= c++
CC ?= cc
CXXFLAGS ?= -O3 -march=native -ffast-math -std=c++20 -Wall -Wextra
CFLAGS ?= -O3 -march=native -ffast-math
SDK_DEFINES = -DOCAMSDK_VERSION=\"2\" -DOCAMSDK_BUILD=\"0\"

.PHONY: all clean
all: ocamDescramble_bench

ocamDescramble_bench: ocamDescramble_bench.cpp ../ocamDescrambler.hpp ../../../libMagAOX/common/streamCopy.hpp ocam2_sdk.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DOCAM_BENCH_CXXFLAGS='"$(CPPFLAGS) $(CXXFLAGS)"' -DOCAM_BENCH_CFLAGS='"$(CFLAGS)"' $< ocam2_sdk.o $(LDFLAGS) $(LDLIBS) -o $@

ocam2_sdk.o: ../fli/ocam2_sdk.c ../fli/ocam2_sdk.h ../fli/ocam2_pvt.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDK_DEFINES) -c $< -o $@

clean:
	rm -f ocamDescramble_bench ocam2_sdk.o
