# CPU Session tests and the real app/monitor interaction; no hardware connections.
CXX = clang++
CXXFLAGS = -std=c++20 -pthread -g -O1 -Wall -Wextra -DCATCH_CONFIG_NO_POSIX_SIGNALS
OPENSSL_CFLAGS ?= $(shell pkg-config --cflags openssl)
OPENSSL_LIBS ?= $(shell pkg-config --libs libcrypto)
INDI_LIBS = ../../../INDI/libcommon/libcommon.a ../../../INDI/liblilxml/liblilxml.a

all: test
.build/dlDataCollection_test: dlDataCollection_test.cpp ../dlCollection.hpp ../dlObservation.hpp $(INDI_LIBS)
	mkdir -p .build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fno-fast-math $(OPENSSL_CFLAGS) -DCATCH_CONFIG_MAIN $< $(INDI_LIBS) $(LDFLAGS) $(OPENSSL_LIBS) -o $@

# Reuse the existing monitor mapping declarations; link the repository's built INDI value types.
.build/dlDataCollection_monitor_test: dlDataCollection_monitor_test.cpp ../dlDataCollection.hpp ../dlCollection.hpp ../dlObservation.hpp ../../../libMagAOX/app/dev/shmimMonitor.hpp ../../po4ao/tests/stubs/ImageStreamIO/ImageStreamIO.h $(INDI_LIBS)
	mkdir -p .build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fno-fast-math $(OPENSSL_CFLAGS) -I../../po4ao/tests/stubs -I../../../libMagAOX/app $< $(INDI_LIBS) $(LDFLAGS) $(OPENSSL_LIBS) -o $@

test: .build/dlDataCollection_test .build/dlDataCollection_monitor_test
	./.build/dlDataCollection_test
	./.build/dlDataCollection_monitor_test

asan:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer' test

clean:
	rm -rf .build
.PHONY: all test asan clean
