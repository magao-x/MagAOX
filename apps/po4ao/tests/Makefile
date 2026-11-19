# CPU-only helpers and production frame/publication methods; no GPU ABI or live INDI.
CXX = clang++
CXXFLAGS = -std=c++20 -pthread -g -O1 -Wall -Wextra -DCATCH_CONFIG_NO_POSIX_SIGNALS
all: test
.build/catch_main.o: ../../../tests/catch2/catch.hpp
	mkdir -p .build
	printf '#define CATCH_CONFIG_MAIN\n#include "../../../tests/catch2/catch.hpp"\n' | $(CXX) $(CXXFLAGS) -x c++ -c -o $@ -
.build/po4ao_test: po4ao_test.cpp ../po4ao_utils.hpp ../po4ao_control.hpp ../../nnReconstructor/tests/nnTestSupport.hpp .build/catch_main.o
	$(CXX) $(CXXFLAGS) po4ao_test.cpp .build/catch_main.o -o $@
test: .build/po4ao_test
	./.build/po4ao_test
asan:
	$(MAKE) clean
	$(MAKE) CXXFLAGS='$(CXXFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer' test
clean:
	rm -rf .build
.PHONY: all test asan clean

# Optional schema test after the normal logger code-generation step.
FLATBUFFERS_CFLAGS ?= $(shell pkg-config --cflags flatbuffers)
.build/po4ao_telemetry_test: po4ao_telemetry_test.cpp ../../../libMagAOX/logger/types/generated/telem_po4ao_generated.h .build/catch_main.o
	$(CXX) $(CXXFLAGS) $(FLATBUFFERS_CFLAGS) po4ao_telemetry_test.cpp .build/catch_main.o -o $@
telemetry: .build/po4ao_telemetry_test
	./.build/po4ao_telemetry_test
.PHONY: telemetry

# Unchanged upstream monitor with real POSIX semaphores and a mapping boundary double.
.build/po4ao_monitor_test: po4ao_monitor_test.cpp ../../../libMagAOX/app/dev/shmimMonitor.hpp stubs/ImageStreamIO/ImageStreamIO.h
	mkdir -p .build
	$(CXX) $(CXXFLAGS) -Istubs -I../../../libMagAOX/app $< -o $@
monitor: .build/po4ao_monitor_test
	./.build/po4ao_monitor_test
.PHONY: monitor
