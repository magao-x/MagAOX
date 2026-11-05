# Standalone numerical/input validation; native-test additionally requires ImageStreamIO.
CXX ?= c++
CXXFLAGS ?= -O3
override CXXFLAGS += -std=c++20 -fno-fast-math -Wall -Wextra -pthread
EIGEN_CFLAGS ?= $(shell pkg-config --cflags eigen3)
CPPFLAGS += $(EIGEN_CFLAGS) -D_GNU_SOURCE
BUILD_DIR ?= /tmp/accelFFCtrl-numerics
REPO_ROOT := ../../..
MODEL := ../accelFFCtrlModel.hpp
INPUT := ../accelFFCtrlInput.hpp
SOURCES := accelFFCtrlModel_test.cpp accelFFCtrlReplay_test.cpp accelFFCtrlInput_test.cpp $(REPO_ROOT)/tests/testMain.cpp
TEST := $(BUILD_DIR)/accelFFCtrl_test
BENCHMARK := $(BUILD_DIR)/accelFFCtrl_benchmark
INPUT_BENCHMARK := $(BUILD_DIR)/accelFFCtrlInput_benchmark
NATIVE := $(BUILD_DIR)/accelFFCtrlStream_test
ISIO_CFLAGS ?= $(shell pkg-config --cflags ImageStreamIO)
ISIO_LIBS ?= $(shell pkg-config --libs ImageStreamIO)

.PHONY: all test benchmark native-test native-rate-test
all: $(TEST) $(BENCHMARK) $(INPUT_BENCHMARK)

$(BUILD_DIR):
	mkdir -p $@

$(TEST): $(SOURCES) $(MODEL) $(INPUT) ../../accelSim/accelSimModel.hpp $(REPO_ROOT)/tests/testXWC.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $@

$(BENCHMARK): accelFFCtrl_benchmark.cpp $(MODEL) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DEIGEN_RUNTIME_NO_MALLOC $< $(LDFLAGS) -o $@

$(INPUT_BENCHMARK): accelFFCtrlInput_benchmark.cpp $(INPUT) $(MODEL) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DEIGEN_RUNTIME_NO_MALLOC $< $(LDFLAGS) -o $@

test: $(TEST)
	$(TEST) '[accelFFCtrl]'

benchmark: $(BENCHMARK) $(INPUT_BENCHMARK)
	$(BENCHMARK)
	$(INPUT_BENCHMARK)

$(NATIVE): accelFFCtrlStream_test.cpp ../accelFFCtrlStream.hpp $(INPUT) $(MODEL) $(REPO_ROOT)/tests/testMain.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(ISIO_CFLAGS) $(CXXFLAGS) $< $(REPO_ROOT)/tests/testMain.cpp $(LDFLAGS) $(ISIO_LIBS) -o $@

# A fresh process/namespace is mandatory: the native library can cache MILK_SHM_DIR.
native-test native-rate-test: $(NATIVE)
	@if [ "$@" = native-rate-test ] && [ "$$(uname -s)" != Linux ]; then echo 'native-rate-test requires Linux'; exit 1; fi
	@task_shm_dir=$$(mktemp -d /tmp/accelFFCtrl-native-XXXXXX) || exit 1; \
	  trap 'task_status=$$?; rmdir "$$task_shm_dir" || task_status=1; exit $$task_status' EXIT; \
	  MILK_SHM_DIR="$$task_shm_dir" $(NATIVE) '$(if $(filter native-rate-test,$@),[accelFFCtrl-native-rate],[accelFFCtrl-native])'
