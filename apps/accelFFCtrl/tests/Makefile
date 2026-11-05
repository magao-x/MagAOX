# Standalone numerical validation; no MagAO-X library, hardware SDK, or ImageStreamIO dependency.
CXX ?= c++
CXXFLAGS ?= -O3
override CXXFLAGS += -std=c++20 -fno-fast-math -Wall -Wextra
EIGEN_CFLAGS ?= $(shell pkg-config --cflags eigen3)
CPPFLAGS += $(EIGEN_CFLAGS)
BUILD_DIR ?= /tmp/accelFFCtrl-numerics
REPO_ROOT := ../../..
MODEL := ../accelFFCtrlModel.hpp
SOURCES := accelFFCtrlModel_test.cpp accelFFCtrlReplay_test.cpp $(REPO_ROOT)/tests/testMain.cpp
TEST := $(BUILD_DIR)/accelFFCtrl_test
BENCHMARK := $(BUILD_DIR)/accelFFCtrl_benchmark

.PHONY: all test benchmark
all: $(TEST) $(BENCHMARK)

$(BUILD_DIR):
	mkdir -p $@

$(TEST): $(SOURCES) $(MODEL) ../../accelSim/accelSimModel.hpp $(REPO_ROOT)/tests/testXWC.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) -o $@

$(BENCHMARK): accelFFCtrl_benchmark.cpp $(MODEL) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DEIGEN_RUNTIME_NO_MALLOC $< $(LDFLAGS) -o $@

test: $(TEST)
	$(TEST) '[accelFFCtrl]'

benchmark: $(BENCHMARK)
	$(BENCHMARK)
