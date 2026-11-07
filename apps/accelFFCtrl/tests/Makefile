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
SOURCES := accelFFCtrlModel_test.cpp accelFFCtrlReplay_test.cpp accelFFCtrlInput_test.cpp accelFFCtrlValidation_test.cpp accelFFCtrlGap_test.cpp accelFFCtrlJitter_test.cpp accelFFCtrlCancellation_test.cpp accelFFCtrlShadow_test.cpp $(REPO_ROOT)/tests/testMain.cpp
TEST := $(BUILD_DIR)/accelFFCtrl_test
BENCHMARK := $(BUILD_DIR)/accelFFCtrl_benchmark
INPUT_BENCHMARK := $(BUILD_DIR)/accelFFCtrlInput_benchmark
NATIVE := $(BUILD_DIR)/accelFFCtrlStream_test
CANCELLATION := $(BUILD_DIR)/accelFFCtrlCancellation_export
AO := $(BUILD_DIR)/accelFFCtrlAO_export
TELEMETRY := $(BUILD_DIR)/accelFFCtrlTelemetry_replay
RESPONSE := $(BUILD_DIR)/accelFFCtrlResponse_replay
CLOSED_LOOP := $(BUILD_DIR)/accelFFCtrlClosedLoop_export
ISIO_CFLAGS ?= $(shell pkg-config --cflags ImageStreamIO)
ISIO_LIBS ?= $(shell pkg-config --libs ImageStreamIO)

PYTHON ?= python3
VALIDATION_DIR ?= /tmp/accelFFCtrl-validation
VALIDATION_SECONDS ?= 60

.PHONY: all test benchmark native-test native-rate-test validation validation-test gap-validation cancellation-validation shadow-tools telemetry-tools telemetry-test response-tools closed-loop-tools
all: $(TEST) $(BENCHMARK) $(INPUT_BENCHMARK)

$(BUILD_DIR):
	mkdir -p $@

$(TEST): $(SOURCES) accelFFCtrlGap_test.hpp accelFFCtrlCancellation_test.hpp accelFFCtrlAO_test.hpp ../offline/accelFFCtrlCancellation.hpp ../offline/accelFFCtrlShadow.hpp $(MODEL) $(INPUT) ../../accelSim/accelSimModel.hpp ../../accelSim/accelSimDisturbance.hpp $(REPO_ROOT)/tests/testXWC.hpp | $(BUILD_DIR)
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

# Entirely offline: no native reader, INDI, application launch or shared-memory access.
validation: $(TEST) validation-test
	ACCELFFCTRL_VALIDATION_DIR='$(VALIDATION_DIR)' ACCELFFCTRL_VALIDATION_SECONDS='$(VALIDATION_SECONDS)' $(TEST) '[accelFFCtrl-validation-export]'
	MPLCONFIGDIR='$(VALIDATION_DIR)/.matplotlib' $(PYTHON) validate_predictions.py '$(VALIDATION_DIR)'

validation-test:
	$(PYTHON) -m unittest -v test_validate_predictions.py test_validate_gaps.py test_analyze_timestamps.py test_validate_cancellation.py test_validate_shadow.py test_qualify_telemetry.py test_inspect_reduced_telemetry.py test_analyze_timed_telemetry.py test_validate_real_harmonic.py test_validate_phase_calibration.py test_validate_recorded_response.py test_validate_closed_loop.py

gap-validation: $(TEST) validation-test
	ACCELFFCTRL_VALIDATION_DIR='$(VALIDATION_DIR)' $(TEST) '[accelFFCtrl-jitter]'
	ACCELFFCTRL_VALIDATION_DIR='$(VALIDATION_DIR)' ACCELFFCTRL_VALIDATION_SECONDS='$(VALIDATION_SECONDS)' $(TEST) '[accelFFCtrl-gap-export]'
	MPLCONFIGDIR='$(VALIDATION_DIR)/.matplotlib' $(PYTHON) validate_gaps.py '$(VALIDATION_DIR)'

$(CANCELLATION): accelFFCtrlCancellation_export.cpp accelFFCtrlCancellation_test.hpp accelFFCtrlAO_test.hpp ../offline/accelFFCtrlCancellation.hpp ../offline/accelFFCtrlShadow.hpp $(MODEL) $(INPUT) ../../accelSim/accelSimDisturbance.hpp ../../accelSim/accelSimModel.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

cancellation-validation: $(CANCELLATION)
	$(CANCELLATION) '$(VALIDATION_DIR)' '$(VALIDATION_SECONDS)'
	MPLCONFIGDIR='$(VALIDATION_DIR)/.matplotlib' $(PYTHON) validate_cancellation.py '$(VALIDATION_DIR)'

$(AO): accelFFCtrlAO_export.cpp accelFFCtrlAO_test.hpp accelFFCtrlCancellation_test.hpp ../offline/accelFFCtrlCancellation.hpp $(MODEL) ../../accelSim/accelSimDisturbance.hpp ../../accelSim/accelSimModel.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

shadow-tools: $(AO) $(CANCELLATION)

$(TELEMETRY): accelFFCtrlTelemetry_replay.cpp accelFFCtrlPhase_test.hpp ../offline/accelFFCtrlShadow.hpp ../offline/accelFFCtrlCancellation.hpp $(INPUT) $(MODEL) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

telemetry-tools: $(TELEMETRY)

$(RESPONSE): accelFFCtrlResponse_replay.cpp ../offline/accelFFCtrlShadow.hpp ../offline/accelFFCtrlCancellation.hpp $(MODEL) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

response-tools: $(RESPONSE)

$(CLOSED_LOOP): accelFFCtrlClosedLoop_export.cpp accelFFCtrlAO_test.hpp accelFFCtrlPhase_test.hpp accelFFCtrlCancellation_test.hpp ../offline/accelFFCtrlShadow.hpp ../offline/accelFFCtrlCancellation.hpp $(MODEL) ../../accelSim/accelSimModel.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LDFLAGS) -o $@

closed-loop-tools: $(CLOSED_LOOP)

telemetry-test: $(TELEMETRY)
	ACCELFFCTRL_TELEMETRY_REPLAY='$(abspath $(TELEMETRY))' $(PYTHON) -m unittest -v test_qualify_telemetry.py

$(NATIVE): accelFFCtrlStream_test.cpp ../accelFFCtrlStream.hpp $(INPUT) $(MODEL) $(REPO_ROOT)/tests/testMain.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(ISIO_CFLAGS) $(CXXFLAGS) $< $(REPO_ROOT)/tests/testMain.cpp $(LDFLAGS) $(ISIO_LIBS) -o $@

# A fresh process/namespace is mandatory: the native library can cache MILK_SHM_DIR.
native-test native-rate-test: $(NATIVE)
	@if [ "$@" = native-rate-test ] && [ "$$(uname -s)" != Linux ]; then echo 'native-rate-test requires Linux'; exit 1; fi
	@task_shm_dir=$$(mktemp -d /tmp/accelFFCtrl-native-XXXXXX) || exit 1; \
	  trap 'task_status=$$?; rmdir "$$task_shm_dir" || task_status=1; exit $$task_status' EXIT; \
	  MILK_SHM_DIR="$$task_shm_dir" $(NATIVE) '$(if $(filter native-rate-test,$@),[accelFFCtrl-native-rate],[accelFFCtrl-native])'
