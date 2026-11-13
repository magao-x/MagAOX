
allall: all

NEED_CUDA = no
# ADC records and shared windows require C++20 even when no local/common.mk sets the standard.
CXXSTD ?= -std=c++20
export CXXSTD
OTHER_HEADERS = adc_features.hpp shadow_policy.hpp shadow_pipeline.hpp adc_input.hpp adc_reader.hpp ar_controller.hpp shadow_controller.hpp causal_input.hpp
OTHER_OBJS = adc_features.o shadow_policy.o shadow_pipeline.o adc_input.o adc_reader.o recursive_least_squares.o qrd_rls.o ar_controller.o utils.o shadow_controller.o causal_input.o
TARGET = loPredCtrlAcc
# Manual commissioning builds refuse legacy execution before any input/output helper starts.
ifeq ($(DDSPC_SHADOW_ONLY),yes)
DDSPC_SHADOW_BUILD = yes
endif
# Opt-in shadow builds must rebuild all objects without finite-only/fast-math assumptions.
# Use a clean build with: make DDSPC_SHADOW_BUILD=yes
# The default legacy build flags remain unchanged.
ifeq ($(DDSPC_SHADOW_BUILD),yes)
OPTIMIZE = -fno-fast-math
export OPTIMIZE
endif
#TARGET = testPredCtrl
#TARGET = testRLS
include ../../Make/magAOXApp.mk

# Append after common flags; even an explicit CXXFLAGS override must retain the commissioning guard.
ifeq ($(DDSPC_SHADOW_ONLY),yes)
override CXXFLAGS += -DDDSPC_SHADOW_ONLY
endif
