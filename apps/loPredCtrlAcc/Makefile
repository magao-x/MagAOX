
allall: all

NEED_CUDA = no
# ADC records and shared windows require C++20 even when no local/common.mk sets the standard.
CXXSTD ?= -std=c++20
export CXXSTD
OTHER_HEADERS = loPredCtrlAcc.hpp loPredCtrlAcc_native.hpp native_observer.hpp native_observation.hpp ../../libMagAOX/common/dmCompletion.hpp ../../libMagAOX/common/dmTiming.hpp ../../libMagAOX/common/timingStats.hpp adc_features.hpp shadow_policy.hpp shadow_pipeline.hpp adc_input.hpp adc_reader.hpp ar_controller.hpp shadow_controller.hpp causal_input.hpp
OTHER_OBJS = native_observer.o native_observation.o adc_features.o shadow_policy.o shadow_pipeline.o adc_input.o adc_reader.o recursive_least_squares.o qrd_rls.o ar_controller.o utils.o shadow_controller.o causal_input.o
TARGET = loPredCtrlAcc

# Every installed/default build is read-only. Runtime config and command-line make overrides cannot enable output.
override DDSPC_SHADOW_ONLY = yes
override DDSPC_SHADOW_BUILD = yes
override OPTIMIZE = -O3 -fno-fast-math
export OPTIMIZE

# Build in a fresh private checkout. This target never installs or starts either executable.
.PHONY: native-observe
native-observe:
	$(MAKE) TARGET=loPredCtrlAcc_native_observe DDSPC_SHADOW_ONLY=yes all
include ../../Make/magAOXApp.mk

# Append after common flags; even an explicit CXXFLAGS override must retain the commissioning guard.
ifeq ($(DDSPC_SHADOW_ONLY),yes)
override CXXFLAGS += -std=c++20 -fno-fast-math -DDDSPC_SHADOW_ONLY
endif

# Rebuild old application objects once when adopting the mandatory guarded build profile.
$(OBJNAME) $(OTHER_OBJS): build/native-shadow-v1.stamp
$(OTHER_OBJS): $(OTHER_HEADERS)
build/native-shadow-v1.stamp:
	mkdir -p build
	touch $@

APP_CONFIG_PATH ?= $(MAGAOX_PREFIX)/config
.PHONY: install-default-config
install: install-default-config
install-default-config:
	sudo install -d $(APP_CONFIG_PATH)
	@if test -e "$(APP_CONFIG_PATH)/loPredCtrlAcc.conf" || test -L "$(APP_CONFIG_PATH)/loPredCtrlAcc.conf"; then \
	  echo 'Keeping existing loPredCtrlAcc.conf; compare it with the supplied shadow-only configuration.'; \
	else \
	  sudo install -m 0644 loPredCtrlAcc.conf "$(APP_CONFIG_PATH)/loPredCtrlAcc.conf"; \
	fi
