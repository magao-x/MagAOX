
allall: all

NEED_CUDA = no
# ADC records and shared windows require C++20 even when no local/common.mk sets the standard.
CXXSTD ?= -std=c++20
export CXXSTD
OTHER_HEADERS = ../../libMagAOX/app/syncoOutput.hpp phase_shadow.hpp native_pipeline.hpp synco_shadow.hpp ../../libMagAOX/app/dmSyncopationTiming.hpp ../../libMagAOX/app/dev/dmSyncopationProducer.hpp synchronized_inputs.hpp multirate_shadow.hpp online_shadow.hpp loPredCtrlAcc.hpp loPredCtrlAcc_native.hpp native_observer.hpp native_observation.hpp ../../libMagAOX/common/dmCompletion.hpp ../../libMagAOX/common/dmTiming.hpp ../../libMagAOX/common/timingStats.hpp adc_features.hpp shadow_policy.hpp shadow_pipeline.hpp adc_input.hpp adc_reader.hpp ar_controller.hpp shadow_controller.hpp causal_input.hpp
OTHER_OBJS = phase_shadow.o native_pipeline.o synco_shadow.o synchronized_inputs.o multirate_shadow.o online_shadow.o native_observer.o native_observation.o adc_features.o shadow_policy.o shadow_pipeline.o adc_input.o adc_reader.o recursive_least_squares.o qrd_rls.o ar_controller.o utils.o shadow_controller.o causal_input.o
TARGET = loPredCtrlAcc

# Default builds are read-only. Only the explicit operational profile permits the guarded producer.
ifeq ($(DDSPC_OPERATIONAL_BUILD),yes)
override DDSPC_SHADOW_ONLY = no
else
override DDSPC_SHADOW_ONLY = yes
endif
override DDSPC_SHADOW_BUILD = yes
override OPTIMIZE = -O3 -fno-fast-math
export OPTIMIZE

# Build in a fresh private checkout. This target never installs or starts either executable.
.PHONY: native-observe
native-observe:
	$(MAKE) TARGET=loPredCtrlAcc_native_observe DDSPC_OPERATIONAL_BUILD=no DDSPC_SHADOW_ONLY=yes all
.PHONY: operational
operational:
	$(MAKE) TARGET=loPredCtrlAcc DDSPC_OPERATIONAL_BUILD=yes all
include ../../Make/magAOXApp.mk

# Append after common flags; even an explicit CXXFLAGS override must retain the commissioning guard.
ifeq ($(DDSPC_SHADOW_ONLY),yes)
override CXXFLAGS += -std=c++20 -fno-fast-math -DEIGEN_DONT_PARALLELIZE -DDDSPC_SHADOW_ONLY
endif
ifeq ($(DDSPC_OPERATIONAL_BUILD),yes)
override CXXFLAGS += -std=c++20 -fno-fast-math -DEIGEN_DONT_PARALLELIZE -DDDSPC_OPERATIONAL_BUILD
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

# A profile change invalidates every application object; never reuse a shadow PCH for operational linking.
.PHONY: output-build-profile
output-build-profile:
	@mkdir -p build
	@if test "$(DDSPC_OPERATIONAL_BUILD)" != "$$(cat build/output-profile 2>/dev/null)"; then printf "%s\n" "$(DDSPC_OPERATIONAL_BUILD)" > build/output-profile; fi
$(OBJNAME) $(OTHER_OBJS): build/output-profile
build/output-profile: output-build-profile
