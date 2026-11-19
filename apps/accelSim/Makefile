allall: all

CXXSTD ?= -std=c++20
export CXXSTD
NEED_CUDA=no
TARGET=accelSim
OTHER_HEADERS=../../libMagAOX/app/syncoOutput.hpp accelSimModel.hpp accelSimOutput.hpp accelSimDisturbance.hpp accelSimDM.hpp accelSimDMStream.hpp
override OPTIMIZE = -O3 -fno-fast-math
export OPTIMIZE
include ../../Make/magAOXApp.mk
override CXXFLAGS += -std=c++20 -fno-fast-math -DEIGEN_DONT_PARALLELIZE

# Explicit profile; default simulator builds preserve ADC acquisition but cannot publish DM payloads.
ifeq ($(DDSPC_OPERATIONAL_BUILD),yes)
override CXXFLAGS += -DDDSPC_OPERATIONAL_BUILD
else
override CXXFLAGS += -DDDSPC_SHADOW_ONLY
endif
.PHONY: operational
operational:
	$(MAKE) DDSPC_OPERATIONAL_BUILD=yes all

.PHONY: output-build-profile
output-build-profile:
	@mkdir -p build
	@if test "$(DDSPC_OPERATIONAL_BUILD)" != "$$(cat build/output-profile 2>/dev/null)"; then printf "%s\n" "$(DDSPC_OPERATIONAL_BUILD)" > build/output-profile; fi
$(OBJNAME) $(OTHER_OBJS): build/output-profile
build/output-profile: output-build-profile

# Match the app installation convention while preserving any commissioned configuration.
APP_CONFIG_PATH ?= $(MAGAOX_PREFIX)/config
.PHONY: install-default-config
install: install-default-config
install-default-config:
	sudo install -d $(APP_CONFIG_PATH)
	@if test -e "$(APP_CONFIG_PATH)/acc000.conf" || test -L "$(APP_CONFIG_PATH)/acc000.conf"; then \
	  echo 'Keeping existing acc000.conf; compare with the supplied disabled experiment configuration.'; \
	else \
	  sudo install -m 0644 acc000.conf "$(APP_CONFIG_PATH)/acc000.conf"; \
	fi
