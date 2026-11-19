allall: all

CXXSTD ?= -std=c++20
export CXXSTD
NEED_CUDA=no
TARGET=accelSim
OTHER_HEADERS=../../libMagAOX/app/dev/dmSyncopationProducer.hpp accelSimModel.hpp accelSimTimeline.hpp accelSimOutput.hpp accelSimDisturbance.hpp accelSimDM.hpp
override OPTIMIZE = -O3 -fno-fast-math
export OPTIMIZE
include ../../Make/magAOXApp.mk
override CXXFLAGS += -std=c++20 -fno-fast-math -DEIGEN_DONT_PARALLELIZE

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
