allall: check-platform all

OTHER_HEADERS = dependencies/AD7689.h
OTHER_OBJS = dependencies/AD7689.o
TARGET = ad7689Ctrl
GPIOD_CFLAGS = $(shell pkg-config --cflags libgpiod)
GPIOD_LIBS = $(shell pkg-config --libs libgpiod)

include ../../Make/magAOXApp.mk

# Validation must retain IEEE finite checks, including when common.mk enables fast-math.
CXXFLAGS += -fno-fast-math $(GPIOD_CFLAGS)
LDLIBS += $(GPIOD_LIBS)

.PHONY: check-platform
check-platform:
	@test "$$(uname -s)" = Linux || { echo 'ad7689Ctrl requires a non-instrument Linux development build environment.' >&2; exit 1; }
	@pkg-config --atleast-version=2.0 libgpiod || { echo 'libgpiod >= 2.0 development files are required.' >&2; exit 1; }

ad7689Ctrl.o dependencies/AD7689.o: | check-platform
dependencies/AD7689.o: dependencies/AD7689.h
