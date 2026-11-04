# K-Cube control uses the companion tmcController dependency with bounded I/O recovery.
allall: all

TMC_CONTROLLER_PATH ?= /opt/MagAOX/source/tmcController
FTDI_CFLAGS ?= $(shell pkg-config --cflags libftdi1)
FTDI_LIBS ?= $(shell pkg-config --libs libftdi1)

OTHER_HEADERS=$(TMC_CONTROLLER_PATH)/tmcController.hpp
TARGET=kcubeCtrl
INCLUDES+=-I$(TMC_CONTROLLER_PATH) $(FTDI_CFLAGS)
LDLIBS+=$(FTDI_LIBS)

include ../../Make/magAOXApp.mk
