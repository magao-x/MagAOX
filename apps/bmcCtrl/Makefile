
allall: all

OTHER_HEADERS=
TARGET=bmcCtrl

BMC_PATH=/opt/Boston\ Micromachines
LDFLAGS += -L$(BMC_PATH)/lib -Wl,-rpath,$(BMC_PATH)/lib
LDLIBS+= -lBMC -lBMC_PCIeAPI
INCLUDES+= -I/$(BMC_PATH)/include

include ../../Make/magAOXApp.mk

