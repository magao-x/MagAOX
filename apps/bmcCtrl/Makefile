
allall: all

OTHER_HEADERS=
TARGET=bmcCtrl

BMC_PATH=/opt/Boston\ Micromachines
LDFLAGS += -L$(BMC_PATH)/lib -Wl,-rpath,$(BMC_PATH)/lib 
LDLIBS+= -lBMC -lBMC_PCIeAPI -lBMC_USBAPI -l:liblog4cxx.so.10 -l:libapr-1.so.0 -l:libaprutil-1.so.0
INCLUDES+= -I/$(BMC_PATH)/include -DXWC_DMTIMINGS

include ../../Make/magAOXApp.mk

