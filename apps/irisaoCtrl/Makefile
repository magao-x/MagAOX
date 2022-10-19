
allall: all

OTHER_HEADERS=
TARGET=irisaoCtrl

IRISAO_PATH=/opt/IrisAO
LDFLAGS += -L$(IRISAO_PATH)/lib -Wl,-rpath,$(IRISAO_PATH)/lib
LDLIBS+= -lirisao.devices.1.0.2.5
INCLUDES+= -I/$(IRISAO_PATH)/include

include ../../Make/magAOXApp.mk

