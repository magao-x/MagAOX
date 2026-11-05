allall: all

TARGET=accelFFCtrl
OTHER_HEADERS=accelFFCtrlModel.hpp accelFFCtrlInput.hpp accelFFCtrlStream.hpp
OPTIMIZE ?= -O3 -fno-fast-math
CXXFLAGS += -D_GNU_SOURCE -pthread
LDLIBS += -lImageStreamIO
include ../../Make/magAOXApp.mk
