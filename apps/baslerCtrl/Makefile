
allall: all

TARGET=baslerCtrl

PYLON_ROOT ?= /opt/pylon5
CFLAGS     += -D_XOPEN_SOURCE=700
CXXFLAGS   += -O3 -D_XOPEN_SOURCE=700
CPPFLAGS   := $(shell $(PYLON_ROOT)/bin/pylon-config --cflags)
LDFLAGS    := $(shell $(PYLON_ROOT)/bin/pylon-config --libs-rpath)
LDLIBS     := $(shell $(PYLON_ROOT)/bin/pylon-config --libs)  


include ../../Make/magAOXApp.mk
