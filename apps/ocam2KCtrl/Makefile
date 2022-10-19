
allall: all

OTHER_HEADERS=ocamUtils.hpp
OTHER_OBJS=fli/ocam2_sdk.o
TARGET=ocam2KCtrl

EDT_PATH=/opt/EDTpdv

EDT_INCLUDES=-I$(EDT_PATH)
EDT_LIBS = -L/opt/EDTpdv -lpdv -lpthread -lm -ldl

INCLUDES += $(EDT_INCLUDES)
LDLIBS += $(EDT_LIBS)

include ../../Make/magAOXApp.mk

