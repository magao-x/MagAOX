
allall: all

OTHER_HEADERS=cred2Utils.hpp
TARGET=cred2Ctrl

EDT=true
EDT_PATH=/opt/EDTpdv

EDT_INCLUDES=-I$(EDT_PATH)
EDT_LIBS = -L/opt/EDTpdv -lpdv -lpthread -lm -ldl

INCLUDES += $(EDT_INCLUDES)
LDLIBS += $(EDT_LIBS)

include ../../Make/magAOXApp.mk
