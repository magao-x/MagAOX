allall: all
OTHER_HEADERS=orcaUtils.hpp
TARGET=orcaCtrl

EDT=true
EDT_PATH=/opt/EDTpdv

EDT_INCLUDES=-I$(EDT_PATH)
EDT_LIBS = -L/opt/EDTpdv -lpdv -lpthread -lm -ldl -ldcamapi

INCLUDES += $(EDT_INCLUDES)
LDLIBS += $(EDT_LIBS)

include ../../Make/magAOXApp.mk