
allall: all

OTHER_HEADERS=
OTHER_OBJS=
TARGET=orcaCtrl

ORCA_PATH=/opt/hamamatsu_new/hamamatsu_sdk/dcamsdk4/inc/

ORCA_INCLUDES=-I$(ORCA_PATH)
ORCA_LIBS = -ldcamapi
INCLUDES += $(ORCA_INCLUDES)
LDLIBS += $(ORCA_LIBS)

include ../../Make/magAOXApp.mk