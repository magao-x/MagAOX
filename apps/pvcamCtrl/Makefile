
allall: all 

OTHER_HEADERS=
TARGET=pvcamCtrl

PVCAM_PATH=/opt/pvcam/sdk/include/

PVCAM_INCLUDES=-I$(PVCAM_PATH)
PVCAM_LIBS = -L/opt/pvcam/library/x86_64/ -lpvcam

INCLUDES += $(PVCAM_INCLUDES)
LDLIBS += $(PVCAM_LIBS)

include ../../Make/magAOXApp.mk

