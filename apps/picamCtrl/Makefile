
allall: all

OTHER_HEADERS=
OTHER_OBJS=
TARGET=picamCtrl

PICAM_PATH=/opt/PrincetonInstruments/picam/includes/

PICAM_INCLUDES=-I$(PICAM_PATH)
PICAM_LIBS = -lpicam

INCLUDES += $(PICAM_INCLUDES)
LDLIBS += $(PICAM_LIBS)

include ../../Make/magAOXApp.mk
