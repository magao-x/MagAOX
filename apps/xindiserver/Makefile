
allall: all

CPPFLAGS   += -DGIT_TAG_STRING='"$(shell git describe --tags)"'
INCLUDES   += -I../../INDI/liblilxml
OTHER_HEADERS=
TARGET      = xindiserver
include ../../Make/magAOXApp.mk
