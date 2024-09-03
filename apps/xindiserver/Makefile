
allall: all 

CPPFLAGS   += -DGIT_TAG_STRING='"$(shell git describe --tags)"'
INCLUDES   += -I../../INDI/liblilxml
LDLIBS     += $(shell pkg-config --libs zlib)
OTHER_HEADERS=
OTHER_OBJS  = callable_indiserver.o
TARGET      = xindiserver
include ../../Make/magAOXApp.mk
