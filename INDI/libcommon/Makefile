# MagAO-X makefile for the libcommon component of liblbt
# This is originally libcommon in the LBT sources.
#
# Modified from original LBT file:
#  -- Include common.mk to inherit MagAO-X build settings.
#  -- Changed to targets all and clean to conform to MagAO-X conventions
#  -- Changed to ar -r, and ranlib is ar -s
#  -- Removed files not used in MagAO-X.


#Get MagAO-X build settings
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk



all: libcommon.a


SRCS = \
	 Cmd.cpp \
	 IndiConnection.cpp \
	 IndiClient.cpp \
	 IndiDriver.cpp \
	 IndiElement.cpp \
	 IndiMessage.cpp \
	 IndiProperty.cpp \
	 IndiPropertyMap.cpp \
	 IndiXmlParser.cpp \
	 System.cpp \
	 SystemSocket.cpp \
	 Thread.cpp \
	 TimeStamp.cpp


OBJS = $(SRCS:.cpp=.o)

libcommon.a: $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@

IndiElement.o: IndiElement.hpp IndiElement.cpp 

clean:
	rm -f *.o *.a *.so

