# MagAO-X makefile for the libcommon component of liblbt
# This is originally libcommon in the LBT sources.
#
# Modified from original LBT file:
#  -- Include common.mk to inherit MagAO-X build settings.
#  -- Changed to targets all and clean to conform to MagAO-X conventions
#  -- Changed to ar -r, and ranlib is ar -s
#  -- Removed files not used in MagAO-X.


#Get MagAO-X build settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../../Make/common.mk



all: libcommon.a

SRCS = \
	 IndiElement.cpp \
	 IndiMessage.cpp \
	 IndiProperty.cpp \
	 IndiXmlParser.cpp \
	 LogFile.cpp \
	 Logger.cpp \
	 System.cpp \
	 SystemSocket.cpp \
	 Thread.cpp \
	 TimeStamp.cpp

OBJS = $(SRCS:.cpp=.o)

libcommon.a: $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@



clean:
	rm -f *.o *.a *.so

