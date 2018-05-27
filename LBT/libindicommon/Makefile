# MagAO-X makefile for libindicommon
# This is originally libcommon in the LBT sources.
#
# Modified from original LBT file:
#  -- Include common.mk to inherit MagAO-X build settings.
#  -- Changed to targets all and clean to conform to MagAO-X conventions
#  -- Changed to ar -r, and ranlib is ar -s, and addid command to make shared library.
#  -- Changed install to install .so to LIB_PATH, and update ldconfig


#Get MagAO-X build settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../../Make/common.mk



all: libindicommon.a

install: all
	su -c 'install -d $(LIB_PATH)'
	su -c 'install libindicommon.so $(LIB_PATH)'
	su -c 'echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf'
	su -c 'ldconfig'
	su -c 'install -d $(INCLUDE_PATH)/indi'
	su -c 'install *.hpp $(INCLUDE_PATH)/indi'	
SRCS = \
	 Cmd.cpp \
	 Config.cpp \
	 ConfigFile.cpp \
	 Daemon.cpp \
	 Event.cpp \
	 IndiConnection.cpp \
	 IndiClient.cpp \
	 IndiDriver.cpp \
	 IndiElement.cpp \
	 IndiMessage.cpp \
	 IndiProperty.cpp \
	 IndiPropertyMap.cpp \
	 IndiXmlParser.cpp \
	 LogFile.cpp \
	 Logger.cpp \
	 RtdbMessage.cpp \
	 RtdbVariable.cpp \
	 System.cpp \
	 SystemSocket.cpp \
	 TCPConnection.cpp \
	 TCPListener.cpp \
	 Thread.cpp \
	 TimeStamp.cpp


OBJS = $(SRCS:.cpp=.o)

libindicommon.a: $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@
	$(CC) -shared -Wl,-soname,libindicommon.so -o libindicommon.so *.o -rdynamic



clean:
	rm -f *.o *.a *.so

