# Makefile for libcommon.a

# gcc
CC = g++
CXX = g++
CXXFLAGS= -O2 -Wall -g


pass3: libcommon.a

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

libcommon.a: $(OBJS)
	ar r $@ $(OBJS)
	ranlib $@

clean: clobber

clobber:
	touch x.o x.a
	rm -f *.o *.a

