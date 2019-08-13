# MagAO-X makefile for libtelnet

#Get MagAO-X build settings
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk


all: libtelnet.a
	$(CC) $(CFLAGS) -shared -Wl,-soname,libtelnet.so -o libtelnet.so libtelnet*.o -rdynamic

install: all
	sudo install -d $(LIB_PATH)
	sudo install libtelnet.so $(LIB_PATH)

HS = libtelnet.h

OBJS = libtelnet.o

libtelnet.a: $(HS) $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@

clean:
	rm -f *.o *.a *.so
