# MagAO-X makefile for libtelnet

#Get MagAO-X build settings
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk


all: libtelnet.a
	$(CC) -shared -Wl,-soname,libtelnet.so -o libtelnet.so libtelnet*.o -rdynamic

install: all
	/bin/sudo install -d $(LIB_PATH)
	/bin/sudo install libtelnet.so $(LIB_PATH)
	/bin/sudo echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf
	/bin/sudo ldconfig


HS = libtelnet.h

OBJS = libtelnet.o

libtelnet.a: $(HS) $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@


clean:
	rm -f *.o *.a *.so
