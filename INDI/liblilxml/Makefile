# MagAO-X makefile for liblilxml
# Modified from original LBT file:
#  -- Include common.mk to inherit MagAO-X build settings.
#  -- Changed to targets all and clean to conform to MagAO-X conventions
#  -- Changed to ar -r, and ranlib is ar -s
#  -- Removed xmlcheck and liltest


#Get MagAO-X build settings
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk

all: liblilxml.a
	$(CC) -shared -Wl,-soname,liblilxml.so -o liblilxml.so lilxml*.o -rdynamic

install: all
	sudo install -d $(LIB_PATH)
	sudo install liblilxml.so $(LIB_PATH)

HS = lilxml.h

OBJS = lilxml.o base64.o

liblilxml.a: $(HS) $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@

clean:
	rm -f *.o *.a *.so
