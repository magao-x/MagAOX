# MagAO-X makefile for liblilxml
# Modified from original LBT file:
#  -- Include common.mk to inherit MagAO-X build settings.
#  -- Changed to targets all and clean to conform to MagAO-X conventions
#  -- Changed to ar -r, and ranlib is ar -s, and addid command to make shared library.
#  -- Changed install to install .so to LIB_PATH, and update ldconfig
#  -- Removed xmlcheck and liltest




#Get MagAO-X build settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../../Make/common.mk


all: liblilxml.a 


install: all
	su -c 'install -d $(LIB_PATH)'
	su -c 'install liblilxml.so $(LIB_PATH)'
	su -c 'echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf'
	su -c 'ldconfig'

HS = lilxml.h

OBJS = lilxml.o base64.o

liblilxml.a: $(HS) $(OBJS)
	ar -r $@ $(OBJS)
	ar -s $@
	$(CC) -shared -Wl,-soname,liblilxml.so -o liblilxml.so *.o -rdynamic
	

clean:
	rm -f *.o *.a *.so
