

#Get MagAO-X build settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../Make/common.mk




all: lilxml common
	$(CC) -shared -Wl,-soname,liblbti.so -o liblbti.so liblilxml/*.o libcommon/*.o -rdynamic

install: all
	su -c 'install -d $(LIB_PATH)'
	su -c 'install liblbti.so $(LIB_PATH)'
	su -c 'echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf'
	su -c 'ldconfig'

lilxml:
	cd liblilxml; ${MAKE} all

common:
	cd libcommon; ${MAKE} all

clean:
	cd liblilxml; ${MAKE} clean
	cd libcommon; ${MAKE} clean
	rm -f *.o *.a *.so
