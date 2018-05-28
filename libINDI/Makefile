

#Get MagAO-X build settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../Make/common.mk




all: lilxml indicommon

install: all
	su -c 'install -d $(LIB_PATH)'
	cd liblilxml; ${MAKE} install
	cd libindicommon; ${MAKE} install
	su -c 'echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf'
	su -c 'ldconfig'

lilxml:
	cd liblilxml; ${MAKE} all

indicommon:
	cd libindicommon; ${MAKE} all
