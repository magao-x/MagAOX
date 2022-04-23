
allall: all

TARGET=qhyCtrl

QHYCAM_INCLUDES=-I/usr/local/lib/ -I/usr/include/
INCLUDES += $(QHYCAM_INCLUDES)

QHY_LIB  = /usr/local/lib/libqhyccd.so
LDLIBS += -Wl,${QHY_LIB} -lusb-1.0 -pthread


include ../../Make/magAOXApp.mk
