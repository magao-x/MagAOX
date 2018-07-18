#MagAO-X makefile for INDI utilities.

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../../Make/common.mk
include $(SELF_DIR)config.mk

all: indiserver getINDI setINDI evalINDI

indiserver:
	$(CC) $(CCFLAGS) -o indiserver -I../liblilxml  indiserver.c fq.c ../liblilxml/liblilxml.a -lpthread

getINDI:
	$(CC) $(CCFLAGS) -o getINDI -I../liblilxml  getINDI.c connect_to.c ../liblilxml/liblilxml.a -lz


setINDI:
	$(CC) $(CCFLAGS) -o setINDI -I../liblilxml  setINDI.c connect_to.c ../liblilxml/liblilxml.a

evalINDI:
	$(CC) $(CCFLAGS) -o evalINDI -I../liblilxml  evalINDI.c connect_to.c compiler.c ../liblilxml/liblilxml.a -lm

install: all
	su -c 'install -d $(BIN_PATH) &  \
          install indiserver getINDI setINDI evalINDI $(BIN_PATH) & \
	       ln -sf $(BIN_PATH)/indiserver /usr/local/bin/indiserver & \
			 ln -sf $(BIN_PATH)/getINDI /usr/local/bin/getINDI & \
			 ln -sf $(BIN_PATH)/setINDI /usr/local/bin/setINDI & \
			 ln -sf $(BIN_PATH)/evalINDI /usr/local/bin/evalINDI'

clean:
	rm -f indiserver getINDI setINDI evalINDI
	rm -f *.o
