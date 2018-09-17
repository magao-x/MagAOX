#MagAO-X makefile for INDI utilities.

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk
include $(SELF_DIR)/config.mk

all: indiserver getINDI setINDI evalINDI

indiserver: indiapi.h fq.h fq.c indiserver.c
	$(CC) $(CCFLAGS) -o indiserver -I../liblilxml  indiserver.c fq.c ../liblilxml/liblilxml.a -lpthread

getINDI: connect_to.h connect_to.c indiapi.h getINDI.c
	$(CC) $(CCFLAGS) -o getINDI -I../liblilxml  getINDI.c connect_to.c ../liblilxml/liblilxml.a -lz


setINDI: connect_to.h connect_to.c indiapi.h setINDI.c
	$(CC) $(CCFLAGS) -o setINDI -I../liblilxml  setINDI.c connect_to.c ../liblilxml/liblilxml.a

evalINDI: connect_to.h connect_to.c indiapi.h evalINDI.c
	$(CC) $(CCFLAGS) -o evalINDI -I../liblilxml  evalINDI.c connect_to.c compiler.c ../liblilxml/liblilxml.a -lm

install: all
	/bin/sudo install -d $(BIN_PATH)
	/bin/sudo install indiserver getINDI setINDI evalINDI $(BIN_PATH)
	/bin/sudo ln -sf $(BIN_PATH)/indiserver /usr/local/bin/indiserver
	/bin/sudo ln -sf $(BIN_PATH)/getINDI /usr/local/bin/getINDI
	/bin/sudo ln -sf $(BIN_PATH)/setINDI /usr/local/bin/setINDI
	/bin/sudo ln -sf $(BIN_PATH)/evalINDI /usr/local/bin/evalINDI

clean:
	rm -f indiserver getINDI setINDI evalINDI
	rm -f *.o
