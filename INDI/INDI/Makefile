#MagAO-X makefile for INDI utilities.

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk
LILXMLDIR=../liblilxml

all: indiserver getINDI setINDI evalINDI

indiserver: strcat_varargs.c strcat_varargs.h open_named_fifo.c open_named_fifo.h config.h indiapi.h indidevapi.h fq.c fq.h $(LILXMLDIR)/liblilxml.a indiserver.c
	$(CC) $(CFLAGS) -DGIT_TAG_STRING='"$(shell git describe --tags)"' -g -o $@ -I$(LILXMLDIR) $(shell pkg-config --cflags zlib) indiserver.c strcat_varargs.c open_named_fifo.c fq.c $(LILXMLDIR)/liblilxml.a $(shell pkg-config --libs zlib)


getINDI: connect_to.h connect_to.c indiapi.h getINDI.c
	$(CC) $(CFLAGS) -o getINDI -I../liblilxml  getINDI.c connect_to.c ../liblilxml/liblilxml.a -lz


setINDI: connect_to.h connect_to.c indiapi.h setINDI.c
	$(CC) $(CFLAGS) -o setINDI -I../liblilxml  setINDI.c connect_to.c ../liblilxml/liblilxml.a

evalINDI: connect_to.h connect_to.c indiapi.h evalINDI.c
	$(CC) $(CFLAGS) -o evalINDI -I../liblilxml  evalINDI.c connect_to.c compiler.c ../liblilxml/liblilxml.a -lm

install: all
	sudo install -d $(BIN_PATH)
	sudo install indiserver getINDI setINDI evalINDI $(BIN_PATH)
	sudo ln -sf $(BIN_PATH)/indiserver /usr/local/bin/indiserver
	sudo ln -sf $(BIN_PATH)/getINDI /usr/local/bin/getINDI
	sudo ln -sf $(BIN_PATH)/setINDI /usr/local/bin/setINDI
	sudo ln -sf $(BIN_PATH)/evalINDI /usr/local/bin/evalINDI

clean:
	rm -f indiserver getINDI setINDI evalINDI old*.indiserver
	rm -f *.o
