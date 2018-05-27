# Makefile for the basic XML routines.
# The idea is to compile and archive them into liblilxml.a

# compiler and flags

# gcc
CC = g++
CFLAGS= -O2 -Wall -g

def: pass1 pass5

# use pass1 because this lib is required for pass3 common librares
pass1: liblilxml.a

pass5: xmlcheck

xmlcheck: xmlcheck.o liblilxml.a
	$(CC) -Wall -o xmlcheck xmlcheck.o -L. -llilxml

install: xmlcheck
	cp -f xmlcheck $(LBTIINDIHOME)/bin

HS = lilxml.h

OBJS = lilxml.o base64.o

liblilxml.a: $(HS) $(OBJS)
	ar r $@ $(OBJS)
	ranlib $@

liltest: liltest.o liblilxml.a
	$(CC) $(LDFLAGS) -o liltest liltest.o liblilxml.a

liltest.o: $(HS) lilxml.c
	$(CC) -DMAIN_TST $(CFLAGS) -c -o liltest.o lilxml.c

clobber:
	touch x.o x.a
	rm -f *.o *.a core liltest xmlcheck
