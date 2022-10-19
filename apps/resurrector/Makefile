CPPFLAGS = -I../../INDI/INDI -Wall

EXES=test/gausswait resurrector

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

all: $(EXES)

resurrector :: resurrector.hpp resurrector.cpp HexbeatMonitor.hpp
test/gausswait :: test/gausswait.cpp

test: all
	for i in a b c d e f g h i j k l m n o p q r s t u v w x y z ; do \
          ii= \
	; for j in 0 1 2 3 ; do \
	    ii=$$ii$$i \
	  ; echo $$ii=test/gausswait \
	  ; done \
	; done \
	| xargs ./resurrector

clean:
	$(RM) $(EXES)

self_dir:
	echo $(SELF_DIR)
