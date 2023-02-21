CPPFLAGS = -I. -I../../INDI/INDI -I../../libMagAOX/common -Wall $(EXTRACPPFLAGS)

EXES=test/gausswait test/test_resurrector resurrector_indi

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

all: $(EXES)

resurrector_indi :: resurrector.hpp HexbeatMonitor.hpp resurrector_indi.hpp resurrector_indi.cpp
test/test_resurrector :: resurrector.hpp test/test_resurrector.cpp HexbeatMonitor.hpp
test/gausswait :: test/gausswait.cpp

test: all
	for i in a b c d e f g h i j k l m n o p q r s t u v w x y z ; do \
          ii= \
	; for j in 0 1 2 3 ; do \
	    ii=$$ii$$i \
	  ; echo $$ii=test/gausswait \
	  ; done \
	; done \
	| xargs test/test_resurrector

clean:
	$(RM) $(EXES)

self_dir:
	echo $(SELF_DIR)
