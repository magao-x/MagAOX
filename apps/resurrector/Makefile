
allall: all

INCLUDES +=  -I../../libMagAOX/common -Wall $(EXTRACPPFLAGS)
OTHER_HEADERS=resurrector.hpp HexbeatMonitor.hpp resurrector_indi.hpp
OTHER_OBJS=redirect_prototype.o
TARGET=resurrector_indi
include ../../Make/magAOXApp.mk

########################################################################
########################################################################


CPPFLAGS = -I. -I../../INDI/INDI -I../../libMagAOX/common -Wall $(EXTRACPPFLAGS)

EXES=test/gausswait test/test_resurrector

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

lclall: $(EXES)

#resurrector_indi :: resurrector.hpp HexbeatMonitor.hpp redirect_prototype.cpp resurrector_indi.hpp resurrector_indi.cpp
test/test_resurrector :: resurrector.hpp test/test_resurrector.cpp HexbeatMonitor.hpp
test/gausswait :: test/gausswait.cpp

lcltest: lclall
	for i in a b c d e f g h i j k l m n o p q r s t u v w x y z ; do \
          ii= \
	; for j in 0 1 2 3 ; do \
	    ii=$$ii$$i \
	  ; echo $$ii=test/gausswait \
	  ; done \
	; done \
	| xargs test/test_resurrector

lclclean:
	$(RM) $(EXES)

self_dir:
	echo $(SELF_DIR)
