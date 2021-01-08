

       
OBJS = ../apps/streamWriter/tests/streamWriter_test.o \
       ../libMagAOX/app/dev/tests/outletController_tests.o \
        ../libMagAOX/sys/tests/thSetuid_tests.o \
        ../libMagAOX/tty/tests/ttyIOUtils_test.o \
        ../apps/ocam2KCtrl/tests/ocamUtils_test.o \
        ../apps/siglentSDG/tests/siglentSDG_test.o

       
       
##################################################
## Should not need to edit from here on:
##################################################

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../Make/common.mk



########################################
## Targets
#######################################

# Single-file app name can be supplied as `TARGET=`,
# or `t=` for short
TARGET ?= testMagAOX.o

all: $(TARGET) $(OBJS) magaox_git_version.h
	$(LINK.o)  -o testMagAOX $(TARGET) $(OBJS) $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.a) $(LDFLAGS) $(LDLIBS)

#The GIT status header
#This always gets regenerated.
.PHONY: magaox_git_version.h
magaox_git_version.h:
	gengithead.sh $(abspath $(SELF_DIR)/../) $(SELF_DIR)/../magaox_git_version.h MAGAOX
	
.PHONY: clean
clean:
	rm -f testMagaOX
	rm -f $(OBJS)
	rm -f *~
			
	
