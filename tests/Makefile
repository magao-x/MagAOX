####################################################
# Makefile for building the MagAOX test program
#
# See tests/README.md for instructions.
#
# Manages creation of the git repo status header.
#
####################################################

## Add complete path to tests here:
## NOTE: follow same order as in testMagAOX.cpp!!

TEST_INCLUDES ?= ../libMagAOX/tty/tests/ttyIOUtils_test.cpp \
                 ../libMagAOX/app/dev/tests/powerDevice_tests.cpp \
                 ../apps/ocam2KCtrl/tests/ocamUtils_test.cpp \
                 ../apps/siglentSDG/tests/siglentSDG_test.cpp







##################################################
## Should not need to edit from here on:
##################################################

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/magAOXApp.mk
include $(SELF_DIR)/../Make/common.mk
include $(SELF_DIR)/../Make/config.mk



########################################
## Targets
#######################################

# Single-file app name can be supplied as `TARGET=`,
# or `t=` for short
TARGET ?= testMagAOX

all:  magaox_git_version.h $(TARGET)
	rm ../magaox_git_version.h

$(TARGET): $(TARGET).o $(TARGET).cpp $(TEST_INCLUDES)
	$(LINK.o) -o $(TARGET) $(TARGET).o  $(LDFLAGS) $(LDLIBS)


#The GIT status header
#This always gets regenerated.
.PHONY: magaox_git_version.h
magaox_git_version.h:
	gengithead.sh $(abspath ../) ../magaox_git_version.h MAGAOX


.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *~
