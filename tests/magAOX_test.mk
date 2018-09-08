####################################################
# Makefile for building MagAOX test programs
#
# Manages dependance on the libMagAOX pre-compiled header
# as well as local files (such as .hpp)
#
# Manages creation of the git repo status header.
#
# This works standalone for a single-header application:
#    -- only an <app-name>.hpp and <app-name>.cpp are needed
#    -- in the directory containing those files, invoke make with
#         make -f ../../Make/magAOXApp.mk t=<app-name>
#
# More complicated builds are also supported:
#    -- In a local Makefile, make the first rule:
#          allall: all
#    -- Then specify rules for building other .o files
#       --- Copy the $(TARGETS).o rule to depend on the pch.
#    -- List those .o files after OTHER_DEPENDS=
#    -- Define TARGET=
#    -- Then include this file.
####################################################
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/magAOXApp.mk
include $(SELF_DIR)/../Make/common.mk
include $(SELF_DIR)/../Make/config.mk

#To ensure pre-compiled header gets used
CXXFLAGS += -include $(abspath $(SELF_DIR)/../tests/testMagAOX.hpp)

#Uncomment to test whether pre-compiled header is used
CXXFLAGS += -H


########################################
## Targets
#######################################

# Single-file app name can be supplied as `TARGET=`,
# or `t=` for short
TARGET ?= $(t)

all:  pch magaox_git_version.h $(TARGET)

pch:
	cd $(SELF_DIR)/../tests; ${MAKE}


$(TARGET):  $(TARGET).o  $(SELF_DIR)/../tests/testMagAOX.hpp.gch $(TARGET).cpp $(OTHER_HEADERS)
	$(LINK.o) -o $(TARGET) $(TARGET).o  $(LDFLAGS) $(LDLIBS)


#The GIT status header
#This always gets regenerated.
.PHONY: magaox_git_version.h
magaox_git_version.h:
	gengithead.sh $(abspath ../../../) ../magaox_git_version.h MAGAOX


.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *~
