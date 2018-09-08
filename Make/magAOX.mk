####################################################
# Makefile for building MagAOX programs
#
# Manages dependance on the libMagAOX pre-compiled header
# as well as local files (such as .hpp)
#
# Manages creation of the git repo status header.
#
# In principle this works standalone for a single-header application:
#    -- only an <app-name>.hpp and <app-name>.cpp are needed
#    -- in the directory containing those files, invoke make with
#         make -f ../../Make/magAOX.mk t=<app-name>
#    -- See the files "magAOXApp.mk" and "magAOXUtil.mk" for specializations of this for Apps and Utilities.

# More complicated builds are also supported:
#    -- In a local Makefile, make the first rule:
#          allall: all
#    -- Then specify rules for building other .o files
#       --- Copy the $(TARGETS).o rule to depend on the pch.
#    -- List those .o files after OTHER_OBJS=
#    -- List any header dependencies after OTHER_HEADERS
#    -- Define TARGET=
#    -- Then finally include this file (or magAOXApp.mk or magAOXUtil.mk)
####################################################
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../Make/common.mk
include $(SELF_DIR)/../Make/config.mk

#To ensure pre-compiled header gets used
CXXFLAGS += -include $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp)

#Uncomment to test whether pre-compiled header is used
# CXXFLAGS += -H


########################################
## Targets
#######################################

# Single-file app name can be supplied as `TARGET=`,
# or `t=` for short
TARGET ?= $(t)

all:  pch magaox_git_version.h $(TARGET)
	rm magaox_git_version.h

pch:
	cd ../../libMagAOX; ${MAKE}

$(TARGET).o: $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp.gch) $(TARGET).hpp $(OTHER_HEADERS)

$(TARGET):  $(TARGET).o  $(OTHER_OBJS)
	$(LINK.o)  -o $(TARGET) $(TARGET).o $(OTHER_OBJS) $(LDFLAGS) $(LDLIBS)


#The GIT status header
#This always gets regenerated.
.PHONY: magaox_git_version.h
magaox_git_version.h:
	gengithead.sh $(abspath $(SELF_DIR)/../) ./magaox_git_version.h MAGAOX

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f magaox_git_version.h
	rm -f *.o
	rm -f *~
