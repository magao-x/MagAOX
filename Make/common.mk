# Common path and make variable definitions
#
# NOTE: This file should only be edited in mxlib/local, not in the root mxlib directory.
#
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../local/config.mk
-include $(SELF_DIR)/../local/Common.mk

UNAME ?= $(shell uname)
ifeq ($(UNAME),Darwin)
	CFLAGS += -D_BSD_SOURCE
	CXXFLAGS += -D_BSD_SOURCE
else
	CFLAGS += -D_XOPEN_SOURCE=700
	CXXFLAGS += -D_XOPEN_SOURCE=700
endif
#PREFIX ?= /opt/MagAOX   #-- now set in local/config.mk
#BIN_PATH ?= $(PREFIX)/bin #-- now set in local/config.mk

LIB_PATH ?= $(PREFIX)/lib
INCLUDE_PATH ?= $(PREFIX)/include
LIB_SOFA ?= $(LIB_PATH)/libsofa_c.a
ARFLAGS ?= rvs

MXLIB_PREFIX ?= $(HOME)
MXLIB_LIB_PATH ?= $(MXLIB_PREFIX)/lib 
MXLIB_INCLUDE_PATH ?= $(MXLIB_PREFIX)/include 

INCLUDES += -I$(INCLUDE_PATH) -I$(MXLIB_INCLUDE_PATH)


########################################
## Optimize Flags
#######################################
OPTIMIZE ?= -O3 -fopenmp -ffast-math

########################################
## Libraries
#######################################

#location of liblilxml, libindicommon, mxlib and sofa:
EXTRA_LDFLAGS ?=  -L$(MXLIB_LIB_PATH)    

#the required librarires
EXTRA_LDLIBS ?= -lmxlib -lsofa_c -lboost_system -lboost_filesystem -ludev  ../../INDI/libcommon/libcommon.a ../../INDI/liblilxml/liblilxml.a

#Add rt on Darwin:
ifneq ($(UNAME),Darwin)
    EXTRA_LDLIBS += -lrt
endif

LDLIBS += $(EXTRA_LDLIBS) 
LDFLAGS += $(EXTRA_LDFLAGS)

#Hard-code the paths to system libraries so setuid works
LDLIBRPATH := $(shell echo $$LD_LIBRARY_PATH | sed 's/::/:/g' |  sed 's/:/ -Wl,-rpath,/g')
LDLIBS += -Wl,-rpath,$(LDLIBRPATH)

########################################
## Compilation and linking
#######################################

CFLAGS += -std=c99 -fPIC $(INCLUDES) $(OPTIMIZE)
CXXFLAGS += -std=c++14 -fPIC $(INCLUDES) $(OPTIMIZE) 

#This is needed to force use of g++ for linking
LINK.o = $(LINK.cc)

#Create an implicit rule for pre-compiled headers
%.hpp.gch: %.hpp
	$(CXX) $(CXXFLAGS) -c $<


