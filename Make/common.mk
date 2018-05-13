# Common path and make variable definitions
#
# NOTE: This file should only be edited in mxlib/local, not in the root mxlib directory.
#
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
-include $(SELF_DIR)/../local/Common.mk

UNAME ?= $(shell uname)
ifeq ($(UNAME),Darwin)
	CFLAGS += -D_BSD_SOURCE
	CXXFLAGS += -D_BSD_SOURCE
else
	CFLAGS += -D_XOPEN_SOURCE=700
	CXXFLAGS += -D_XOPEN_SOURCE=700
endif
PREFIX ?= $(HOME)
BIN_PATH ?= $(PREFIX)/bin
LIB_PATH ?= $(PREFIX)/lib
INCLUDE_PATH ?= $(PREFIX)/include
LIB_SOFA ?= $(LIB_PATH)/libsofa_c.a
ARFLAGS ?= rvs

INCLUDES += -I$(INCLUDE_PATH)
