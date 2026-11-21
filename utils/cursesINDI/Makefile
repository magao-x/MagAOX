####################################################
# Makefile for cursesINDI
#
# cursesINDI only needs the INDI client library, so it is built standalone
# instead of through magAOXUtil.mk, which force-includes the libMagAOX
# pre-compiled header and links libMagAOX.  This lets INDI-client-only
# machines (see the top-level `client` target) build it without libMagAOX.
#
# Requires INDI/libcommon and INDI/liblilxml to be built first (`make indi_all`).
####################################################

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
INDI_DIR := $(abspath $(SELF_DIR)/../../INDI)

TARGET = cursesINDI

OPTIMIZE = -ggdb
CXXSTD = -std=c++20

#Replaces the libMagAOX link set from common.mk
EXTRA_LDLIBS = $(INDI_DIR)/libcommon/libcommon.a \
               $(INDI_DIR)/liblilxml/liblilxml.a \
               -lncurses \
               -lpthread

include $(SELF_DIR)/../../Make/common.mk

all: $(TARGET)

$(TARGET).o: $(TARGET).hpp cursesTableGrid.hpp $(INDI_DIR)/libcommon/IndiClient.hpp

$(TARGET): $(TARGET).o
	$(LINK.o) -o $(TARGET) $(TARGET).o $(LDFLAGS) $(LDLIBS)

install: all
	sudo install -d $(BIN_PATH)
	sudo install $(TARGET) $(BIN_PATH)
	sudo ln -sf $(BIN_PATH)/$(TARGET) /usr/local/bin/$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f *~
