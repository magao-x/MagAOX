# Makefile for resuctrl, which needs only the C++ standard library

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk

TARGET = resuctrl

all: $(TARGET)

$(TARGET): resuctrl.cpp resuctrl.hpp ../../libMagAOX/common/paths.hpp
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ resuctrl.cpp

# setuid root so xsup and group magaox can start and stop apps; configuration changes still need sudo.
# xctrl is the same program.
install: all
	sudo install -d $(BIN_PATH)
	sudo install $(TARGET) $(BIN_PATH) --mode=4755 --owner=root
	sudo ln -sf $(TARGET) $(BIN_PATH)/xctrl
	sudo ln -sf $(BIN_PATH)/$(TARGET) /usr/local/bin/$(TARGET)
	sudo ln -sf $(BIN_PATH)/$(TARGET) /usr/local/bin/xctrl

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
