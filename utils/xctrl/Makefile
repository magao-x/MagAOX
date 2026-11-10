# Makefile for xctrl, which needs only the C++ standard library

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk

TARGET = xctrl

all: $(TARGET)

$(TARGET): xctrl.cpp xctrl.hpp ../../libMagAOX/common/paths.hpp
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ xctrl.cpp

# setuid root so xsup and group magaox can start and stop apps; configuration changes still need sudo
install: all
	sudo install -d $(BIN_PATH)
	sudo install $(TARGET) $(BIN_PATH) --mode=4755 --owner=root
	sudo ln -sf $(BIN_PATH)/$(TARGET) /usr/local/bin/$(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
