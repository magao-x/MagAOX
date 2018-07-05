SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)/../Make/magAOX.mk

install: all
	su -c 'install -d $(BIN_PATH) & install $(TARGET) $(BIN_PATH)'
