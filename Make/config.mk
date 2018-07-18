
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
-include $(SELF_DIR)/../local/config.mk

PREFIX?=/opt/MagAOX

BIN_PATH?=$(PREFIX)/bin
DRIVER_PATH?=$(PREFIX)/drivers
FIFO_PATH?=$(PREFIX)/drivers/fifos
