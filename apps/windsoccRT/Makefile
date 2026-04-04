# WindsoCC: Python package install only (C++ MagAO-X apps live under archive/).
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../../Make/python.mk

WINDSOCC_PYTHON_DIR := ./windsocc

.PHONY: all install install-windsocc-python

## Default: no C++ build; install the Python package when needed.
all:
	@echo "WindsoCC: Python-only. Use 'make install-windsocc-python' or: pip install -e $(WINDSOCC_PYTHON_DIR)"

## Install the windsocc package into $(PYTHON)'s environment (e.g. RTC conda env).
install-windsocc-python:
	sudo -H $(PYTHON) -m pip install $(WINDSOCC_PYTHON_DIR)

install: install-windsocc-python
