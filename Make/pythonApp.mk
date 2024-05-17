SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/python.mk

TARGET ?= $(t)

all:
	@echo "*** $(TARGET) is a Python app, skipping 'build' step. Use 'make install' to install. ***"

install:
	sudo $(PYTHON) -c 'import purepyindi2' || (echo "Need purepyindi2 installed to $(shell which python)" && exit 1)
	sudo $(PYTHON) -c 'import xconf' || (echo "Need xconf installed to $(shell which python)" && exit 1)
	sudo $(PYTHON) -m pip install .
	sudo ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(TARGET) /opt/MagAOX/bin/$(TARGET)
	@echo "*** Install succeeded, app located in /opt/MagAOX/bin/$(TARGET) ***"

.PHONY: all install

clean:
	@echo "Nothing to clean for $(TARGET)"