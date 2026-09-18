TARGET=stateruletree

SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/python.mk

all:
	@echo "*** $(TARGET) is a Python utility. Use 'make install' to install. ***"

.PHONY: clean
clean:
	rm -rf build *.egg-info

.PHONY: install
install:
	sudo -H $(PYTHON) -m pip install .
	sudo -H ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(TARGET) /opt/MagAOX/bin/$(TARGET)
	@echo "*** Install succeeded, $(TARGET) available at /opt/MagAOX/bin/$(TARGET) ***"
