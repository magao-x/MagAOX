# WindsoCC realtime: handbook-style Python INDI app (``magaox.indi.device.XDevice``).
# Installs the sibling ``windsocc`` package first, then this app and
# ``/opt/MagAOX/bin/windsoccRT`` for ``xctrl`` / ``proclist_RTC``.
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../../Make/python.mk

TARGET := windsoccRT
WINDSOCC_PKG_DIR := $(SELF_DIR)/windsocc

.PHONY: all clean check pipinstall normalize_permissions install install-windsocc-python

all:
	@echo "*** $(TARGET): Python INDI app — use 'make install' ***"

clean:
	rm -rf $(SELF_DIR)/build
	rm -rf $(SELF_DIR)/xapp_windsoccRT.egg-info

check:
	$(PYTHON) -c 'import purepyindi2' || (echo "Need purepyindi2 installed to $(PYTHON)" && exit 1)
	$(PYTHON) -c 'import xconf' || (echo "Need xconf installed to $(PYTHON)" && exit 1)

pipinstall: check
	sudo -H $(PYTHON) -m pip install "$(WINDSOCC_PKG_DIR)" || (echo "Unable to install windsocc from $(WINDSOCC_PKG_DIR)" && exit 1)
	sudo -H $(PYTHON) -m pip install "$(SELF_DIR)" || (echo "Unable to install $(TARGET) with $(PYTHON)" && exit 1)
	sudo -H ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(TARGET) /opt/MagAOX/bin/$(TARGET) || (echo "Couldn't symlink /opt/MagAOX/bin/$(TARGET)" && exit 1)

normalize_permissions: pipinstall
	sudo chmod -R o+rX $(shell cd && $(PYTHON) -c "import xapp.windsoccRT, os.path; print(os.path.dirname(xapp.windsoccRT.__file__))")

install: normalize_permissions
	@echo "*** Install succeeded, app located in /opt/MagAOX/bin/$(TARGET) ***"

install-windsocc-python:
	sudo -H $(PYTHON) -m pip install "$(WINDSOCC_PKG_DIR)"