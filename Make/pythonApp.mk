SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/python.mk

TARGET ?= $(t)

all:
	@echo "*** $(TARGET) is a Python app, skipping 'build' step. Use 'make install' to install. ***"

.PHONY: clean
clean:
	rm -rf $(SELF_DIR)/build
	rm -rf $(SELF_DIR)/xapp_$(TARGET).egg-info

.PHONY: check
check:
	$(PYTHON) -c 'import purepyindi2' || (echo "Need purepyindi2 installed to $(PYTHON)" && exit 1)
	$(PYTHON) -c 'import xconf' || (echo "Need xconf installed to $(PYTHON)" && exit 1)

.PHONY: pipinstall
pipinstall: check
	sudo -H $(PYTHON) -m pip install . || (echo "Unable to install $(TARGET) with $(PYTHON)" && exit 1)
	sudo -H ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(TARGET) /opt/MagAOX/bin/$(TARGET) || (echo "Couldn't symlink /opt/MagAOX/bin/$(TARGET) to $(PYTHON_SCRIPTS_PREFIX)/$(TARGET)" && exit 1)
	cd && $(PYTHON) -c "import xapp.$(TARGET)" || (echo "Unable to import installed app" && exit 1)

.PHONY: normalize_permissions
normalize_permissions: pipinstall
	sudo chmod -R o+rX $$(cd && $(PYTHON) -c "import xapp.$(TARGET), os.path;print(os.path.dirname(xapp.$(TARGET).__file__))")

.PHONY: install
install: normalize_permissions
	@echo "*** Install succeeded, app located in /opt/MagAOX/bin/$(TARGET) ***"
