TARGET=audibleAlerts

include ../../Make/pythonApp.mk

.PHONY: pipinstall-client
pipinstall-client:
	sudo -H ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(TARGET)Client /opt/MagAOX/bin/$(TARGET)Client || (echo "Couldn't symlink /opt/MagAOX/bin/$(TARGET)Client to $(PYTHON_SCRIPTS_PREFIX)/$(TARGET)Client" && exit 1)

.PHONY: install
install: normalize_permissions pipinstall-client
	@echo "*** Install succeeded, app located in /opt/MagAOX/bin/$(TARGET) ***"
