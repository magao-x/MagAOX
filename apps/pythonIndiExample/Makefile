APP=pythonIndiExample
PYTHON_SCRIPTS_PREFIX=$(shell python -c "import sysconfig; print(sysconfig.get_path('scripts'))")

all : install

.PHONY: all install

install:
	sudo -H python -c 'import purepyindi2' || (echo "Need purepyindi2 installed to $(shell which python)" && exit 1)
	sudo -H python -c 'import xconf' || (echo "Need xconf installed to $(shell which python)" && exit 1)
	sudo -H python -m pip install .
	sudo -H ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(APP) /opt/MagAOX/bin/$(APP)
	@echo "*** Install succeeded, app located in /opt/MagAOX/bin/$(APP) ***"
