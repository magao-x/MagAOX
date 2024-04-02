APP=visxCtrl
PYTHON_SCRIPTS_PREFIX=$(shell python -c "import sysconfig; print(sysconfig.get_path('scripts'))")

all : install

.PHONY: all install

install:
	@if [[ -z "$(USER)" || "$(USER)" == "root" ]]; then echo "*** NOTE: Use 'make install', not 'sudo make install'. ***"; exit 1; fi
	sudo python -c 'import purepyindi2' || (echo "Need purepyindi2 installed to $(shell which python)" && exit 1)
	sudo python -c 'import xconf' || (echo "Need xconf installed to $(shell which python)" && exit 1)
	sudo python -m pip install -e .
	sudo ln -sfv $(PYTHON_SCRIPTS_PREFIX)/$(APP) /opt/MagAOX/bin/$(APP)
	@echo "*** Editable install succeeded, app located in /opt/MagAOX/bin/$(APP) ***"
	@echo "*** NOTE: This step only needs to be run once; edits to $(PWD) will take effect immediately without reinstallation ***"
