allall: all

OTHER_HEADERS=
TARGET=windsoccRT

INCLUDES += $(shell python3-config --includes)
EXTRA_LDFLAGS += $(shell python3-config --embed --ldflags)
EXTRA_LDFLAGS += -Wl,-rpath,/opt/conda/envs/xpy3_13/lib

# Same interpreter MagAO-X uses for Python apps (see Make/python.mk).
include ../../Make/python.mk

include ../../Make/magAOXApp.mk

# Path to the installable WindsoCC package (pyproject.toml lives here).
WINDSOCC_PYTHON_DIR := ./windsocc
.PHONY: install-windsocc-python
## Install the windsocc package into $(PYTHON)'s environment (e.g. RTC conda env).
install-windsocc-python:
	sudo -H $(PYTHON) -m pip install $(WINDSOCC_PYTHON_DIR)
## Override magAOXApp.mk: install C++ binary and Python package together.
install: all install-windsocc-python
	sudo install -d $(BIN_PATH)
	sudo install $(TARGET) $(BIN_PATH) --mode=4755 --owner=root