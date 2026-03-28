allall: all

OTHER_HEADERS=
TARGET=windsoccRT
PROBE_TARGET=windsoccImportProbe
PROBE_OBJ=$(PROBE_TARGET).o

INCLUDES += $(shell python3-config --includes)
EXTRA_LDFLAGS += $(shell python3-config --embed --ldflags)
EXTRA_LDFLAGS += -Wl,-rpath,/opt/conda/envs/xpy3_13/lib

# Same interpreter MagAO-X uses for Python apps (see Make/python.mk).
include ../../Make/python.mk

include ../../Make/magAOXApp.mk

all: $(PROBE_TARGET)

$(PROBE_OBJ): $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp.gch) $(PROBE_TARGET).cpp

$(PROBE_TARGET): $(PROBE_OBJ)
	$(LINK.o) -o $(PROBE_TARGET) $(PROBE_OBJ) $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.a) $(LDFLAGS) $(LDLIBS)

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
	sudo install $(PROBE_TARGET) $(BIN_PATH) --mode=0755 --owner=root

.PHONY: clean-probes
clean: clean-probes

clean-probes:
	rm -f $(PROBE_TARGET)