SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/python.mk

PYTHON ?= /opt/conda/bin/python
PYTHON_SCRIPTS_PREFIX=$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('scripts'))")
