SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/python.mk
ifeq ($(INSTRUMENT_CONDA_ENV),)
    PYTHON ?= python
else
    PYTHON := /opt/conda/envs/$(INSTRUMENT_CONDA_ENV)/bin/python
endif
PYTHON_SCRIPTS_PREFIX=$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('scripts'))")
