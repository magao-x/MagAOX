TARGET=po4aoTrainer
# Keep heavy training dependencies out of the general instrument environment.
override INSTRUMENT_CONDA_ENV := $(if $(INSTRUMENT_CONDA_ENV),$(INSTRUMENT_CONDA_ENV),po4aoTrainer)
include ../../Make/pythonApp.mk
