allall: all

OTHER_HEADERS=
TARGET=windsoccRT

INCLUDES += $(shell python3-config --includes)
EXTRA_LDFLAGS += $(shell python3-config --embed --ldflags)
EXTRA_LDFLAGS += -Wl,-rpath,/opt/conda/envs/xpy3_13/lib

include ../../Make/magAOXApp.mk
