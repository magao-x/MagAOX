
allall: all

OTHER_HEADERS=
TARGET=xrif2fits
LDLIBS += -lcfitsio
#OPTIMIZE=-ggdb

ifeq ($(DEBUG),1)
  CXXFLAGS += -DXRIF2FITS_DEBUG
endif

include ../../Make/magAOXUtil.mk
