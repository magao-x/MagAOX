# Definitions for building MagAOX applications
#
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
-include $(SELF_DIR)/../local/magAOXApp.mk
include $(SELF_DIR)/../Make/common.mk

# Provide default build options in case they weren't defined in local/magAOXApp.mk
ifeq ($UNAME,Darwin)  # macOS
    USE_BLAS_FROM ?= Accelerate
else
    USE_BLAS_FROM ?= mkl
endif

#default FFT is fftw
USE_FFT_FROM ?= fftw

# Configure includes and libraries based on build options
ifeq ($(USE_BLAS_FROM),mkl)
    BLAS_INCLUDES ?= -DMXLIB_MKL -m64 -I${MKLROOT}/include
    BLAS_LDFLAGS ?= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed
    BLAS_LDLIBS ?= -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
endif
ifeq ($(USE_BLAS_FROM),ATLAS)
    BLAS_INCLUDES ?= -I/usr/local/atlas/include
    BLAS_LDFLAGS ?= -L/usr/local/atlas/lib
    BLAS_LDLIBS ?= -llapack -lf77blas -lcblas -latlas -lgfortran
endif
ifeq ($(USE_BLAS_FROM),Accelerate)
    BLAS_LDFLAGS ?= -framework Accelerate
endif

ifeq ($(USE_FFT_FROM),fftw)
    #Order matters, _threads first.
    FFT_LDLIBS += -lfftw3_threads -lfftw3f_threads -lfftw3l_threads -lfftw3 -lfftw3f  -lfftw3l 
endif

OPTIMIZE ?= -O3 -fopenmp -ffast-math
EXTRA_LDLIBS ?= -lmxlib -lsofa_c -llevmar -lcfitsio -lboost_system -lboost_filesystem -lgsl
ifneq ($(UNAME),Darwin)
    EXTRA_LDLIBS += -lrt
endif
EXTRA_LDFLAGS ?= -L$(PREFIX)/lib

INCLUDES += $(BLAS_INCLUDES)
LDLIBS += $(EXTRA_LDLIBS) $(BLAS_LDLIBS) $(FFT_LDLIBS) 
LDFLAGS += $(BLAS_LDFLAGS) $(FFT_LDFLAGS) $(EXTRA_LDFLAGS)

#Hard-code the paths to system libraries so setuid works
LDLIBRPATH := $(shell echo $$LD_LIBRARY_PATH | sed 's/::/:/g' |  sed 's/:/ -Wl,-rpath,/g')
LDLIBS += -Wl,-rpath,$(LDLIBRPATH)

#Add libudev for device discovery
#\todo this needs to be a define for compat
LDLIBS += -ludev

CFLAGS += -std=c99 -fPIC $(INCLUDES) $(OPTIMIZE)
CXXFLAGS += -std=c++14 -fPIC $(INCLUDES) $(OPTIMIZE)

#This is needed to force use of g++ for linking
LINK.o = $(LINK.cc)

# Single-file app name can be supplied as `TARGET=`,
# or `t=` for short
TARGET ?= $(t)
	
all:  $(TARGET)

install: all
	install -d $(BIN_PATH)
	install $(TARGET) $(BIN_PATH)

$(TARGET): magaox_git_version.h

.PHONY: magaox_git_version.h
magaox_git_version.h:
	gengithead.sh ../../ ./magaox_git_version.h MAGAOX


.PHONY: clean
clean:
	rm $(TARGET)
	rm -f *.o 
	rm -f *~
