# Definitions for pre-compiling the libMagAOX header
#
-include ../local/magAOXApp.mk
include ../Make/common.mk

# Provide default build options in case they weren't defined in local/magAOXApp.mk
ifeq ($UNAME,Darwin)  # macOS
    USE_BLAS_FROM ?= Accelerate
else
    USE_BLAS_FROM ?= mkl
endif

# Configure includes and libraries based on build options
ifeq ($(USE_BLAS_FROM),mkl)
    BLAS_INCLUDES ?= -DMXLIB_MKL -m64 -I${MKLROOT}/include
endif

ifeq ($(USE_BLAS_FROM),ATLAS)
    BLAS_INCLUDES ?= -I/usr/local/atlas/include
endif


OPTIMIZE ?= -O3 -fopenmp -ffast-math

INCLUDES += $(BLAS_INCLUDES)


CFLAGS += -std=c99 -fPIC $(INCLUDES) $(OPTIMIZE)
CXXFLAGS += -std=c++14 -fPIC $(INCLUDES) $(OPTIMIZE)

TARGET=libMagAOX.hpp
INCLUDEDEPS= app/MagAOXApp.hpp \
             app/stateCodes.hpp \
             common/config.hpp \
             common/defaults.hpp \
             common/environment.hpp \
             logger/logBuffer.hpp \
             logger/logCodes.hpp \
             logger/logFileRaw.hpp \
             logger/logLevels.hpp \
             logger/logManager.hpp \
             logger/logStdFormat.hpp \
             logger/logTypes.hpp \
             logger/logTypesBasics.hpp \
             time/timespecX.hpp \
             tty/ttyErrors.hpp \
             tty/ttyIOUtils.hpp \
             tty/ttyUSB.hpp \
             tty/usbDevice.hpp

$(TARGET).gch: $(TARGET) $(INCLUDEDEPS)
	$(CXX) $(CXXFLAGS) -c $(TARGET)


.PHONY: clean
clean:
	rm -f $(TARGET).gch
	rm -f *~
