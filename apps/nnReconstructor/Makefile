allall: all
OTHER_OBJS=cacaoWriter.o
OTHER_HEADERS=cacaoWriter.hpp reconstructorControls.hpp
NEED_CUDA = yes
TARGET=nnReconstructor
CXXFLAGS += -Wno-deprecated-declarations
CXXFLAGS += -lnvinfer -lcudart
EXTRA_LDFLAGS += -L/opt/MagAOX/vendor/TensorRT-10.0.0.6/lib -L/usr/local/cuda/lib64/
CACAO_INCLUDES=-I/usr/local/milk/include -I/usr/local/milk/include/CommandLineInterface -I/usr/local/milk/include/ImageStreamIO
INCLUDES += -I/opt/MagAOX/vendor/TensorRT-10.0.0.6/include -I/usr/local/cuda/include
include ../../Make/magAOXApp.mk

# Keep milk's BOOL separate from EDT's BOOL in the forced MagAO-X header.
# Select one milk installation before any include paths supplied by mxlib.
cacaoWriter.o: cacaoWriter.cpp cacaoWriter.hpp Makefile
	$(CXX) $(CACAO_INCLUDES) $(subst -include $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp),,$(CXXFLAGS)) -c $< -o $@
