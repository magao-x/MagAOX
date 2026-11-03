allall: all
OTHER_OBJS=cacaoWriter.o
OTHER_HEADERS=cacaoWriter.hpp reconstructorControls.hpp
NEED_CUDA = yes
TARGET=nnReconstructor
CXXFLAGS += -Wno-deprecated-declarations
CXXFLAGS += -lnvinfer -lcudart
EXTRA_LDFLAGS += -L/opt/MagAOX/vendor/TensorRT-10.0.0.6/lib -L/usr/local/cuda/lib64/
INCLUDES += -I/usr/local/milk/include -I/usr/local/milk/include/CommandLineInterface -I/usr/local/milk/include/ImageStreamIO
INCLUDES += -I/opt/MagAOX/vendor/TensorRT-10.0.0.6/include -I/usr/local/cuda/include
include ../../Make/magAOXApp.mk

cacaoWriter.o: cacaoWriter.cpp cacaoWriter.hpp $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp.gch)
	$(CXX) $(CXXFLAGS) -c $< -o $@
