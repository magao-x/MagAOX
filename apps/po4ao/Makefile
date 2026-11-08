allall: all
OTHER_OBJS=po4aoRouteIO.o cacaoWriter.o
OTHER_HEADERS=po4ao_impl.hpp po4ao_control.hpp po4ao_utils.hpp po4aoStatistics.hpp po4aoRouting.hpp po4aoRouteIO.hpp po4ao_run.hpp
NEED_CUDA = yes
TARGET=po4ao
CXXFLAGS += -Wno-deprecated-declarations
CXXFLAGS += -lnvinfer -lcudart
EXTRA_LDFLAGS += -L/opt/MagAOX/vendor/TensorRT-10.0.0.6/lib -L/usr/local/cuda/lib64/
INCLUDES += -I/opt/MagAOX/vendor/TensorRT-10.0.0.6/include -I/usr/local/cuda/include
include ../../Make/magAOXApp.mk

# Isolate the installed milk ABI from EDT's conflicting BOOL typedef.
CACAO_INCLUDES=-I/usr/local/milk/include -I/usr/local/milk/include/CommandLineInterface -I/usr/local/milk/include/ImageStreamIO
po4aoRouteIO.o: po4aoRouteIO.cpp po4aoRouteIO.hpp po4aoRouting.hpp Makefile
	$(CXX) $(CACAO_INCLUDES) $(subst -include $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp),,$(CXXFLAGS)) -c $< -o $@

cacaoWriter.o: ../nnReconstructor/cacaoWriter.cpp ../nnReconstructor/cacaoWriter.hpp Makefile
	$(CXX) $(CACAO_INCLUDES) $(subst -include $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp),,$(CXXFLAGS)) -c $< -o $@
