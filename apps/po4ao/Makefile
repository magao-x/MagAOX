allall: all
OTHER_HEADERS=po4ao_impl.hpp po4ao_utils.hpp
NEED_CUDA = yes
TARGET=po4ao
CXXFLAGS += -Wno-deprecated-declarations
CXXFLAGS += -lnvinfer -lcudart
EXTRA_LDFLAGS += -L/opt/MagAOX/vendor/TensorRT-10.0.0.6/lib -L/usr/local/cuda/lib64/
INCLUDES += -I/opt/MagAOX/vendor/TensorRT-10.0.0.6/include -I/usr/local/cuda/include
include ../../Make/magAOXApp.mk
