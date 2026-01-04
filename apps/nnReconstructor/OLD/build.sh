#!/usr/bin/env bash
set  -xe
#  -fsanitize=address \
g++ \
 -Wno-deprecated-declarations \
 -D_XOPEN_SOURCE=700 \
 -I/usr/local/cuda/include \
 -lnvinfer \
 -lcudart \
 -I/opt/MagAOX/vendor/TensorRT-10.0.0.6/include \
 -c \
 -o nnReconstructor.o nnReconstructor.cpp
#  -DEIGEN_NO_CUDA \
#  -DHAVE_CUDA \
#  -DMXLIB_MILK \
#  -std=c++17 \
#  -Wall \
#  -Wextra \
#  -fPIC \
#  -I/usr/local/include \
#  -I/opt/MagAOX/source/MagAOX/flatlogs/include \
#  -I/usr/local/include/eigen3  \
#  -I/usr/local/cuda/include \
#  -I/usr/local/milk/include  \
#  -I/opt/EDTpdv \
#  -O3 \
#  -fopenmp \
#  -ffast-math \
#  -include /opt/MagAOX/source/MagAOX/libMagAOX/libMagAOX.hpp   \
g++ \
 -Wno-deprecated-declarations \
 -lnvinfer \
 -lcudart \
 -L/opt/MagAOX/vendor/TensorRT-10.0.0.6/lib \
 -L/usr/local/cuda/lib64/   \
 -fsanitize=address \
 -o ./nnReconstructor ./nnReconstructor.o  /opt/MagAOX/source/MagAOX/libMagAOX/libMagAOX.a

#  -D_XOPEN_SOURCE=700 \
#  -DEIGEN_NO_CUDA \
#  -DHAVE_CUDA \
#  -DMXLIB_MILK \
#  -std=c++17 \
#  -Wall \
#  -Wextra \
#  -fPIC \
#  -I/opt/MagAOX/vendor/TensorRT-10.0.0.6/include \
#  -I/usr/local/cuda/include \
#  -I/usr/local/include \
#  -I/usr/local/cuda/include \
#  -I/opt/MagAOX/source/MagAOX/flatlogs/include \
#  -I/usr/local/include/eigen3  \
#  -I/usr/local/milk/include  \
#  -I/opt/EDTpdv \
#  -O3 \
#  -fopenmp \
#  -ffast-math \
#  -include /opt/MagAOX/source/MagAOX/libMagAOX/libMagAOX.hpp  \
#  -L/usr/local/lib    \
#  -L/opt/MagAOX/vendor/TensorRT-10.0.0.6/lib \
#  -L/usr/local/cuda/lib64/   \
#  -L/usr/local/lib  \
#  -lmxlib \
#  -ludev \
#  -lpthread \
#  -ltelnet \
#  -lcfitsio \
#  -lxrif \
#  -lfftw3 \
#  -lfftw3f \
#  -lfftw3l \
#  -lgsl \
#  -lboost_system \
#  -lboost_filesystem /opt/MagAOX/source/MagAOX/INDI/libcommon/libcommon.a /opt/MagAOX/source/MagAOX/INDI/liblilxml/liblilxml.a \
#  -lfftw3q  \
#  -L/usr/local/cuda/lib64/ \
#  -lcudart \
#  -lcublas \
#  -lcufft \
#  -lcurand \
#  -L/usr/local/milk/lib \
#  -lImageStreamIO \
#  -lopenblas  \
#  -L/opt/EDTpdv \
#  -lpdv \
#  -lpthread \
#  -lm \
#  -ldl \
#  -Wl,-rpath,