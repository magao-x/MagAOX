allall: all 


H_FILES = buffer.hpp welchpsd.hpp
C_FILES = buffer.cpp welchpsd.cpp

func: $(H_FILES) $(C_FILES)
	gcc -O3 -ffast-math -march=native -c $(C_FILES) -lfftw3 -lm -lpthread

OPTIMIZE = -O3 -fopenmp -ffast-math -march=native 

OTHER_HEADERS=
OTHER_OBJ=buffer.o welchpsd.o
TARGET=camtipPSD
include ../../Make/magAOXApp.mk
