allall: all

C_FILES=reg_functions.cpp cc-functions.cpp
H_FILES=reg_functions.hpp cc-functions.hpp

func: $(C_FILES) $(H_FILES)
	gcc -O3 -ffast-math -march=native -c $(C_FILES) -lfftw3 -lm 

sa.o: 
	cd STREHL && make data

OPTIMIZE = -O3 -fopenmp -ffast-math -march=native -lgsl -lgslcblas

OTHER_HEADERS=
OTHER_OBJS=reg_functions.o cc-functions.o sa.o
TARGET=imgChar
include ../../Make/magAOXApp.mk
