sratio: cstrehl.cpp
	g++ -march=native -Ofast -o sratio cstrehl.cpp -lfftw3 -lm

data: sa.dat
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 sa.dat sa.o
	mv sa.o ..

.PHONY: clean

clean:
	rm sratio *.o
