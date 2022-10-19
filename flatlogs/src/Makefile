
all: flatlogcodes

flatlogcodes: flatlogcodes.cpp
	g++ -o flatlogcodes flatlogcodes.cpp

install: all
	mkdir -p ../bin
	install flatlogcodes ../bin

.PHONY: clean
clean:
	rm -f flatlogcodes
	rm -f *.o
