
all: flatlogcodes

flatlogcodes: flatlogcodes.cpp
	g++ -o flatlogcodes flatlogcodes.cpp

install: all
	install -d ../bin 
	install flatlogcodes ../bin

.PHONY: clean
clean:
	rm -f flatlogcodes 
	rm -f *.o
