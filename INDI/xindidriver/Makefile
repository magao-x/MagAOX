
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk

all: xindidriver

xindidriver: xindidriver.cpp
	$(CXX) $(CXXFLAGS) -o xindidriver xindidriver.cpp -lpthread -DXINDID_FIFODIR=\"$(FIFO_PATH)\"

#Note this is not symlinke from /usr/local/bin in a normal MagAO-X install.
install: all
	sudo install -d $(BIN_PATH)
	sudo install xindidriver $(BIN_PATH)

clean:
	rm -f xindidriver
	rm -f *.o
