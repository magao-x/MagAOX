
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(SELF_DIR)/../../Make/common.mk
include $(SELF_DIR)/config.mk

all: xindidriver

xindidriver: xindidriver.cpp
	$(CXX) $(CXXFLAGS) -o xindidriver xindidriver.cpp -lpthread -DXINDID_FIFODIR=\"$(FIFO_PATH)\"

#Note this is not symlinke from /usr/local/bin in a normal MagAO-X install.
install: all
	su -c 'install -d $(BIN_PATH) & \
	       install xindidriver $(BIN_PATH)'

clean:
	rm -f xindidriver
	rm -f *.o
