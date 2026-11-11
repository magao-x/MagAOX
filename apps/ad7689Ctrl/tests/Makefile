# Portable real-driver validation with fake GPIO/SPI; never opens a hardware endpoint.
CXX ?= c++
BUILD_DIR ?= /tmp/ad7689Ctrl-tests
CPPFLAGS += -Istubs
CXXFLAGS ?= -O2
override CXXFLAGS += -std=c++20 -fno-fast-math -Wall -Wextra

.PHONY: all test sanitize
all: $(BUILD_DIR)/AD7689_test

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/AD7689_test: AD7689_test.cpp ../dependencies/AD7689.h ../dependencies/AD7689.cpp stubs/gpiod.h stubs/linux/spi/spidev.h ../../../tests/testMain.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) AD7689_test.cpp ../../../tests/testMain.cpp $(LDFLAGS) -o $@

test: all
	$(BUILD_DIR)/AD7689_test

sanitize:
	$(MAKE) test BUILD_DIR=$(BUILD_DIR)/sanitize CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-fsanitize=address,undefined'
