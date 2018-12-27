
TARGET = singleTest
TEST_INCLUDES = $(testfile)

CXXFLAGS+= -D"TEST_FILE \"$(testfile)\""

include Makefile
