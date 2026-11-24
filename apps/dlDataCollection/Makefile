allall: all
OTHER_HEADERS=dlCollection.hpp dlObservation.hpp
TARGET=dlDataCollection
# File and image validation must not assume NaN/Inf are impossible.
OPTIMIZE = -fno-fast-math
EXTRA_LDFLAGS += -lcrypto
include ../../Make/magAOXApp.mk
