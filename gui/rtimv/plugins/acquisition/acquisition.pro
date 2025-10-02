TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

QMAKE_CXXFLAGS += -std=c++20

CONFIG += -O2

MAKEFILE = makefile.acquisition

QT           += widgets

HEADERS       = acquisition.hpp
SOURCES       = acquisition.cpp
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_acquisitionPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
