TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

MAKEFILE = makefile.acquisition

QT           += widgets

HEADERS       = acquisition.hpp
SOURCES       = acquisition.cpp
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_acquisitionPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
