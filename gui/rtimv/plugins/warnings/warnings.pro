TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

MAKEFILE = makefile.warnings

QT           += widgets

HEADERS       = warnings.hpp
SOURCES       = warnings.cpp 
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_warningsPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
