TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

MAKEFILE = makefile.dmStatus

QT           += widgets

HEADERS       = dmStatus.hpp
SOURCES       = dmStatus.cpp 
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_dmStatusPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
