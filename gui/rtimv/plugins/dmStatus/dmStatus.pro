TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

QMAKE_CXXFLAGS += -std=c++20

CONFIG += -O2

MAKEFILE = makefile.dmStatus

QT           += widgets

HEADERS       = dmStatus.hpp
SOURCES       = dmStatus.cpp
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_dmStatusPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
