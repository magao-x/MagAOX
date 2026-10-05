TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

QMAKE_CXXFLAGS += -std=c++20

CONFIG += -O2

MAKEFILE = makefile.pwfsAlignment

QT           += widgets

HEADERS       = pwfsAlignment.hpp
SOURCES       = pwfsAlignment.cpp
TARGET        = $$qtLibraryTarget(rtimv_pwfsAlignmentPlugin)
DESTDIR       = ./

CONFIG += link_pkgconfig
PKGCONFIG += mxlib

unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
