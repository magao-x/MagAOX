TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

MAKEFILE = makefile.pwfsAlignment

QT           += widgets

HEADERS       = pwfsAlignment.hpp
SOURCES       = pwfsAlignment.cpp
TARGET        = $$qtLibraryTarget(rtimv_pwfsAlignmentPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
