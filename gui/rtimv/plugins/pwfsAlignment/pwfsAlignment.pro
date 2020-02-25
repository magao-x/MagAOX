TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

QT           += widgets

INCLUDEPATH  += /home/jrmales/Source/rtimv/src
HEADERS       = pwfsAlignment.hpp
SOURCES       = pwfsAlignment.cpp
TARGET        = $$qtLibraryTarget(rtimv_pwfsAlignmentPlugin)
DESTDIR       = ./
