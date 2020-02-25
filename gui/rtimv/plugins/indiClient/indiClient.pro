TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

QT           += widgets

INCLUDEPATH  += /home/jrmales/Source/rtimv/src
INCLUDEPATH  += ../../../../INDI/libcommon/
HEADERS       = indiClient.hpp
SOURCES       = indiClient.cpp

LIBS         += ../../../../INDI/libcommon/libcommon.a
LIBS         += ../../../../INDI/liblilxml/liblilxml.a

TARGET        = $$qtLibraryTarget(rtimv_indiClientPlugin)
DESTDIR       = ./
