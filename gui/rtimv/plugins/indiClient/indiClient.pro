TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

MAKEFILE = makefile.indiClient

QT           += widgets

INCLUDEPATH  += ../../../../INDI/libcommon/
HEADERS       = indiClient.hpp
SOURCES       = indiClient.cpp

LIBS         += ../../../../INDI/libcommon/libcommon.a
LIBS         += ../../../../INDI/liblilxml/liblilxml.a

TARGET        = $$qtLibraryTarget(rtimv_indiClientPlugin)
DESTDIR       = ./

unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
