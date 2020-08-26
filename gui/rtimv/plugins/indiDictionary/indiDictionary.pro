TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

MAKEFILE = makefile.indiDictionary

QT           += widgets

INCLUDEPATH  += ../../../../INDI/libcommon/
HEADERS       = indiDictionary.hpp
SOURCES       = indiDictionary.cpp

LIBS         += ../../../../INDI/libcommon/libcommon.a
LIBS         += ../../../../INDI/liblilxml/liblilxml.a

TARGET        = $$qtLibraryTarget(rtimv_indiDictionaryPlugin)
DESTDIR       = ./

unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
