TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

QMAKE_CXXFLAGS += -std=c++20

CONFIG += -O2

MAKEFILE = makefile.indiDictionary

QT           += widgets

INCLUDEPATH  += ../../../../INDI/libcommon/
HEADERS       = indiDictionary.hpp
SOURCES       = indiDictionary.cpp

LIBS         += ../../../../INDI/libcommon/libcommon.a
LIBS         += ../../../../INDI/liblilxml/liblilxml.a

TARGET        = $$qtLibraryTarget(rtimv_indiDictionaryPlugin)
DESTDIR       = ./

CONFIG += link_pkgconfig
PKGCONFIG += mxlib

unix:target.path=/usr/local/bin/plugins
INSTALLS+=target

# Optional workspace build of the command interface; system paths remain the default.
INCLUDEPATH += $$RTIMV_INCLUDEPATH
!isEmpty(RTIMV_LIBDIR) {
    LIBS += -L$$RTIMV_LIBDIR
    QMAKE_RPATHDIR += $$RTIMV_LIBDIR
}
