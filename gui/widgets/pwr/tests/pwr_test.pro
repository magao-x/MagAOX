TEMPLATE = app
TARGET = pwr_test

QT += widgets
CONFIG += console c++20 qwt link_pkgconfig
CONFIG -= app_bundle
PKGCONFIG += mxlib

MOC_DIR = moc/
OBJECTS_DIR = obj/
MAKEFILE = makefile.pwr_test

SOURCES += pwr_test.cpp
HEADERS += ../pwrChannel.hpp ../pwrDevice.hpp

LIBS += ../../../../INDI/libcommon/libcommon.a \
        ../../../../INDI/liblilxml/liblilxml.a
