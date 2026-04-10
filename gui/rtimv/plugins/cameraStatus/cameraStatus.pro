TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

QMAKE_CXXFLAGS += -std=c++20

CONFIG += -O2

MAKEFILE = makefile.cameraStatus

QT           += widgets

HEADERS       = cameraStatus.hpp
SOURCES       = cameraStatus.cpp
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_cameraStatusPlugin)
DESTDIR       = ./

CONFIG += link_pkgconfig
PKGCONFIG += mxlib

unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
