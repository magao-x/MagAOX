TEMPLATE = lib
CONFIG += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

QMAKE_CXXFLAGS += -std=c++20
CONFIG += -O2
MAKEFILE = makefile.valueDisplay
QT += widgets

HEADERS = valueDisplay.hpp
SOURCES = valueDisplay.cpp
LIBS += -lrtimv
TARGET = $$qtLibraryTarget(rtimv_valueDisplayPlugin)
DESTDIR = ./

CONFIG += link_pkgconfig
PKGCONFIG += mxlib

unix:target.path=/usr/local/bin/plugins
INSTALLS += target

# Optional matching workspace headers/library; system paths remain the default.
INCLUDEPATH += $$RTIMV_INCLUDEPATH
!isEmpty(RTIMV_LIBDIR) {
    LIBS += -L$$RTIMV_LIBDIR
    QMAKE_RPATHDIR += $$RTIMV_LIBDIR
}
