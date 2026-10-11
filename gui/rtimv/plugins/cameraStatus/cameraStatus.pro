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
SOURCES       = cameraStatus.cpp cameraStatus_roi.cpp
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_cameraStatusPlugin)
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
