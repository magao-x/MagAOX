TEMPLATE      = lib
CONFIG       += plugin

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14

CONFIG += -O3

QT           += widgets

HEADERS       = cameraStatus.hpp
SOURCES       = cameraStatus.cpp 
LIBS          += -lrtimv
TARGET        = $$qtLibraryTarget(rtimv_cameraStatusPlugin)
DESTDIR       = ./


unix:target.path=/usr/local/bin/plugins
INSTALLS+=target
