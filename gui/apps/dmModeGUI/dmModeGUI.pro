
TEMPLATE = app
TARGET = dmModeGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/dmMode

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}
CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.dmModeGUI

# Input
HEADERS += ../../widgets/dmMode/dmModeGUI.hpp
           
SOURCES += ../../widgets/dmMode/dmModeGUI.cpp \
           dmModeGUI_main.cpp
           
FORMS += ../../widgets/dmMode/dmMode.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a 

QT += widgets
