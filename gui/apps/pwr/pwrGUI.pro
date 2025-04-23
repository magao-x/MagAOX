######################################################################
# project file for pwrGUI
######################################################################

TEMPLATE = app
TARGET = pwrGUI
DESTDIR = bin/
DEPENDPATH += ./ ../../lib

INCLUDEPATH += ../../lib ../../widgets ../../widgets/xWidgets

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/pwr

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}
CONFIG += c++17
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}
exists( $$(CONDA_PREFIX)/lib ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/lib
}

MAKEFILE = makefile.pwrGUI

# Input
HEADERS += ../../widgets/xWidgets/app.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/pwr/pwr.hpp \
           ../../widgets/pwr/pwrDevice.hpp \
           ../../widgets/pwr/pwrChannel.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += pwrGUI_main.cpp

FORMS += ../../widgets/pwr/pwr.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

LIBS += -lmxlib

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
