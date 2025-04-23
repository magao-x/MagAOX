######################################################################
# project file for singleModeGUI
######################################################################

TEMPLATE = app
TARGET = singleModeGUI
DESTDIR = bin/
DEPENDPATH += ./ ../../lib

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/singleMode

CONFIG+=debug
CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}
exists( $$(CONDA_PREFIX)/lib ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/lib
}

MAKEFILE = makefile.singleModeGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/singleMode ../../widgets/xWidgets

HEADERS += ../../widgets/singleMode/singleMode.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/gainCtrl.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += singleModeGUI_main.cpp
#\
#           ../../widgets/singleMode/singleMode.cpp \

FORMS += ../../widgets/singleMode/singleMode.ui \
         ../../widgets/xWidgets/gainCtrl.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
