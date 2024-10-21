######################################################################
# project file for loopCtrlGUI
######################################################################

TEMPLATE = app
TARGET = loopCtrlGUI
DESTDIR = bin/
DEPENDPATH += ./ ../../lib

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/loopCtrl

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

MAKEFILE = makefile.loopCtrlGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/loopCtrl ../../widgets/xWidgets

HEADERS += ../../widgets/loopCtrl/loopCtrl.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/gainCtrl.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += loopCtrlGUI_main.cpp
#\
#           ../../widgets/loopCtrl/loopCtrl.cpp \

FORMS += ../../widgets/loopCtrl/loopCtrl.ui \
         ../../widgets/xWidgets/gainCtrl.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
