######################################################################
# project file for dmCtrlGUI
######################################################################

TEMPLATE = app
TARGET = dmCtrlGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/dmCtrl

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

MAKEFILE = makefile.dmCtrlGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/dmCtrl

HEADERS += ../../widgets/dmCtrl/dmCtrl.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../lib/multiIndiManager.hpp
           
SOURCES += dmCtrlGUI_main.cpp 
#\
#           ../../widgets/dmCtrl/dmCtrl.cpp \
           
FORMS += ../../widgets/dmCtrl/dmCtrl.ui \
         ../../widgets/xWidgets/fsmDisplay.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
