######################################################################
# project file for offloadCtrlGUI
######################################################################

TEMPLATE = app
TARGET = offloadCtrlGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/offloadCtrl

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.offloadCtrlGUI

# Input
HEADERS += ../../widgets/offloadCtrl/offloadCtrl.hpp \
           
SOURCES += offloadCtrlGUI_main.cpp \
           ../../widgets/offloadCtrl/offloadCtrl.cpp
        
           
FORMS += ../../widgets/offloadCtrl/offloadCtrl.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
