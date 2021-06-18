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

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.loopCtrlGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/loopCtrl

HEADERS += ../../widgets/loopCtrl/loopCtrl.hpp \
           ../../lib/multiIndi.hpp \
           ../../lib/multiIndiManager.hpp
           
SOURCES += loopCtrlGUI_main.cpp 
#\
#           ../../widgets/loopCtrl/loopCtrl.cpp \
           
FORMS += ../../widgets/loopCtrl/loopCtrl.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
