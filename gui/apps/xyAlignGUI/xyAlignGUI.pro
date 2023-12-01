######################################################################
# project file for xyAlignGUI
######################################################################

TEMPLATE = app
TARGET = xyAlignGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/xyAlign

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

MAKEFILE = makefile.xyAlignGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/xyAlign

HEADERS += ../../widgets/xyAlign/xyAlign.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../lib/multiIndiManager.hpp 
           
SOURCES += xyAlignGUI_main.cpp 
           
FORMS += ../../widgets/xyAlign/xyAlign.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/fsmDisplay.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc 

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
