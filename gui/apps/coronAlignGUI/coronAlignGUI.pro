######################################################################
# project file for coronAlignGUI
######################################################################

TEMPLATE = app
TARGET = coronAlignGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/coronAlign

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

MAKEFILE = makefile.coronAlignGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/coronAlign

HEADERS += ../../widgets/coronAlign/coronAlign.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../lib/multiIndiManager.hpp 
           
SOURCES += coronAlignGUI_main.cpp 
           
FORMS += ../../widgets/coronAlign/coronAlign.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc 

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
