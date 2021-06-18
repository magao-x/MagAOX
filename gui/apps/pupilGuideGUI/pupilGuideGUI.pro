######################################################################
# project file for pupilGuideGUI
######################################################################

TEMPLATE = app
TARGET = pupilGuideGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/pupilGuide

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.pupilGuideGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/pupilGuide

HEADERS += ../../widgets/pupilGuide/pupilGuide.hpp \
           ../../lib/multiIndiManager.hpp 
           
SOURCES += pupilGuideGUI_main.cpp 
           
FORMS += ../../widgets/pupilGuide/pupilGuide.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc 

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
