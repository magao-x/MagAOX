######################################################################
# project file for ttmGUI
######################################################################

TEMPLATE = app
TARGET = ttmGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/ttm

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.ttmGUI

# Input
INCLUDEPATH += ../../lib ../../lib ../../widgets 

HEADERS += ../../widgets/ttm/ttm.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../lib/multiIndi.hpp \
           ../../lib/multiIndiManager.hpp
           
SOURCES += ttmGUI_main.cpp 
#\
#           ../../widgets/ttm/ttm.cpp \
           
FORMS += ../../widgets/ttm/ttm.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
