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
exists( $$(CONDA_PREFIX)/lib ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/lib
}

MAKEFILE = makefile.ttmGUI

# Input
INCLUDEPATH += ../../lib ../../lib ../../widgets 

HEADERS += ../../widgets/ttm/ttm.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../lib/multiIndiManager.hpp
           
SOURCES += ttmGUI_main.cpp 
#\
#           ../../widgets/ttm/ttm.cpp \
           
FORMS += ../../widgets/ttm/ttm.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/fsmDisplay.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
