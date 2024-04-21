######################################################################
# project file for roiGUI
######################################################################

TEMPLATE = app
TARGET = roiGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/roi

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

MAKEFILE = makefile.roiGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/roi

HEADERS += ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/roi/roi.hpp \
           ../../lib/multiIndiSubscriber.hpp
           ../../lib/multiIndiPublisher.hpp
           ../../lib/multiIndiManager.hpp
           
SOURCES += roiGUI_main.cpp 
#\
#           ../../widgets/roi/roi.cpp \
           
FORMS += ../../widgets/roi/roi.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
