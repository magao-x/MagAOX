######################################################################
# project file for stageGUI
######################################################################

TEMPLATE = app
TARGET = stageGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/stage

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}

CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.stageGUI

# Input
INCLUDEPATH += ../../lib ../../widgets/stage

HEADERS += ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/stage/stage.hpp \
           ../../lib/multiIndiSubscriber.hpp
           ../../lib/multiIndiPublisher.hpp
           ../../lib/multiIndiManager.hpp
           
SOURCES += stageGUI_main.cpp 
#\
#           ../../widgets/stage/stage.cpp \
           
FORMS += ../../widgets/stage/stage.ui \
         ../../widgets/xWidgets/fsmDisplay.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a \
        -lqwt-qt5

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
