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
INCLUDEPATH += ../../lib ../../widgets/coronAlign ../../widgets/xWidgets

HEADERS += ../../widgets/coronAlign/coronAlign.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../lib/multiIndiManager.hpp \
           ../../widgets/xWidgets/selectionSwStatus.hpp \
           ../../widgets/xWidgets/statusDisplay.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/selectionSw.hpp \
           ../../widgets/xWidgets/stageStatus.hpp \
           ../../widgets/stage/stage.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusCombo.hpp

SOURCES += coronAlignGUI_main.cpp

FORMS += ../../widgets/coronAlign/coronAlign.ui \
         ../../widgets/xWidgets/statusDisplay.ui \
         ../../widgets/xWidgets/selectionSw.ui \
         ../../widgets/stage/stage.ui \
         ../../widgets/xWidgets/fsmDisplay.ui \
         ../../widgets/xWidgets/statusCombo.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
