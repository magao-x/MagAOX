######################################################################
# project file for cameraGUI
######################################################################

TEMPLATE = app
TARGET = cameraGUI
DESTDIR = bin/
DEPENDPATH += ./ ../../lib

CONFIG+=debug

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/camera

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

MAKEFILE = makefile.cameraGUI

# Input
INCLUDEPATH += ../../lib ../../widgets ../../widgets/xWidgets

HEADERS += ../../widgets/camera/camera.hpp \
           ../../widgets/xWidgets/app.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/selectionSwStatus.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/statusDisplay.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/selectionSw.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp \
           ../../widgets/camera/roiStatus.hpp \
           ../../widgets/xWidgets/statusCombo.hpp \
           ../../widgets/xWidgets/stageStatus.hpp \
           ../../widgets/roi/roi.hpp \
           ../../widgets/stage/stage.hpp \
           ../../widgets/camera/shutterStatus.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += cameraGUI_main.cpp

FORMS += ../../widgets/camera/camera.ui \
         ../../widgets/xWidgets/fsmDisplay.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/statusDisplay.ui \
         ../../widgets/xWidgets/statusCombo.ui \
         ../../widgets/xWidgets/toggleSlider.ui \
         ../../widgets/roi/roi.ui \
         ../../widgets/stage/stage.ui \
         ../../widgets/camera/shutterStatus.ui \
         ../../widgets/xWidgets/selectionSw.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a \
        -lmxlib

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
