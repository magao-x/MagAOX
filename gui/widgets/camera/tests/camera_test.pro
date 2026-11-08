TEMPLATE = app
TARGET = camera_test
QT += widgets
CONFIG += console c++20 qwt link_pkgconfig
CONFIG -= app_bundle
PKGCONFIG += mxlib
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
MAKEFILE = makefile.camera_test
SOURCES += camera_test.cpp
INCLUDEPATH += ../.. ../../xWidgets ../../../lib
HEADERS += ../camera.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/fsmDisplay.hpp \
           ../../xWidgets/statusEntry.hpp \
           ../../xWidgets/statusDisplay.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusLabel.hpp \
           ../../xWidgets/selectionSw.hpp \
           ../../xWidgets/toggleSlider.hpp \
           ../roiStatus.hpp \
           ../../xWidgets/statusCombo.hpp \
           ../../xWidgets/stageStatus.hpp \
           ../../roi/roi.hpp \
           ../../stage/stage.hpp \
           ../shutterStatus.hpp

FORMS += ../camera.ui \
         ../../xWidgets/fsmDisplay.ui \
         ../../xWidgets/statusEntry.ui \
         ../../xWidgets/statusDisplay.ui \
         ../../xWidgets/statusCombo.ui \
         ../../xWidgets/toggleSlider.ui \
         ../../roi/roi.ui \
         ../../stage/stage.ui \
         ../shutterStatus.ui \
         ../../xWidgets/selectionSw.ui

LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a $$PWD/../../../../INDI/liblilxml/liblilxml.a
