TEMPLATE = app
TARGET = coronAlign_test
QT += widgets
CONFIG += console c++20 qwt
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
MAKEFILE = makefile.coronAlign_test
SOURCES += coronAlign_test.cpp
INCLUDEPATH += ../../xWidgets
HEADERS += ../coronAlign.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/statusDisplay.hpp \
           ../../xWidgets/statusLabel.hpp \
           ../../xWidgets/selectionSw.hpp \
           ../../xWidgets/stageStatus.hpp \
           ../../stage/stage.hpp \
           ../../xWidgets/fsmDisplay.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusCombo.hpp

FORMS += ../coronAlign.ui \
         ../../xWidgets/statusDisplay.ui \
         ../../xWidgets/selectionSw.ui \
         ../../stage/stage.ui \
         ../../xWidgets/fsmDisplay.ui \
         ../../xWidgets/statusCombo.ui

LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a $$PWD/../../../../INDI/liblilxml/liblilxml.a
