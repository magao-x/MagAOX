TEMPLATE = app
TARGET = loPredCtrlAccPanel_test
QT += widgets
CONFIG += console c++20
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
MAKEFILE = makefile.loPredCtrlAccPanel_test

INCLUDEPATH += ../../xWidgets
HEADERS += ../loPredCtrlAccPanel.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/statusEntry.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusLabel.hpp \
           ../../xWidgets/toggleSlider.hpp

SOURCES += loPredCtrlAccPanel_test.cpp ../loPredCtrlAccPanel.cpp

FORMS += ../../xWidgets/statusEntry.ui ../../xWidgets/toggleSlider.ui

LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a \
        $$PWD/../../../../INDI/liblilxml/liblilxml.a
