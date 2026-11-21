TEMPLATE = app
TARGET = timingPanel_test
QT += widgets
CONFIG += console c++20
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
MAKEFILE = makefile.timingPanel_test

INCLUDEPATH += ../../xWidgets
HEADERS += ../timingPanel.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/statusEntry.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusLabel.hpp \
           ../../xWidgets/toggleSlider.hpp

SOURCES += timingPanel_test.cpp ../timingPanel.cpp

FORMS += ../../xWidgets/statusEntry.ui ../../xWidgets/toggleSlider.ui

LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a \
        $$PWD/../../../../INDI/liblilxml/liblilxml.a
