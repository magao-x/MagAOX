TEMPLATE = app
TARGET = polarimetry_test
QT += widgets
CONFIG += console c++20
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
MAKEFILE = makefile.polarimetry_test
SOURCES += polarimetry_test.cpp
HEADERS += ../polarimetry.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/fsmDisplay.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusLabel.hpp \
           ../../xWidgets/statusEntry.hpp \
           ../../xWidgets/statusCombo.hpp \
           ../../xWidgets/toggleSlider.hpp
FORMS += ../polarimetry.ui \
         ../../xWidgets/fsmDisplay.ui \
         ../../xWidgets/statusEntry.ui \
         ../../xWidgets/statusCombo.ui \
         ../../xWidgets/toggleSlider.ui
INCLUDEPATH += ../../xWidgets
LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a $$PWD/../../../../INDI/liblilxml/liblilxml.a
