TEMPLATE = app
TARGET = pupilGuide_test
QT += widgets
CONFIG += console c++20
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
MAKEFILE = makefile.pupilGuide_test
SOURCES += pupilGuide_test.cpp
HEADERS += ../pupilGuide.hpp \
           ../../xWidgets/stagePairStatus.hpp \
           ../../xWidgets/statusCombo.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusEntry.hpp \
           ../../xWidgets/statusLabel.hpp \
           ../../xWidgets/fsmDisplay.hpp \
           ../../xWidgets/toggleSlider.hpp
FORMS += ../pupilGuide.ui \
         ../../xWidgets/statusCombo.ui \
         ../../xWidgets/statusEntry.ui \
         ../../xWidgets/fsmDisplay.ui \
         ../../xWidgets/toggleSlider.ui
INCLUDEPATH += ../../xWidgets
LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a $$PWD/../../../../INDI/liblilxml/liblilxml.a
