TEMPLATE = app
TARGET = po4ao_test
QT += widgets
CONFIG += console c++20
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
RCC_DIR = res/
MAKEFILE = makefile.po4ao_test
SOURCES += po4ao_test.cpp
HEADERS += ../po4ao.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusLabel.hpp
FORMS += ../po4ao.ui
INCLUDEPATH += ../../xWidgets
LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a \
        $$PWD/../../../../INDI/liblilxml/liblilxml.a
RESOURCES += ../../../resources/magaox.qrc \
             ../../../resources/MagAOXStyleSheets/MagAOXStyle.qrc
