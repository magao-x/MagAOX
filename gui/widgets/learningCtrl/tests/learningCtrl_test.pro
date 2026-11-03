TEMPLATE = app
TARGET = learningCtrl_test
QT += widgets
CONFIG += console c++20
CONFIG -= app_bundle
MOC_DIR = moc/
OBJECTS_DIR = obj/
UI_DIR = ui/
RCC_DIR = res/
MAKEFILE = makefile.learningCtrl_test
SOURCES += learningCtrl_test.cpp
HEADERS += ../learningCtrl.hpp \
           ../../xWidgets/xWidget.hpp \
           ../../xWidgets/statusLineEdit.hpp \
           ../../xWidgets/statusLabel.hpp
FORMS += ../learningCtrl.ui
INCLUDEPATH += ../../xWidgets
LIBS += $$PWD/../../../../INDI/libcommon/libcommon.a \
        $$PWD/../../../../INDI/liblilxml/liblilxml.a
RESOURCES += ../../../resources/magaox.qrc \
             ../../../resources/MagAOXStyleSheets/MagAOXStyle.qrc
