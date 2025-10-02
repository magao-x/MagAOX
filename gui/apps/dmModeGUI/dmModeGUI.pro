######################################################################
# project file for dmModeGUI
######################################################################

TARGET = dmModeGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/dmMode

HEADERS += ../../widgets/dmMode/dmMode.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += dmModeGUI_main.cpp

FORMS += ../../widgets/dmMode/dmMode.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
