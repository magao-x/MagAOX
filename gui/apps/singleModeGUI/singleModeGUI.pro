######################################################################
# project file for singleModeGUI
######################################################################

TARGET = singleModeGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/singleMode ../../widgets/xWidgets

HEADERS += ../../widgets/singleMode/singleMode.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/gainCtrl.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += singleModeGUI_main.cpp
#\
#           ../../widgets/singleMode/singleMode.cpp \

FORMS += ../../widgets/singleMode/singleMode.ui \
         ../../widgets/xWidgets/gainCtrl.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
