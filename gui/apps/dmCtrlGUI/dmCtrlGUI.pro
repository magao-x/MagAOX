######################################################################
# project file for dmCtrlGUI
######################################################################

TARGET = dmCtrlGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/dmCtrl

HEADERS += ../../widgets/dmCtrl/dmCtrl.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += dmCtrlGUI_main.cpp
#\
#           ../../widgets/dmCtrl/dmCtrl.cpp \

FORMS += ../../widgets/dmCtrl/dmCtrl.ui \
         ../../widgets/xWidgets/fsmDisplay.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
