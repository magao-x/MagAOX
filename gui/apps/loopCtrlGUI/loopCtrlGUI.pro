######################################################################
# project file for loopCtrlGUI
######################################################################

TARGET = loopCtrlGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/loopCtrl ../../widgets/xWidgets

HEADERS += ../../widgets/loopCtrl/loopCtrl.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/gainCtrl.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += loopCtrlGUI_main.cpp
#\
#           ../../widgets/loopCtrl/loopCtrl.cpp \

FORMS += ../../widgets/loopCtrl/loopCtrl.ui \
         ../../widgets/xWidgets/gainCtrl.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
