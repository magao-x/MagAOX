######################################################################
# project file for offloadCtrlGUI
######################################################################

TARGET = offloadCtrlGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/offloadCtrl ../../widgets/xWidgets

HEADERS += ../../widgets/offloadCtrl/offloadCtrl.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/gainCtrl.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += offloadCtrlGUI_main.cpp


FORMS += ../../widgets/offloadCtrl/offloadCtrl.ui \
         ../../widgets/xWidgets/gainCtrl.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
