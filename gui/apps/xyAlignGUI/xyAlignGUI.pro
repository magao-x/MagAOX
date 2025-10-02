######################################################################
# project file for xyAlignGUI
######################################################################

TARGET = xyAlignGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/xyAlign

HEADERS += ../../widgets/xyAlign/xyAlign.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += xyAlignGUI_main.cpp

FORMS += ../../widgets/xyAlign/xyAlign.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/fsmDisplay.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
