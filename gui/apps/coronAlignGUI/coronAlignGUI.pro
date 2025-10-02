######################################################################
# project file for coronAlignGUI
######################################################################

TARGET = coronAlignGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/coronAlign ../../widgets/xWidgets

HEADERS += ../../widgets/coronAlign/coronAlign.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../lib/multiIndiManager.hpp \
           ../../widgets/xWidgets/selectionSwStatus.hpp \
           ../../widgets/xWidgets/statusDisplay.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/selectionSw.hpp \
           ../../widgets/xWidgets/stageStatus.hpp \
           ../../widgets/stage/stage.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusCombo.hpp

SOURCES += coronAlignGUI_main.cpp

FORMS += ../../widgets/coronAlign/coronAlign.ui \
         ../../widgets/xWidgets/statusDisplay.ui \
         ../../widgets/xWidgets/selectionSw.ui \
         ../../widgets/stage/stage.ui \
         ../../widgets/xWidgets/fsmDisplay.ui \
         ../../widgets/xWidgets/statusCombo.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
