######################################################################
# project file for hwpSequencerGUI
######################################################################

TARGET = hwpSequencerGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets ../../widgets/xWidgets

HEADERS += ../../widgets/xWidgets/app.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/statusDisplay.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/selectionSw.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp \
           ../../widgets/xWidgets/statusCombo.hpp \
           ../../widgets/hwpSequencer/hwpSequencer.hpp \
           ../../lib/multiIndiManager.hpp 

SOURCES += hwpSequencerGUI_main.cpp

FORMS += ../../widgets/xWidgets/fsmDisplay.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/statusDisplay.ui \
         ../../widgets/xWidgets/statusCombo.ui \
         ../../widgets/xWidgets/toggleSlider.ui \
         ../../widgets/xWidgets/selectionSw.ui \
         ../../widgets/hwpSequencer/hwpSequencer.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a \
        -lmxlib

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
