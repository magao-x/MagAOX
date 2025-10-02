######################################################################
# project file for pupilGuideGUI
######################################################################

TARGET = pupilGuideGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/pupilGuide ../../widgets/xWidgets

HEADERS += ../../widgets/pupilGuide/pupilGuide.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += pupilGuideGUI_main.cpp

FORMS += ../../widgets/pupilGuide/pupilGuide.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/fsmDisplay.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
