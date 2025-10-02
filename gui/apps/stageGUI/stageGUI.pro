######################################################################
# project file for stageGUI
######################################################################

TARGET = stageGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/stage

HEADERS += ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/stage/stage.hpp \
           ../../lib/multiIndiSubscriber.hpp
           ../../lib/multiIndiPublisher.hpp
           ../../lib/multiIndiManager.hpp

SOURCES += stageGUI_main.cpp
#\
#           ../../widgets/stage/stage.cpp \

FORMS += ../../widgets/stage/stage.ui \
         ../../widgets/xWidgets/fsmDisplay.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
