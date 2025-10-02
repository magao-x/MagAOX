######################################################################
# project file for ttmGUI
######################################################################

TARGET = ttmGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../lib ../../widgets

HEADERS += ../../widgets/ttm/ttm.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/fsmDisplay.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
	   ../../lib/multiIndiManager.hpp

SOURCES += ttmGUI_main.cpp
#\
#           ../../widgets/ttm/ttm.cpp \

FORMS += ../../widgets/ttm/ttm.ui \
         ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/fsmDisplay.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
