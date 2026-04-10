######################################################################
# project file for pwrGUI
######################################################################
TARGET = pwrGUI

include(../magaoxQtApp.pri)

# Input
HEADERS += ../../widgets/xWidgets/app.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/pwr/pwr.hpp \
           ../../widgets/pwr/pwrDevice.hpp \
           ../../widgets/pwr/pwrChannel.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += pwrGUI_main.cpp

FORMS += ../../widgets/pwr/pwr.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
