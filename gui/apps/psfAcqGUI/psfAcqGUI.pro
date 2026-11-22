######################################################################
# project file for psfAcqGUI
######################################################################

TARGET = psfAcqGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets ../../widgets/xWidgets

HEADERS += ../../widgets/psfAcq/psfAcq.hpp \
           ../../widgets/xWidgets/app.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../lib/multiIndiManager.hpp

SOURCES += psfAcqGUI_main.cpp

FORMS += ../../widgets/psfAcq/psfAcq.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
