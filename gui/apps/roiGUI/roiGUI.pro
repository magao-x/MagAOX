######################################################################
# project file for roiGUI
######################################################################

TARGET = roiGUI
include(../magaoxQtApp.pri)

# Input
INCLUDEPATH += ../../lib ../../widgets/roi

HEADERS += ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/roi/roi.hpp \
           ../../lib/multiIndiSubscriber.hpp
           ../../lib/multiIndiPublisher.hpp
           ../../lib/multiIndiManager.hpp

SOURCES += roiGUI_main.cpp
#\
#           ../../widgets/roi/roi.cpp \

FORMS += ../../widgets/roi/roi.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/magaox.qrc
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc

QT += widgets
