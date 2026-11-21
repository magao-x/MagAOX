TARGET = loPredCtrlAccGUI
include(../magaoxQtApp.pri)

QT += widgets
INCLUDEPATH += ../../widgets/loPredCtrlAcc

HEADERS += ../../widgets/loPredCtrlAcc/loPredCtrlAccPanel.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp

SOURCES += loPredCtrlAccGUI_main.cpp \
           ../../widgets/loPredCtrlAcc/loPredCtrlAccPanel.cpp

FORMS += ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc
