TARGET = timingGUI
include(../magaoxQtApp.pri)

QT += widgets
INCLUDEPATH += ../../widgets/timing

HEADERS += ../../widgets/timing/timingPanel.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusEntry.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../widgets/xWidgets/toggleSlider.hpp

SOURCES += timingGUI_main.cpp \
           ../../widgets/timing/timingPanel.cpp

FORMS += ../../widgets/xWidgets/statusEntry.ui \
         ../../widgets/xWidgets/toggleSlider.ui

LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc
