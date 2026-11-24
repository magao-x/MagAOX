TARGET = po4aoGUI
include(../magaoxQtApp.pri)
INCLUDEPATH += ../../lib ../../widgets/po4ao
HEADERS += ../../widgets/po4ao/po4ao.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLineEdit.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../lib/multiIndiManager.hpp
SOURCES += po4aoGUI_main.cpp
FORMS += ../../widgets/po4ao/po4ao.ui
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a
RESOURCES += ../../resources/magaox.qrc \
             ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc
QT += widgets
