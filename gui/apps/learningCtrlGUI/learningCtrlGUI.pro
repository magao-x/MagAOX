TARGET = learningCtrlGUI
include(../magaoxQtApp.pri)
INCLUDEPATH += ../../lib ../../widgets/learningCtrl
HEADERS += ../../widgets/learningCtrl/learningCtrl.hpp \
           ../../widgets/xWidgets/xWidget.hpp \
           ../../widgets/xWidgets/statusLabel.hpp \
           ../../lib/multiIndiManager.hpp
SOURCES += learningCtrlGUI_main.cpp
FORMS += ../../widgets/learningCtrl/learningCtrl.ui
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a
RESOURCES += ../../resources/magaox.qrc \
             ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc
QT += widgets
