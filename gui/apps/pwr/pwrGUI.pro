######################################################################
# project file for pwrGUI
######################################################################

TEMPLATE = app
TARGET = pwrGUI
DESTDIR = bin/ 
DEPENDPATH += ./ ../../lib 
INCLUDEPATH += ../../widgets/multiDial
INCLUDEPATH += ../../lib

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/pwr

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}
CONFIG += c++14
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}

MAKEFILE = makefile.pwrGUI

# Input
HEADERS += pwrGUI.hpp \
           ../../widgets/multiDial/xqwt_multi_dial.h \
           ../../widgets/pwr/pwrDevice.hpp \
           ../../widgets/pwr/pwrChannel.hpp \
           ../../lib/multiIndiManager.hpp 
           
SOURCES += pwrGUI.cpp \
           pwr_main.cpp \
           ../../widgets/multiDial/xqwt_multi_dial.cpp \
           ../../widgets/pwr/pwrDevice.cpp \
           ../../widgets/pwr/pwrChannel.cpp
           
FORMS += ../../widgets/pwr/pwr.ui
     
LIBS += ../../../INDI/libcommon/libcommon.a \
        ../../../INDI/liblilxml/liblilxml.a

LIBS += -lmxlib 

RESOURCES += ../../resources/magaox.qrc         
RESOURCES += ../../resources/MagAOXStyleSheets/MagAOXStyle.qrc 

QT += widgets
