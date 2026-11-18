TEMPLATE = app
TARGET = value_display_test
QT += widgets
CONFIG += console c++20 link_pkgconfig
CONFIG -= app_bundle
PKGCONFIG += mxlib
DEFINES += RTIMV_PLUGIN_TEST
MOC_DIR = moc/value_display_test/
OBJECTS_DIR = obj/value_display_test/
MAKEFILE = makefile.value_display_test
SOURCES += value_display_test.cpp ../valueDisplay/valueDisplay.cpp ../indiDictionary/indiDictionary.cpp
HEADERS += ../valueDisplay/valueDisplay.hpp ../indiDictionary/indiDictionary.hpp
INCLUDEPATH += $$PWD/../../../../INDI/libcommon $$RTIMV_INCLUDEPATH
!isEmpty(RTIMV_LIBDIR) {
    LIBS += -L$$RTIMV_LIBDIR
    QMAKE_RPATHDIR += $$RTIMV_LIBDIR
}
LIBS += -lrtimv $$PWD/../../../../INDI/libcommon/libcommon.a $$PWD/../../../../INDI/liblilxml/liblilxml.a
