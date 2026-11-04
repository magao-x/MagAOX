TEMPLATE = app
TARGET = roi_commands_test
QT += widgets
CONFIG += console c++20 link_pkgconfig
CONFIG -= app_bundle
PKGCONFIG += mxlib
DEFINES += RTIMV_PLUGIN_TEST
MOC_DIR = moc/
OBJECTS_DIR = obj/
MAKEFILE = makefile.roi_commands_test
SOURCES += roi_commands_test.cpp ../cameraStatus/cameraStatus.cpp ../cameraStatus/cameraStatus_roi.cpp ../indiDictionary/indiDictionary.cpp
HEADERS += ../cameraStatus/cameraStatus.hpp ../indiDictionary/indiDictionary.hpp
INCLUDEPATH += ../../../../INDI/libcommon $$RTIMV_INCLUDEPATH
!isEmpty(RTIMV_LIBDIR) {
    LIBS += -L$$RTIMV_LIBDIR
    QMAKE_RPATHDIR += $$RTIMV_LIBDIR
}
LIBS += -lrtimv ../../../../INDI/libcommon/libcommon.a ../../../../INDI/liblilxml/liblilxml.a
