TEMPLATE = app
DESTDIR = bin/
DEPENDPATH += ./ ../../lib

INCLUDEPATH += ../../lib ../../widgets ../../widgets/xWidgets

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
# Keep generated ui_*.h files local to each GUI build so one GUI does not
# dirty another GUI's dependencies during a later build or install pass.
UI_DIR = ui/

CONFIG(release, debug|release) {
    CONFIG += optimize_full
}
#CONFIG += c++14
QMAKE_CXXFLAGS += -std=c++20
CONFIG += qwt
exists( $$(CONDA_PREFIX)/include ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/include
}
exists( $$(CONDA_PREFIX)/lib ) {
    INCLUDEPATH += $$(CONDA_PREFIX)/lib
}

CONFIG += link_pkgconfig
PKGCONFIG += mxlib

MAKEFILE = makefile.$${TARGET}
