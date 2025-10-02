TEMPLATE = app
DESTDIR = bin/
DEPENDPATH += ./ ../../lib

INCLUDEPATH += ../../lib ../../widgets ../../widgets/xWidgets

MOC_DIR = moc/
OBJECTS_DIR = obj/
RCC_DIR = res/
UI_DIR = ../../widgets/pwr

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

MAKEFILE = makefile.$${TARGET}
