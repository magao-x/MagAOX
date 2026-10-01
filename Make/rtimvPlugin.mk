# Resolve a qmake binary. Rocky 9's qt5-devel ships only qmake-qt5, while
# Ubuntu's qtchooser provides plain qmake.

QMAKE ?= qmake
QMAKE_PATH := $(shell which $(QMAKE) 2>/dev/null)

ifeq "$(QMAKE_PATH)" ""
  QMAKE := qmake-qt5
  QMAKE_PATH := $(shell which $(QMAKE) 2>/dev/null)
endif

ifeq "$(QMAKE_PATH)" ""
  $(error No qmake found on PATH (are the Qt development libraries installed?))
endif

##############################

all: $(TARGET)

.PHONY: $(TARGET)
$(TARGET):
	$(QMAKE) -makefile $(TARGET).pro
	$(MAKE) -f makefile.$(TARGET)

install: $(TARGET)
	sudo $(MAKE) -f makefile.$(TARGET) install

clean:
	rm -f *.so *.o *~
	rm -f moc_*
	rm -f makefile.$(TARGET)
