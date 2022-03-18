####################################################
# Makefile for building MagAOX Qt GUIs
#
# The only thing that needs to be customized here is TARGET. The 
# remaing app-specific details are handled by the Qt .pro file.
# 
# Usage: In the folder of a GUI app, create a Makefile with a minimum of:
#       TARGET=<name>
#       include ../../../Make/magAOXGUI.mk
#
####################################################



#############################
# The qt5 qmake 
#
# On a system where qt5 qmake is the one in the bath, then no
# argument is needed.  If not, then invoke with, e.g., `make QMAKE=qmake-qt5`

QMAKE?=qmake

QMAKE_PATH := $(shell which qmake 2>/dev/null)

$(info $(QMAKE_PATH))

ifeq "$(QMAKE_PATH)" ""
  QMAKE=qmake-qt5
endif


##############################


all: $(TARGET)

.PHONY: $(TARGET)
$(TARGET):
	$(QMAKE) -makefile $(TARGET).pro
	$(MAKE) -f makefile.$(TARGET)

install: $(TARGET)
	sudo install bin/$(TARGET) /usr/local/bin

clean:
ifneq (,$(wildcard ./makefile.$(TARGET)))  #Test if the generated makefile exists to avoid errors on 2nd make clean
	$(MAKE) -f makefile.$(TARGET) distclean
endif
	rm -f *~
	rm -f bin/$(TARGET)
	rm -rf bin moc obj res
