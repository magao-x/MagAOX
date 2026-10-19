####################################################
# Makefile for building MagAOX Qt GUIs
#
# The only thing that needs to be customized here is TARGET. The
# remaining app-specific details are handled by the Qt .pro file.
#
# Usage: In the folder of a GUI app, create a Makefile with a minimum of:
#       TARGET=<name>
#       include ../../../Make/magAOXGUI.mk
#
####################################################



# Resolve a qmake binary. Rocky 9's qt5-devel ships only qmake-qt5, while
# Ubuntu's qtchooser provides plain qmake.

QMAKE ?= qmake
QMAKE_PATH := $(shell which $(QMAKE) 2>/dev/null)

ifeq "$(QMAKE_PATH)" ""
  QMAKE := qmake-qt5
  QMAKE_PATH := $(shell which $(QMAKE) 2>/dev/null)
endif

# Cleaning must also work on systems without Qt development tools.
ifneq ($(MAKECMDGOALS),clean)
ifeq "$(QMAKE_PATH)" ""
  $(error No qmake found on PATH (are the Qt development libraries installed?))
endif
endif

QMAKE_PROJECT := $(TARGET).pro
QMAKE_MAKEFILE := makefile.$(TARGET)

# Most GUIs share the common qmake settings in gui/apps/magaoxQtApp.pri.
# Generate the qmake makefile only when those inputs change instead of
# regenerating it on every install pass.
QMAKE_DEPS := $(QMAKE_PROJECT) ../magaoxQtApp.pri


##############################


all: $(TARGET)

.PHONY: $(TARGET)
$(TARGET): $(QMAKE_MAKEFILE)
	$(MAKE) -f $(QMAKE_MAKEFILE)

$(QMAKE_MAKEFILE): $(QMAKE_DEPS)
	$(QMAKE) -makefile $(QMAKE_PROJECT)

install: $(TARGET)
	sudo install bin/$(TARGET) /usr/local/bin

# A stale generated makefile can rerun qmake during distclean. Only recurse
# when qmake and the generated makefile are both present.
clean:
ifneq (,$(QMAKE_PATH))
ifneq (,$(wildcard ./$(QMAKE_MAKEFILE)))
	$(MAKE) -f $(QMAKE_MAKEFILE) distclean
endif
endif
	rm -f *~ $(QMAKE_MAKEFILE) .qmake.stash
	rm -f bin/$(TARGET)
	rm -rf bin moc obj res ui
