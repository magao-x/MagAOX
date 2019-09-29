SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/common.mk

apps_common = \
	sshDigger \
	sysMonitor \
	xindiserver \
	xt1121Ctrl \
	xt1121DCDU

apps_rtcicc = \
	mzmqServer \
	streamWriter \
	alpaoCtrl \
	dmMode

apps_rtc = \
	ocam2KCtrl \
	siglentSDG \
	ttmModulator \
	bmcCtrl \
	pi335Ctrl \
	pupilFit \
	shmimIntegrator

apps_icc = \
	filterWheelCtrl \
	hsfwCtrl \
	baslerCtrl \
	zaberCtrl \
	zaberLowLevel \
	picamCtrl \
	smc100ccCtrl \
	andorCtrl

apps_aoc = \
	trippLitePDU \
	mzmqClient

apps_vm = \
	magAOXMaths \
	timeSeriesSimulator

libs_to_build = libtelnet

apps_to_build = $(apps_common)

ifeq ($(MAGAOX_ROLE),AOC)
  apps_to_build += $(apps_aoc)
else ifeq ($(MAGAOX_ROLE),ICC)
  apps_to_build += $(apps_rtcicc)
  apps_to_build += $(apps_icc)
else ifeq ($(MAGAOX_ROLE),RTC)
  apps_to_build += $(apps_rtcicc)
  apps_to_build += $(apps_rtc)
else ifeq ($(MAGAOX_ROLE),vm)
  apps_to_build += $(apps_vm)
endif

all_guis = dmCtrlGUI \
	dmModeGUI \
	modwfsGUI \
	pupilGuideGUI \
	pwr \
	ttmpupilGUI

ifeq ($(MAGAOX_ROLE),RTC)
  guis_to_build =
else ifeq ($(MAGAOX_ROLE),ICC)
  guis_to_build =
else
  guis_to_build = $(all_guis)
endif

utils_to_build = logdump \
				 logstream \
                 cursesINDI \
				 xrif2shmim

scripts_to_install = magaox

all: indi_all libs_all apps_all utils_all

install: indi_install libs_install apps_install utils_install scripts_install

#We clean just libMagAOX, and the apps and utils for normal devel work.
clean: lib_clean apps_clean utils_clean

#Clean everything.
all_clean: indi_clean libs_clean lib_clean apps_clean utils_clean doc_clean

indi_all:
	cd INDI; ${MAKE} all

indi_install:
	cd INDI; ${MAKE} install

indi_clean:
	cd INDI; ${MAKE} clean

libs_all:
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib; ${MAKE} )|| exit 1; \
	done

libs_install:
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib; ${MAKE}  install) || exit 1; \
	done
	sudo bash -c "echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf"
	sudo ldconfig

libs_clean:
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib; ${MAKE}  clean) || exit 1; \
	done

lib_clean:
	cd libMagAOX; ${MAKE} clean

apps_all: libs_install

	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE} )|| exit 1; \
	done

apps_install:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE}  install) || exit 1; \
	done

apps_clean:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE}  clean) || exit 1; \
	done

guis_all: libs_install
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui; ${MAKE} )|| exit 1; \
	done

guis_install:
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui; ${MAKE} install) || exit 1; \
	done

guis_clean:
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui; ${MAKE} clean) || exit 1; \
	done

scripts_install:
	for script in ${scripts_to_install}; do \
		sudo install -d /opt/MagAOX/bin && \
		sudo install scripts/$$script /opt/MagAOX/bin  && \
		sudo ln -fs /opt/MagAOX/bin/$$script /usr/local/bin/$$script; \
	done

utils_all:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE}) || exit 1; \
		done

utils_install:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} install) || exit 1; \
		done

utils_clean:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} clean) || exit 1; \
		done


.PHONY: doc
doc:
	doxygen doc/config/Doxyfile.libMagAOX
	# cp doc/config/customtheme/index.html doc/config/customtheme/magao-x-logo-white.svg doc/output/

.PHONY: doc_clean
doc_clean:
	rm -rf doc/output

.PHONY: setup
setup:
	@for file in ./local/*.example.mk; do \
		dest=$$(echo $$file | sed 's/.example//'); \
		if [ ! -e $$dest ]; then cp -v $$file $$dest; fi \
	done
	@echo "*** Build settings available in local/common.mk ***"
	@grep "?=" Make/common.mk || true
	@echo "*** Build settings available in local/config.mk ***"
	@grep "?=" Make/config.mk || true
	@echo "***"
