SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/common.mk

apps_common = \
	sshDigger \
	sysMonitor \
	xindiserver
apps_rtcicc = \
	mzmqServer \
	streamWriter \
	alpaoCtrl
apps_rtc = \
	ocam2KCtrl \
	siglentSDG \
	ttmModulator \
	bmcCtrl \
	pi335Ctrl
apps_icc = \
	filterWheelCtrl \
	baslerCtrl \
	zaberCtrl \
	picamCtrl \
	smc100ccCtrl \
	zaberLowLevel
apps_aoc = \
	magAOXMaths \
	trippLitePDU \
	xt1121Ctrl \
    xt1121DCDU \
	mzmqClient

libs_to_build = libtelnet

apps_to_build = $(apps_common)

ifeq ($(MAGAOX_ROLE),aoc)
  apps_to_build += $(apps_aoc)
else ifeq ($(MAGAOX_ROLE),icc)
  apps_to_build += $(apps_rtcicc)
  apps_to_build += $(apps_icc)
else ifeq ($(MAGAOX_ROLE),rtc)
  apps_to_build += $(apps_rtcicc)
  apps_to_build += $(apps_rtc)
endif

ifneq ($(PYLON),false)
apps_to_build += baslerCtrl
endif

ifneq ($(PICAM),false)
apps_to_build += picamCtrl
endif

ifneq ($(EDT),false)
apps_to_build += ocam2KCtrl
apps_to_build += andorCtrl
endif

utils_to_build = logdump \
                 cursesINDI

scripts_to_install = magaox_procstart.sh \
                     magaox_startup.sh \
                     magaox_status.sh \
                     magaox_shutdown.sh

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
		(cd libs/$$lib; ${MAKE} )|| break; \
	done

libs_install:
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib; ${MAKE}  install) || break; \
	done
	sudo bash -c "echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf"
	sudo ldconfig

libs_clean:
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib; ${MAKE}  clean) || break; \
	done

lib_clean:
	cd libMagAOX; ${MAKE} clean

apps_all: libs_install

	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE} )|| break; \
	done

apps_install:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE}  install) || break; \
	done

apps_clean:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE}  clean) || break; \
	done

scripts_install:
	for script in ${scripts_to_install}; do \
		sudo install -d /opt/MagAOX/bin && \
		sudo install scripts/$$script /opt/MagAOX/bin  && \
		sudo ln -fs /opt/MagAOX/bin/$$script /usr/local/bin/$$script; \
	done

utils_all:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE}) || break; \
		done

utils_install:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} install) || break; \
		done

utils_clean:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} clean) || break; \
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
