SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/common.mk

apps_common = \
	sshDigger \
	sysMonitor \
	xindiserver \
	mzmqClient \
	magAOXMaths \
	mzmqServer \
	streamWriter \
	dmMode \
	shmimIntegrator \
	timeSeriesSimulator

apps_rtcicc = alpaoCtrl \
              cacaoInterface \
				  userGainCtrl \
				  zaberCtrl \
	           zaberLowLevel

apps_rtc = \
	ocam2KCtrl \
        andorCtrl \
	siglentSDG \
	ttmModulator \
	bmcCtrl \
   rhusbMon \
	pi335Ctrl \
	pupilFit \
	t2wOffloader \
	dmSpeckle \
   w2tcsOffloader \
	pwfsSlopeCalc

apps_icc = \
   acronameUsbHub \
	flipperCtrl \
	filterWheelCtrl \
	hsfwCtrl \
	baslerCtrl \
	picamCtrl \
	smc100ccCtrl \
	andorCtrl \
	usbtempMon \
	xt1121Ctrl \
	xt1121DCDU \
   picoMotorCtrl \
	koolanceCtrl \
	tcsInterface

apps_aoc = \
	trippLitePDU \
	tcsInterface \
	adcTracker \
	kTracker \
	koolanceCtrl \
	observerCtrl \
	siglentSDG

# apps_vm = none yet
apps_tic = \
	acronameUsbHub \
	baslerCtrl \
	bmcCtrl \
	trippLitePDU 


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
else ifeq ($(MAGAOX_ROLE),TIC)
  apps_to_build += $(apps_tic)
# else ifeq ($(MAGAOX_ROLE),vm)
#   apps_to_build += $(apps_vm)
endif

all_guis = \
   dmCtrlGUI \
	dmModeGUI \
	offloadCtrlGUI \
	pupilGuideGUI \
	pwr \
	coronAlignGUI \
   loopCtrlGUI \
	roiGUI \
	cameraGUI \
	stageGUI

ifeq ($(MAGAOX_ROLE),RTC)
  guis_to_build =
else ifeq ($(MAGAOX_ROLE),ICC)
  guis_to_build =
else ifeq ($(MAGAOX_ROLE),TIC)
  guis_to_build =
else
  guis_to_build = $(all_guis)
endif

all_rtimv_plugins = \
	cameraStatus \
	indiDictionary \
	pwfsAlignment \
	dmStatus

ifeq ($(MAGAOX_ROLE),RTC)
  rtimv_plugins_to_build =
else ifeq ($(MAGAOX_ROLE),ICC)
  rtimv_plugins_to_build =
else ifeq ($(MAGAOX_ROLE),TIC)
  rtimv_plugins_to_build =
else
  rtimv_plugins_to_build = $(all_rtimv_plugins)
endif

utils_to_build = logdump \
				     logstream \
                 cursesINDI \
				     xrif2shmim \
				     xrif2fits

scripts_to_install = magaox \
	query_seeing \
	sync_cacao \
	xctrl \
	netconsole_logger \
	creaimshm \
	dmdispbridge \
	shmimTCPreceive \
	shmimTCPtransmit \
	lookyloo \
	obs_to_movie \
	instrument_backup_sync \
	cacao_startup_if_present \
	git_check_all \
	collect_camera_configs_for_darks \
	shot_in_the_dark \
        howfs_apply \
	lowfs_switch \
	lowfs_apply \
	lowfs_switch_apply

all: indi_all libs_all flatlogs apps_all guis_all utils_all

install: indi_install libs_install flatlogs_all apps_install guis_install utils_install scripts_install rtscripts_install

#We clean just libMagAOX, and the apps, guis, and utils for normal devel work.
clean: lib_clean apps_clean guis_clean utils_clean

#Clean everything.
all_clean: indi_clean libs_clean flatlogs_clean lib_clean apps_clean guis_clean utils_clean doc_clean

flatlogs_all:
	cd flatlogs/src/; ${MAKE} install

flatlogs_clean:
	cd flatlogs/src/; ${MAKE} clean
	rm -rf flatlogs/bin

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

apps_all: libs_install flatlogs_all

	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE} )|| exit 1; \
	done

apps_install: flatlogs_all
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE}  install) || exit 1; \
	done

apps_clean:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE}  clean) || exit 1; \
	done

guis_all: libs_install rtimv_plugins_all
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui; ${MAKE} )|| exit 1; \
	done

guis_install: rtimv_plugins_install
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui; ${MAKE} install) || exit 1; \
	done

guis_clean: rtimv_plugins_clean 
	for gui in ${all_guis}; do \
		(cd gui/apps/$$gui; ${MAKE} clean) || exit 1; \
	done

rtimv_plugins_all: libs_install
	for plg in ${rtimv_plugins_to_build}; do \
		(cd gui/rtimv/plugins/$$plg; ${MAKE} )|| exit 1; \
	done

rtimv_plugins_install:
	for plg in ${rtimv_plugins_to_build}; do \
		(cd gui/rtimv/plugins/$$plg; ${MAKE} install) || exit 1; \
	done

rtimv_plugins_clean:
	for plg in ${rtimv_plugins_to_build}; do \
		(cd gui/rtimv/plugins/$$plg; ${MAKE} clean) || exit 1; \
	done

scripts_install:
	for script in ${scripts_to_install}; do \
		sudo install -d /opt/MagAOX/bin && \
		sudo install scripts/$$script /opt/MagAOX/bin  && \
		sudo ln -fs /opt/MagAOX/bin/$$script /usr/local/bin/$$script; \
	done


rtscripts_install:
	for scriptname in make_cpusets procs_to_cpusets; do \
		sudo install -d /opt/MagAOX/bin && \
		if [ -e rtSetup/$(MAGAOX_ROLE)/$$scriptname ]; then \
			sudo install rtSetup/$(MAGAOX_ROLE)/$$scriptname /opt/MagAOX/bin/$$scriptname && \
			sudo ln -fs /opt/MagAOX/bin/$$scriptname /usr/local/bin/$$scriptname; \
		else \
			echo "echo 'No $$scriptname for $$MAGAOX_ROLE'\nexit 0" | sudo tee /opt/MagAOX/bin/$$scriptname && \
			sudo chmod +x /opt/MagAOX/bin/$$scriptname && \
			sudo ln -fs /opt/MagAOX/bin/$$scriptname /usr/local/bin/$$scriptname; \
		fi \
	; done

utils_all: flatlogs_all
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE}) || exit 1; \
		done

utils_install: flatlogs_all
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
