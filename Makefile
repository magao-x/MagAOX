SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/local/common.mk
-include $(SELF_DIR)/Make/python.mk


#Apps to build for any basic system
apps_basic = \
	sshDigger \
	xindiserver \
	magAOXMaths \
	timeSeriesSimulator \
	mzmqClient
pythonapps_basic = \
	dbIngest

# Apps commmon to all MagAO-X control machines
apps_common = \
    sysMonitor \
	mzmqServer \
	streamWriter \
	dmMode \
	shmimIntegrator \
	closedLoopIndi

apps_aoc = \
	trippLitePDU \
	xt1121Ctrl \
	xt1121DCDU \
	tcsInterface \
	adcTracker \
	hwpTracker \
	kTracker \
	koolanceCtrl \
	observerCtrl \
	stateRuleEngine \
	xInstGraph

pythonapps_aoc = \
	audibleAlerts

# Apps common to RTC and ICC on MagAO-X
apps_rtcicc = \
    alignLoop \
    acronameUsbHub \
	baslerCtrl \
    bmcCtrl \
	flipperCtrl \
	flowRPM \
    hsfwCtrl \
    rhusbMon \
	cacaoInterface \
    modalGainOpt \
    modalPSDs \
	userGainCtrl \
    refRMS \
    streamCircBuff \
	zaberCtrl \
	zaberLowLevel \
	picoMotorCtrl \
	psfFit \
	dbIngest

# Apps needed on RTC
apps_rtc = \
	alpaoCtrl \
	ocam2KCtrl \
	andorCtrl \
	cred2Ctrl \
	siglentSDG \
	ttmModulator \
	pi335Ctrl \
	pupilFit \
	t2wOffloader \
	dmSpeckle \
	w2tcsOffloader \
	pwfsSlopeCalc \
    kcubeCtrl \
	dmPokeXCorr \
	psfAcq \
	strehlEstimator \
	modalFilter \
	dmRecon
pythonapps_rtc = \
	efcControl

# Apps needed on ICC
apps_icc = \
	dmPokeCenter \
	filterWheelCtrl \
	smc100ccCtrl \
	usbtempMon \
	flowRPM \
	koolanceCtrl \
	corAlign \
	adcCtrl \
	picamCtrl \
	pvcamCtrl \
	zaberLowLevelBinary

pythonapps_icc = \
	adcCtrl \
	visxCtrl

# Apps only needed on accelerometers ACC*
apps_acc = \
	mcp3008Ctrl

# Apps needed on TIC
apps_tic = \
	acronameUsbHub \
	baslerCtrl \
	bmcCtrl \
	trippLitePDU \
	rhusbMon \
	dmSpeckle


# Apps with simulator mode
apps_sim = \
	trippLitePDU

all_buildable_apps = \
	adcTracker \
	alignLoop \
	cacaoInterface \
	closedLoopIndi \
	dmMode \
	dmPokeCenter \
	dmPokeXCorr \
	dmSpeckle \
	filterWheelCtrl \
	flipperCtrl \
	flowRPM \
	hwpTracker \
	indiTSAccumulator \
	koolanceCtrl \
	kTracker \
	loPredCtrl \
	magAOXMaths \
	modalFilter \
	modalGainOpt \
	modalPSDs \
	mzmqClient \
	mzmqServer \
	observerCtrl \
	pi335Ctrl \
	picoMotorCtrl \
	psfAcq \
	psfFit \
	pupilFit \
	pwfsSlopeCalc \
	refRMS \
	rhusbMon \
	shmimIntegrator \
	siglentSDG \
	smc100ccCtrl \
	sparkleClock \
	sshDigger \
	stateRuleEngine \
	streamCircBuff \
	streamWriter \
	strehlEstimator \
	sysMonitor \
	t2wOffloader \
	tcsInterface \
	timeSeriesSimulator \
	trippLitePDU \
	ttmModulator \
	usbtempMon \
	w2tcsOffloader \
	xindiserver \
	xt1121Ctrl \
	xt1121DCDU \
	zaberCtrl \
	zaberLowLevel

# EDT-backed camera controllers remain in the generic ALL_APPS coverage build
# only when the SDK headers are present locally. Otherwise they are covered
# through their unit-test harnesses instead of direct app-binary builds.
ifneq ($(wildcard /opt/EDTpdv/edtinc.h),)
all_buildable_apps += \
	cred2Ctrl
endif

libs_to_build = libtelnet

apps_to_build = $(apps_basic)
pythonapps_to_install = $(pythonapps_basic)

has_cacao = 0
ifeq ($(MAGAOX_ROLE),AOC)
  apps_to_build += $(apps_common)
  apps_to_build += $(apps_aoc)
  pythonapps_to_install += $(pythonapps_aoc)
else ifeq ($(MAGAOX_ROLE),ICC)
  apps_to_build += $(apps_common)
  apps_to_build += $(apps_rtcicc)
  apps_to_build += $(apps_icc)
  pythonapps_to_install += $(pythonapps_icc)
  has_cacao = 1
else ifeq ($(MAGAOX_ROLE),RTC)
  apps_to_build += $(apps_common)
  apps_to_build += $(apps_rtcicc)
  apps_to_build += $(apps_rtc)
  pythonapps_to_install += $(pythonapps_rtc)
  has_cacao = 1
else ifeq ($(findstring ACC,$(MAGAOX_ROLE)),ACC)
  apps_to_build += $(apps_common)
  apps_to_build += $(apps_acc)
else ifeq ($(MAGAOX_ROLE),TIC)
  apps_to_build += $(apps_common)
  apps_to_build += $(apps_tic)
  has_cacao = 1
else ifeq ($(MAGAOX_ROLE),SS)
  apps_to_build += $(apps_sim)
endif

# If building for coverage, build everything that you can.
ifeq ($(ALL_APPS),1)
	apps_to_build := ${all_buildable_apps}
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

# If building for coverage, don't build guis for now
ifeq ($(NO_GUIS),1)
	all_guis :=
endif


ifeq ($(MAGAOX_ROLE),RTC)
  guis_to_build =
else ifeq ($(MAGAOX_ROLE),ICC)
  guis_to_build =
else ifeq ($(findstring ACC,$(MAGAOX_ROLE)),ACC)
  guis_to_build =
else ifeq ($(MAGAOX_ROLE),TIC)
  guis_to_build =
else ifeq ($(MAGAOX_ROLE),headless)
  guis_to_build =
else
  guis_to_build = $(all_guis)
endif

all_rtimv_plugins = \
	cameraStatus \
	indiDictionary \
	pwfsAlignment \
	dmStatus \
	warnings \
	acquisition

ifeq ($(MAGAOX_ROLE),RTC)
  rtimv_plugins_to_build =
else ifeq ($(MAGAOX_ROLE),ICC)
  rtimv_plugins_to_build =
else ifeq ($(findstring ACC,$(MAGAOX_ROLE)),ACC)
  rtimv_plugins_to_build =
else ifeq ($(MAGAOX_ROLE),TIC)
  rtimv_plugins_to_build =
else ifeq ($(MAGAOX_ROLE),headless)
  rtimv_plugins_to_build =
else
  rtimv_plugins_to_build = $(all_rtimv_plugins)
endif

utils_to_build = \
	logdump \
	logsurgeon \
	cursesINDI \
	shmimDelta \
	xrif2fits \
	xrif2shmim

scripts_to_install = \
	query_seeing \
	sync_cacao \
	xctrl \
	netconsole_logger \
	dmdispbridge \
	shmimTCPreceive \
	shmimTCPtransmit \
	obs_to_movie \
	instrument_backup_sync \
	cacao_startup_if_present \
	git_check_all \
	collect_camera_configs_for_darks \
	shot_in_the_dark \
	write_magaox_pidfile \
	mount_cgroups1_cpuset \
	killIndiZombies \
	xlog \
	inventory_files \
	list_xfiles_by_semester \
	loop_instrument_backup_sync \
	cyverse_replicate 

ifeq ($(MAGAOX_ROLE),RTC)
  scripts_to_install += cacao/RTC/cacao-startup
  scripts_to_install += cacao/RTC/cacao-shutdown
  scripts_to_install += cacao/RTC/tweeter-vispyr-rootdir-scripts/tweeter-pre-calib-apply
  scripts_to_install += cacao/RTC/tweeter-vispyr-rootdir-scripts/tweeter-post-calib-apply
  scripts_to_install += cacao/RTC/woofer-vispyr-rootdir-scripts/woofer-pre-calib-apply
  scripts_to_install += cacao/RTC/woofer-vispyr-rootdir-scripts/woofer-post-calib-apply
  scripts_to_install += cacao/hoblockleaks
else ifeq ($(MAGAOX_ROLE),ICC)
  scripts_to_install += cacao/ICC/cacao-startup
  scripts_to_install += cacao/ICC/cacao-shutdown
  scripts_to_install += cacao/ICC/ncpc-rootdir-scripts/pre-calib-apply
  scripts_to_install += cacao/ICC/ncpc-rootdir-scripts/post-calib-apply
  scripts_to_install += cacao/hoblockleaks
  scripts_to_install += cacao/ICC/lowfs_switch
  scripts_to_install += shift_ncpc
else ifeq ($(MAGAOX_ROLE),TIC)
  scripts_to_install += cacao/TIC/cacao-startup
  scripts_to_install += cacao/TIC/cacao-shutdown
  scripts_to_install += cacao/TIC/kilo-rootdir-scripts/pre-calib-apply
  scripts_to_install += cacao/TIC/kilo-rootdir-scripts/post-calib-apply
  scripts_to_install += cacao/hoblockleaks
endif

.PHONY: all
all: indi_all libs_all flatlogs/bin/flatlogcodes apps_all guis_all rtimv_plugins_all utils_all

.PHONY: basic
basic: indi_all libs_all flatlogs/bin/flatlogcodes

.PHONY: install
install: indi_install libs_install pythonlibs_install apps_install pythonapps_install guis_install rtimv_plugins_install utils_install scripts_install rtscripts_install

#We clean just libMagAOX, and the apps, guis, and utils for normal devel work.
.PHONY: clean
clean: libs_clean apps_clean pythonapps_clean guis_clean utils_clean tests_clean

#Clean everything.
.PHONY: all_clean
all_clean: indi_clean libs_clean flatlogs_clean libs_clean apps_clean guis_clean rtimv_plugins_clean utils_clean doc_clean tests_clean

flatlogs/bin/flatlogcodes: flatlogs/src/flatlogcodes.cpp
	cd flatlogs/src/ && ${MAKE} install

.PHONY: flatlogs_all
flatlogs_all: flatlogs/bin/flatlogcodes

.PHONY: flatlogs_clean
flatlogs_clean:
	cd flatlogs/src/ && ${MAKE} clean
	rm -rf flatlogs/bin

.PHONY: indi_all
indi_all:
	cd INDI && ${MAKE} all

.PHONY: indi_install
indi_install: indi_all
	cd INDI && ${MAKE} install

.PHONY: indi_clean
indi_clean:
	cd INDI && ${MAKE} clean

libMagAOX/libMagAOX.a: flatlogs/bin/flatlogcodes libMagAOX/app/*.hpp \
		libMagAOX/app/dev/*.hpp libMagAOX/common/*.hpp \
		libMagAOX/file/*.hpp \
		libMagAOX/ImageStreamIO/*.hpp \
		libMagAOX/logger/*.hpp \
		libMagAOX/logger/types/*.hpp \
		libMagAOX/sys/*.hpp \
		libMagAOX/tty/*.hpp \
		libMagAOX/modbus/*.hpp
	cd libMagAOX/ && ${MAKE} all

.PHONY: libs_all
libs_all: flatlogs/bin/flatlogcodes libMagAOX/libMagAOX.a
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib && ${MAKE} )|| exit 1; \
	done

.PHONY: libs_install
libs_install: libs_all
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib && ${MAKE} install) || exit 1; \
	done
	sudo -H bash -c "echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf"
	sudo -H ldconfig


.PHONY: libs_clean
libs_clean:
	cd libMagAOX && ${MAKE} clean
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib && ${MAKE}  clean) || exit 1; \
	done

.PHONY: pythonlibs_install
pythonlibs_install: installed_python_interface_timestamp.txt

# Installing the Python interface makes a file so we're not "phony"
# and it should only re-run if the Python source is changed
PY_SOURCES := magaox-python/pyproject.toml \
	$(shell find magaox-python/magaox -name '*.py')

installed_python_interface_timestamp.txt: $(PY_SOURCES)
	sudo $(PYTHON) -m pip install ./magaox-python
	date -u -Iseconds > ./installed_python_interface_timestamp.txt

.PHONY: apps_all
apps_all: libs_all indi_all
	for app in ${apps_to_build}; do \
		(cd apps/$$app && ${MAKE} )|| exit 1; \
	done

.PHONY: apps_install
apps_install: libs_install indi_install
	for app in ${apps_to_build}; do \
		(cd apps/$$app && ${MAKE}  install) || exit 1; \
	done

.PHONY: pythonapps_install
pythonapps_install: installed_python_interface_timestamp.txt
	for app in ${pythonapps_to_install}; do \
		(cd apps/$$app && ${MAKE} install) || exit 1; \
	done

.PHONY: apps_clean
apps_clean:
	for app in ${apps_to_build}; do \
		(cd apps/$$app && ${MAKE}  clean) || exit 1; \
	done

.PHONY: guis_all
guis_all: libs_all indi_all rtimv_plugins_all libMagAOX/libMagAOX.hpp.gch libMagAOX/libMagAOX.a
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui && ${MAKE} )|| exit 1; \
	done

.PHONY: guis_install
guis_install: libs_install indi_install rtimv_plugins_install
	for gui in ${guis_to_build}; do \
		(cd gui/apps/$$gui && ${MAKE} install) || exit 1; \
	done

.PHONY: guis_clean
guis_clean: rtimv_plugins_clean
	for gui in ${all_guis}; do \
		(cd gui/apps/$$gui && ${MAKE} clean) || exit 1; \
	done

.PHONY: rtimv_plugins_all
rtimv_plugins_all: indi_all libs_all
	for plg in ${rtimv_plugins_to_build}; do \
		(cd gui/rtimv/plugins/$$plg && ${MAKE} )|| exit 1; \
	done

.PHONY: rtimv_plugins_install
rtimv_plugins_install: indi_install libs_install
	for plg in ${rtimv_plugins_to_build}; do \
		(cd gui/rtimv/plugins/$$plg && ${MAKE} install) || exit 1; \
	done

.PHONY: rtimv_plugins_clean
rtimv_plugins_clean:
	for plg in ${rtimv_plugins_to_build}; do \
		(cd gui/rtimv/plugins/$$plg && ${MAKE} clean) || exit 1; \
	done

.PHONY: scripts_install
scripts_install:
	for script in ${scripts_to_install}; do \
		sudo -H install -d /opt/MagAOX/bin && \
		sudo -H install scripts/$$script /opt/MagAOX/bin ; \
	done
	for script in $(notdir ${scripts_to_install}); do \
		sudo -H ln -fs /opt/MagAOX/bin/$$script /usr/local/bin/$$script; \
	done

.PHONY: rtscripts_install
rtscripts_install:
	for scriptname in make_cpusets move_irqs; do \
		sudo -H install -d /opt/MagAOX/bin && \
		if [ -e rtSetup/$(MAGAOX_ROLE)/$$scriptname ]; then \
			sudo -H install rtSetup/$(MAGAOX_ROLE)/$$scriptname /opt/MagAOX/bin/$$scriptname && \
			sudo -H ln -fs /opt/MagAOX/bin/$$scriptname /usr/local/bin/$$scriptname; \
		else \
			echo "echo 'No $$scriptname for $$MAGAOX_ROLE'\nexit 0" | sudo -H tee /opt/MagAOX/bin/$$scriptname && \
			sudo -H chmod +x /opt/MagAOX/bin/$$scriptname && \
			sudo -H ln -fs /opt/MagAOX/bin/$$scriptname /usr/local/bin/$$scriptname; \
		fi \
	; done

.PHONY: utils_all
utils_all: flatlogs/bin/flatlogcodes indi_all libMagAOX/libMagAOX.hpp.gch libMagAOX/libMagAOX.a
		for app in ${utils_to_build}; do \
			(cd utils/$$app && ${MAKE}) || exit 1; \
		done

.PHONY: utils_install
utils_install: flatlogs/bin/flatlogcodes indi_install utils_all
		for app in ${utils_to_build}; do \
			(cd utils/$$app && ${MAKE} install) || exit 1; \
		done

.PHONY: utils_clean
utils_clean:
		for app in ${utils_to_build}; do \
			(cd utils/$$app && ${MAKE} clean) || exit 1; \
		done

.PHONY: test
test: tests_clean
	cd tests && ${MAKE} test || exit 1;

.PHONY: tests_clean
tests_clean:
	cd tests && ${MAKE} clean || exit 1;
	cd libMagAOX/logger/tests && ${MAKE} clean || exit 1;

.PHONY: doc
doc:
	doxygen doc/config/Doxyfile.libMagAOX

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

.PHONY: print_role
print_role:
	@echo "MAGAOX_ROLE=$(MAGAOX_ROLE)"

.PHONY: coverage
coverage: coverage_clean
	${MAKE} all COVERAGE=1 ALL_APPS=1 NO_GUIS=1

.PHONY: coverage_clean
coverage_clean:
	find . -name '*.gcno' -delete
	find . -name '*.gcda' -delete
	find . -name '*.gcov' -delete
	${MAKE} all_clean COVERAGE=1 ALL_APPS=1
	cd tests && ${MAKE} realclean COVERAGE=1 || exit 1;
	cd libMagAOX/logger/tests && ${MAKE} really_clean COVERAGE=1 || exit 1;

.PHONY: valgrind
valgrind:
	${MAKE} all ALL_APPS=1 DEBUG=1
