
SELF_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
-include $(SELF_DIR)/../local/common.mk

libs_to_build = libtelnet

apps_to_build =  filterWheelCtrl \
                 magAOXMaths \
                 mzmqServer \
                 siglentSDG \
                 sshDigger \
                 sysMonitor \
                 trippLitePDU \
                 xindiserver \
                 xt1121Ctrl \
                 xt1121DCDU \
                 zaberCtrl

ifneq ($(PYLON),false)
apps_to_build += baslerCtrl
endif

ifneq ($(PICAM), false)
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
	/bin/sudo bash -c "echo $(LIB_PATH) > /etc/ld.so.conf.d/magaox.conf"
	/bin/sudo ldconfig

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
		/bin/sudo install -d /opt/MagAOX/bin && \
		/bin/sudo install scripts/$$script /opt/MagAOX/bin  && \
		/bin/sudo ln -fs /opt/MagAOX/bin/$$script /usr/local/bin/$$script; \
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
	cd libMagAOX/doc; doxygen libMagAOX_doxygen.in; cp -r sw_html ../../doc/www/;
	cd utils/doc; doxygen magaox_utils_doxygen.in; cp -r  util_html ../../doc/www/;
	cd apps/doc; doxygen magaox_apps_doxygen.in; cp -r apps_html ../../doc/www/;

.PHONY: doc_clean
doc_clean:
	rm -rf libMagAOX/doc/sw_html;
	rm -rf utils/doc/util_html
	rm -rf apps/doc/apps_html
	rm -rf www/*_html

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
