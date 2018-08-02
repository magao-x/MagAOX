
apps_to_build = magAOXMaths

utils_to_build = logdump


all: indi_all apps_all utils_all

install: indi_install apps_install utils_install

clean: indi_clean apps_clean utils_clean

indi_all:
	cd INDI; ${MAKE} all

indi_install:
	cd INDI; ${MAKE} install

indi_clean:
	cd INDI; ${MAKE} clean

apps_all:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE} -f ../../Make/magAOXApp.mk t=$$app) || break; \
	done

apps_install:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE} -f ../../Make/magAOXApp.mk t=$$app install) || break; \
	done

apps_clean:
	for app in ${apps_to_build}; do \
		(cd apps/$$app; ${MAKE} -f ../../Make/magAOXApp.mk t=$$app clean) || break; \
	done

utils_all:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} -f ../../Make/magAOXUtil.mk t=$$app) || break; \
		done

utils_install:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} -f ../../Make/magAOXUtil.mk t=$$app install) || break; \
		done

utils_clean:
		for app in ${utils_to_build}; do \
			(cd utils/$$app; ${MAKE} -f ../../Make/magAOXUtil.mk t=$$app clean) || break; \
		done

.PHONY: setup
setup:
		@for file in ./local/*.example.mk; do \
			dest=$$(echo $$file | sed 's/.example//'); \
			if [ ! -e $$dest ]; then cp -v $$file $$dest; fi \
		done
		@echo "***\nBuild settings available in local/Common.mk\n***"
		@grep "?=" mk/Common.mk || true
		@echo "***"
		@echo "Build settings available in local/MxLib.mk\n***"
		@grep "?=" mk/MxLib.mk || true
		@echo "***"
		@echo "Build settings available in local/MxApp.mk\n***"
		@grep  "?=" mk/MxApp.mk || true
		@echo "***"
