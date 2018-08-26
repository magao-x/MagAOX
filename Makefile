
libs_to_build = libtelnet

apps_to_build = xindiserver trippLitePDU magAOXMaths zaberCtrl

utils_to_build = logdump


all: indi_all libs_all apps_all utils_all

install: indi_install libs_install apps_install utils_install

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

libs_clean:
	for lib in ${libs_to_build}; do \
		(cd libs/$$lib; ${MAKE}  clean) || break; \
	done

lib_clean:
	cd libMagAOX; ${MAKE} clean

apps_all:
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
		@echo "***\nBuild settings available in local/Common.mk\n***"
		@grep "?=" mk/Common.mk || true
		@echo "***"
		@echo "Build settings available in local/MxLib.mk\n***"
		@grep "?=" mk/MxLib.mk || true
		@echo "***"
		@echo "Build settings available in local/MxApp.mk\n***"
		@grep  "?=" mk/MxApp.mk || true
		@echo "***"
