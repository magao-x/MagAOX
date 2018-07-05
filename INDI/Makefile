

toBuild = liblilxml libcommon INDI xindidriver
toInstall = INDI xindidriver

all:
	for dir in ${toBuild}; do \
	 (cd $$dir; ${MAKE} all) || break; \
	done

install: 
	for dir in ${toInstall}; do \
   	(cd $$dir; ${MAKE} install) || break; \
	done

clean:
	for dir in ${toBuild}; do \
   	(cd $$dir; ${MAKE} clean) || break; \
	done
