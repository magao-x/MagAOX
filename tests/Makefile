############################################################
#                makefile for MagAOX tests                 #
#                                                          #  
# Add tests to tests.list  and build with `make`           #
#                                                          #
#                                                          #
# notes:                                                   #
#   -- you do not need to edit anything else in this file  #
#   -- if the test requires setuid, add a conditional to   #
#      Makefile.one                                        #
#                                                          #
############################################################   

TESTOBJS = `cat tests.list`
        
##################################################
## Should not need to edit from here on:
##################################################

all:
	@for test in ${TESTOBJS}; do \
	  $(MAKE) --no-print-directory -f Makefile.one t=$$test; \
	done

.PHONY: test
test: all
	@for test in ${TESTOBJS}; do \
	  ./$$test || exit 1; \
	done

.PHONY: clean
clean:
	@echo cleaning . . .
	@for test in ${TESTOBJS}; do \
	  $(MAKE) --no-print-directory -f Makefile.one t=$$test clean; \
	done

.PHONY: realclean
realclean: clean
	@rm -f testMain.o
	@rm -f test_stderr.txt
	
