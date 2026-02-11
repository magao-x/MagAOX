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

#########################
## Coverage build
#########################

.PHONY: coverage
coverage:
	${MAKE} all COVERAGE=1

coverage_clean:
	find .. -path '*/tests/*.gcno' -delete
	find .. -path '*/tests/*.gcda' -delete
	find .. -path '*/tests/*.gcov' -delete
	${MAKE} clean COVERAGE=1

##################################################################################
## Sanitizer build
##
## Profiles:
##   sanitize_address   - Memory safety (ASan + UBSan + LSan)
##   sanitize_thread    - Concurrency/race detection (TSan + UBSan)
##   sanitize_undefined - Lightweight undefined behavior checks (UBSan only)
##
## The following runtime environment variables are set when running tests:
##   ASAN_OPTIONS=halt_on_error=0:detect_leaks=1
##   TSAN_OPTIONS=halt_on_error=0:second_deadlock_stack=1
##   UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0
##################################################################################

.PHONY: sanitize_address
sanitize_address:
	${MAKE} all SANITIZE=address

.PHONY: sanitize_thread
sanitize_thread:
	${MAKE} all SANITIZE=thread

.PHONY: sanitize_undefined
sanitize_undefined:
	${MAKE} all SANITIZE=undefined

# Aliases for convenience
.PHONY: asan tsan ubsan
asan: sanitize_address
tsan: sanitize_thread
ubsan: sanitize_undefined

# Run tests with sanitizers (continue through all tests to collect all findings)
.PHONY: test_asan test_tsan test_ubsan
test_asan: sanitize_address
	@echo "Running tests with AddressSanitizer..."
	@failed=0; \
	for test in ${TESTOBJS}; do \
	  ASAN_OPTIONS="halt_on_error=0:detect_leaks=1" ./$$test || failed=$$((failed + 1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
	  echo "$$failed test(s) had failures (may be test logic or sanitizer findings)"; \
	fi

# setting no-pie flag doesn't seem to be enough sometimes; use setarch -R to disable Address Space Layout Randomization
# In containers, setarch may fail with "Operation not permitted" - fall back to running without it
test_tsan: sanitize_thread
	@echo "Running tests with ThreadSanitizer..."
	@failed=0; \
	for test in ${TESTOBJS}; do \
	  if command -v setarch >/dev/null 2>&1 && setarch `uname -m` -R true 2>/dev/null; then \
	    env TSAN_OPTIONS="halt_on_error=0:second_deadlock_stack=1" setarch `uname -m` -R ./$$test || failed=$$((failed + 1)); \
	  else \
	    TSAN_OPTIONS="halt_on_error=0:second_deadlock_stack=1" ./$$test || failed=$$((failed + 1)); \
	  fi; \
	done; \
	if [ $$failed -gt 0 ]; then \
	  echo "$$failed test(s) had failures (may be test logic or sanitizer findings)"; \
	fi

test_ubsan: sanitize_undefined
	@echo "Running tests with UndefinedBehaviorSanitizer..."
	@failed=0; \
	for test in ${TESTOBJS}; do \
	  UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0" ./$$test || failed=$$((failed + 1)); \
	done; \
	if [ $$failed -gt 0 ]; then \
	  echo "$$failed test(s) had failures (may be test logic or sanitizer findings)"; \
	fi