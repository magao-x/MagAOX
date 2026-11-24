########################################################################
### Hide commands
AT=@
GEN_TEST_FILES = ./generated_tests/*_generated_tests.cpp
C_STD=c++20

# Flatlogs uses mxlib headers and must share its include paths and ABI flags.
MXLIB_CFLAGS = $(shell pkg-config --cflags mxlib)
MXLIB_LIBS = $(shell pkg-config --libs mxlib)
CXXFLAGS += $(MXLIB_CFLAGS)
LDLIBS += $(MXLIB_LIBS)

### default entropy test settings
# seed
s=1
# entropy level
e=1
# number of flatlog types to use
n=2
# types to use
f=""

ifeq ($(COVERAGE),1)
  LDFLAGS += --coverage
  CFLAGS += --coverage
  CXXFLAGS += --coverage
endif

########################################################################
### Generate code, run tests
all: ../generated/binarySchemata.inc do_catch2_test do_entropy_test

build: ../generated/binarySchemata.inc catch2_tests entropy_test

run: ../generated/binarySchemata.inc do_catch2_test do_entropy_test

.PHONY: coverage_build
coverage_build:
	${MAKE} build COVERAGE=1

.PHONY: coverage
coverage:
	${MAKE} all COVERAGE=1

########################################################################
### Generate code and flatbuffer binary schema from text schemas (*.fbs)
../generated/binarySchemata.inc: ../types/*.hpp ../types/schemas/*.fbs
	$(AT)cd .. && \
	$(RM) -R "./types/generated/" "./generated/" && \
	../../flatlogs/src/flatlogcodes 2>/dev/null && \
	cd types/generated; for bfb in *.bfbs; do xxd -i $$bfb > $$(basename $$bfb).h; done;
	$(AT)cat ../types/generated/*.bfbs.h > $@

########################################################################
### Build and run Catch2 tests
.PHONY: generated_tests do_catch2_test entropy_test do_entropy_test test_generator
test_generator:
	$(AT)python3 -m unittest -v test_generateTemplatedCatch2Tests.py

generated_tests: generated_tests/.stamp

generated_tests/.stamp: generateTemplatedCatch2Tests.py catch2TestTemplate.jinja2 ../types/*.hpp ../types/schemas/*.fbs ../generated/binarySchemata.inc
	$(AT)python3 generateTemplatedCatch2Tests.py
	$(AT)touch $@

catch2_tests: ../generated/binarySchemata.inc generated_tests/.stamp $(wildcard $(GEN_TEST_FILES)) catch2_tests.cpp Makefile
	$(AT)$(CXX) -std=$(C_STD) $(CPPFLAGS) $(CXXFLAGS) -I. -I./../types -I../../../flatlogs/include/ $(GEN_TEST_FILES) catch2_tests.cpp -o $@ $(LDFLAGS) $(LDLIBS)

do_catch2_test: catch2_tests
	$(AT)./catch2_tests

########################################################################
### Build and run Catch2 Entropy tests
entropy_test: ../generated/binarySchemata.inc generated_tests/.stamp catch2_tests.cpp
	$(AT)python3 generateEntropyTests.py -e $(e) -n $(n) -s $(s) -f $(f)
	$(AT)$(CXX) -std=$(C_STD) $(CPPFLAGS) $(CXXFLAGS) -I. -I.. -I./../types -I../../../flatlogs/include/ -I./generated_tests gen_entropy_tests/generated_test_e$(e)_n$(n).cpp catch2_tests.cpp -o generated_entropy_test_e$(e)_n$(n) $(LDFLAGS) $(LDLIBS)

do_entropy_test: entropy_test
	$(AT)./generated_entropy_test_e$(e)_n$(n) "Scenario: test_e$(e)_n$(n)"

########################################################################
### Build and run non-Catch2 test (will probably remove at some point)
#do_tests: ../generated/binarySchemata.inc do_tests.cpp
#	$(AT)g++ -std=c++11 -I. -I./../types -I../../../flatlogs/include/ do_tests.cpp -o $@

#do_test: do_tests
#	$(AT)./do_tests

########################################################################
clean:
	$(AT)$(RM) -R "./types/generated/" "./generated/" ./do_tests Schemata_test catch2_tests
	$(AT)$(RM) -R "./gen_entropy_tests/" generated_entropy_test_e1_n2

really_clean: clean
	$(AT)$(RM) -R "./generated_tests/" "./gen_entropy_tests/" __pycache__

coverage_clean:
	find . -name '*.gcno' -delete
	find . -name '*.gcda' -delete
	find . -name '*.gcov' -delete
	${MAKE} really_clean COVERAGE=1
	$(AT)$(RM) -R "./generated_tests/"
