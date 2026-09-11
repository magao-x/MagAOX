########################################################################
### Hide commands
AT=@
GEN_TEST_FILES = ./generated_tests/*_generated_tests.cpp
# common.mk supplies the repo standard CXXFLAGS. That includes the COVERAGE=1 flags,
# which were previously duplicated here. python.mk supplies $(PYTHON). The test
# generator uses syntax that needs Python 3.10 or newer, which the system python3
# may not be.
include ../../../Make/common.mk
include ../../../Make/python.mk
### default entropy test settings
# seed
s=1
# entropy level
e=1
# number of flatlog types to use
n=2
# types to use
f=""

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
generated_tests: $(GEN_TEST_FILES)

$(GEN_TEST_FILES) :
	$(AT)$(PYTHON) generateTemplatedCatch2Tests.py

# The link needs -lflatbuffers because the generated tests exercise logJsonFormat.
# Its schema reflection uses flatbuffers::Parser::Deserialize and GenText, which live
# in the flatbuffers library. The build needs stateCodes.cpp because the generated
# per-type tests round-trip state_change logs, whose formatters call
# stateCodes::codeText().
catch2_tests: ../generated/binarySchemata.inc $(GEN_TEST_FILES) catch2_tests.cpp
	$(AT)g++ $(CXXFLAGS) -I. -I./../types $(GEN_TEST_FILES) catch2_tests.cpp ../../app/stateCodes.cpp -o $@ -lflatbuffers

do_catch2_test: catch2_tests
	$(AT)./catch2_tests

########################################################################
### Build and run Catch2 Entropy tests
entropy_test: ../generated/binarySchemata.inc catch2_tests.cpp
	$(AT)$(PYTHON) generateEntropyTests.py -e $(e) -n $(n) -s $(s) -f $(f)
	$(AT)g++ $(CXXFLAGS) -I. -I.. -I./../types -I./generated_tests gen_entropy_tests/generated_test_e$(e)_n$(n).cpp catch2_tests.cpp ../../app/stateCodes.cpp -o generated_entropy_test_e$(e)_n$(n) -lflatbuffers

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
