########################################################################
### Hide commands
AT=@
GEN_TEST_FILES = ./generated_tests/*_generated_tests.cpp
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
all: ../generated/binarySchemata.inc do_catch2_test do_test

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
	$(AT)python3 generateTemplatedCatch2Tests.py

catch2_tests: ../generated/binarySchemata.inc $(GEN_TEST_FILES) catch2_tests.cpp
	$(AT)g++ -std=c++14 -I. -I./../types -I../../../flatlogs/include/ $(GEN_TEST_FILES) catch2_tests.cpp -o $@

do_catch2_test: catch2_tests
	$(AT)./catch2_tests

########################################################################
### Build and run Catch2 Entropy tests
entropy_test: ../generated/binarySchemata.inc catch2_tests.cpp 
	$(AT)python3 generateEntropyTests.py -e $(e) -n $(n) -s $(s) -f $(f)
	$(AT)g++ -std=c++14 -I. -I./../types -I../../../flatlogs/include/ -I./generated_tests gen_entropy_tests/generated_test_e$(e)_n$(n).cpp catch2_tests.cpp -o generated_entropy_test_e$(e)_n$(n)

do_entropy_test: entropy_test
	$(AT)./generated_entropy_test_e$(e)_n$(n) "Scenario: test_e$(e)_n$(n)"

########################################################################
### Build and run non-Catch2 test (will probably remove at some point)
do_tests: ../generated/binarySchemata.inc do_tests.cpp
	$(AT)g++ -std=c++11 -I. -I./../types -I../../../flatlogs/include/ do_tests.cpp -o $@

do_test: do_tests
	$(AT)./do_tests

########################################################################
clean:
	$(AT)$(RM) -R "./types/generated/" "./generated/" ./do_tests Schemata_test catch2_tests

really_clean: clean 
	$(AT)$(RM) -R "./generated_tests/"
