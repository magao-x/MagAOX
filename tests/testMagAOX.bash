#!/bin/bash
#############################################################
#            Build and run all MagAO-X tests                #
#                                                           #
# - This should be called with pwd=tests                    #
# - First run make to build all tests (see Makefile)        #
# - Tests are listed in `tests.list`                        #
# - Calls each test in succession.                          #
#                                                           #
# TODO: need to capture catch2 output and reports           #
# TODO: make is so pwd doesn't have to be tests             #
#                                                           #
#############################################################

#Do NOT enable the following, otherwise we won't continue after a failed test
#set -eo pipefail

# Strip any trailing CRs from tests.list, in place.
# If there aren't any CRs, no changes are made.
# This was an issue I ran across. Apparently it would happen if the file
# was edited on Windows, which I assume is unlikely, but also if it
# comes from a git repo that hasn’t normalized line endings
sed -i 's/\r$//' tests.list

echo Running MagAO-X Tests

tests=$(cat tests.list)

: > test_stderr.txt
for test in $tests; do \
   echo running $test | tee -a test_stderr.txt; \
   $test 2>>test_stderr.txt ; \
   echo '' >> test_stderr.txt; \
done

cd ../libMagAOX/logger/tests
make run COVERAGE=1


