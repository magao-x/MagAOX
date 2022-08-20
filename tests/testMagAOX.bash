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


echo Running MagAO-X Tests

tests=$(cat tests.list)

for test in $tests; do \
   echo running $test; \
   $test 2>test_stderr.txt ; \
done

