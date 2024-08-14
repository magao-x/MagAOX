#VERBOSE=--verbose
QUIET=--quiet

test: strcat_varargs_test named_pipe_test_d
	@./strcat_varargs_test $(VERBOSE) $(QUIET)
	@$(RM) notfifo notfifo.failed afifo
	@mkfifo afifo
	@./named_pipe_test_d afifo 1>/dev/null
	@touch notfifo
	@./named_pipe_test_d --exit-on-error notfifo 1>/dev/null 2>&1 || touch notfifo.failed
	@[ -f notfifo.failed ] || ( echo "Failed to fail" && false )
	@$(RM) afifo notfifo notfifo.failed

strcat_varargs_test :: strcat_varargs_test.c ../../INDI/INDI/strcat_varargs.c

named_pipe_test_d :: named_pipe_test_d.c ../../INDI/INDI/pipe_wrapper.c

clean:
	@$(RM) strcat_varargs_test named_pipe_test_d
