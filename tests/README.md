# MagAO-X tests

## Building and running

- `tests/tests.list` names every test binary, one path per line relative to this directory.
- `make` in this directory builds them all through `Makefile.one`, one test per goal, so
  `make -jN` builds in parallel. `make -f Makefile.one t=<path from tests.list>` builds one.
- Build with `COVERAGE=1` when the tree was built for coverage. The shared `testMain.o` and
  the libraries the tests link are instrumented then, and a test built without the flag
  fails to link with `__gcov_*` undefined.
- `bash testMagAOX.bash` runs every binary in `tests.list`, then runs the logger's own
  generated tests with `make run COVERAGE=1` in `libMagAOX/logger/tests`.
- `tests/coverage/update_coverage` rebuilds the HTML coverage report. See its header.

## Generated pieces

Two generators are involved. Both are Python plus a jinja2 template.

| what | input | generator | output | run by |
|---|---|---|---|---|
| fault injection macros | `libMagAOX/<area>/tests/xwcTestNames.txt` | `tests/genTestMacros.py` + `tests/xwcTestMacroTemplate.jinja2` | `libMagAOX/<area>/tests/testMacros.hpp`, committed | by hand, see [genTestMacros.md](genTestMacros.md) |
| per log type Catch2 tests, needs Python 3.12 | `libMagAOX/logger/types/*.hpp` and `*.fbs` | `libMagAOX/logger/tests/generateTemplatedCatch2Tests.py` + `catch2TestTemplate.jinja2` | `libMagAOX/logger/tests/generated_tests/*.cpp`, ignored by git | `make` in `libMagAOX/logger/tests`, see [flatlogs.md](../libMagAOX/logger/tests/flatlogs.md) |
| entropy tests | the generated tests above | `generateEntropyTests.py` + `entropyTestTemplate.jinja2` | `gen_entropy_tests/`, ignored by git | same Makefile |

## Other support files here

- `catch2/catch.hpp` is the single header Catch2 used by every test.
- `testMain.cpp` is the shared `main()` linked into every `tests.list` binary.
- `testXWC.hpp` has the small helpers shared by tests, such as `XWCTEST_DOXYGEN_REF`.
