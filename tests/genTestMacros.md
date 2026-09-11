# XWCTEST fault injection macros

How a unit test reaches an error branch in production code that no real input can
trigger, without leaving any test logic in a production build.

## The pieces

```
libMagAOX/<area>/tests/xwcTestNames.txt      hand written. One XWCTEST_<NAME> per line.
            |
            |  python3 tests/genTestMacros.py --names <that file> --out <header below>
            |  run by hand from the repo root. No Makefile runs it. The output is committed.
            v
libMagAOX/<area>/tests/testMacros.hpp        generated. One XWCTEST_IF_<NAME>(line) macro per name.
            ^
            |  #include "tests/testMacros.hpp"
            |
libMagAOX/<area>/<header>.hpp or .cpp        production code. At the fault site:
                                                 XWCTEST_IF_<NAME>( statement );
            ^
            |  #define XWCTEST_<NAME>          before including the production header
            |
libMagAOX/<area>/tests/<x>_test.cpp          the unit test
```

The template `tests/xwcTestMacroTemplate.jinja2` renders each name to this:

```cpp
#undef XWCTEST_IF_FOO
#ifdef XWCTEST_FOO
#define XWCTEST_IF_FOO(line) { line ; } /* LCOV_EXCL_LINE */
#else
#define XWCTEST_IF_FOO(line) do {} while(0)
#endif
```

So a production build, which never defines any `XWCTEST_` name, compiles every site to an
empty statement. A test that defines `XWCTEST_FOO` before the include gets the statement
compiled in at that site. The `LCOV_EXCL_LINE` inside the macro body hides the injected
statement itself from coverage, because it is not production code.

`testMacros.hpp` has no include guard on purpose, and every macro starts with `#undef`.
That lets a test include the production header more than once with different names
defined and get a different set of macros each time. See the next section.

## Two ways a test uses it

### One fault per translation unit

Define the name once at the top of the test file, before the includes. Every site for that
name fires for the whole file. This works when the test file only needs the faulted
behavior, or has its own binary per fault.

### Several faults in one test file: XWCTEST_NAMESPACE

Most test files need the normal class and several faulted variants side by side. For that
the production header wraps its contents in an optional namespace:

```cpp
namespace MagAOX { namespace app {
#ifdef XWCTEST_NAMESPACE
namespace XWCTEST_NAMESPACE {
#endif
   ... the class ...
#ifdef XWCTEST_NAMESPACE
} // namespace XWCTEST_NAMESPACE
#endif
} }
```

The test includes the header normally once, then again per fault, undefining the include
guard each time:

```cpp
#include "../indiDriver.hpp"                              // production copy

#undef app_indiDriver_hpp                                 // the header's include guard
#define XWCTEST_NAMESPACE XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL_ns
#define XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL
#include "../indiDriver.hpp"                              // faulted copy
#undef XWCTEST_NAMESPACE
#undef XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL
```

The test then uses `MagAOX::app::XWCTEST_INDIDRIVER_CTRL_WRITE_FAIL_ns::indiDriver` where
it wants the fault and plain `MagAOX::app::indiDriver` everywhere else. Each re-inclusion
re-reads `testMacros.hpp`, so only the one name defined at that moment is active in that
copy. Any other production headers pulled in by the re-included header keep their include
guards and are not recompiled.

Things a header needs for this to work:

- Free functions, enums, or constants outside the class must sit inside their own include
  guard, or the second inclusion redefines them. See the helper guard in
  `libMagAOX/logger/logMap.hpp` and the constants guard in `libMagAOX/modbus/modbus.hpp`.
- An explicit template instantiation must be skipped in the namespaced copies. See the
  end of `libMagAOX/logger/logMap.cpp`.
- For a `.cpp` file the test includes the `.cpp` itself. See `libMagAOX/tty/tests/ttyUSB_test.cpp`
  and `libMagAOX/modbus/tests`.

## What coverage sees

gcov counts hits per source line no matter how many copies of the header were compiled,
so a branch reached only through a faulted copy still shows as covered on the one
production line. `genhtml --merge-aliases` in `tests/coverage/update_coverage` folds the
per-namespace function copies back together in the report.

## Rules for a hook site

- The macro argument is one or more statements. Top level commas split the macro
  argument, so wrap a comma expression in parentheses:
  `XWCTEST_IF_X( ( a = 1, b = 2 ) );`
- A hook can only add statements. If the fault needs a statement removed, keep a raw
  `#ifndef XWCTEST_...` around it. See `XWCTEST_LOGMANAGER_LOGTHREADSTART_NOT_JOINABLE`
  in `libMagAOX/logger/logManager.hpp`.
- Several statements, including nested blocks and loops, can go in one `XWCTEST_IF_` call
  as long as no comma sits outside parentheses. See `XWCTEST_MAGAOXAPP_EXEC_LOG_DEATH` in
  `libMagAOX/app/MagAOXApp.hpp`.
- The block scope means a declaration that later code uses cannot be injected with the
  macro. That is the one case kept as a raw `#ifdef`: the `testTimesThrough` counter for
  `XWCTEST_MAGAOXAPP_EXEC_NORM` in `MagAOXApp.hpp`. A bare `return -1` is fine inside the
  block and uses `XWCTEST_IF_`.
- Redefining a macro, such as the signal names in `setSigTermHandler()`, cannot be done
  through a function-like macro and stays a raw `#ifdef`. The header comment of
  `libMagAOX/app/tests/xwcTestNames.txt` lists those.
- A fault that replaces a constant uses `#ifndef X / #define X <production value>` at the
  top of the file so the test can predefine it. See the sysfs overrides in
  `libMagAOX/tty/ttyUSB.cpp`.
- Names are `XWCTEST_<AREA>_<SITE>_<FAULT>`, all upper case, for example
  `XWCTEST_LOGFILERAW_FLUSH_FFLUSH_FAIL`. The generator strips the leading `XWCTEST_` and
  adds `XWCTEST_IF_` for the callable form.
- A site that uses a name missing from the names file fails to compile with
  "XWCTEST_IF_... was not declared", because the macro does not exist.

## Adding a hook

1. Add `XWCTEST_<NAME>` to the `xwcTestNames.txt` in the tests directory next to the
   production file. Keep it under the comment naming the production file it belongs to.
2. Regenerate that directory's header. From the repo root:
   `python3 tests/genTestMacros.py --names libMagAOX/<area>/tests/xwcTestNames.txt --out libMagAOX/<area>/tests/testMacros.hpp`
3. Make sure the production file has `#include "tests/testMacros.hpp"` after its other
   includes and inside its include guard.
4. Put `XWCTEST_IF_<NAME>( statement );` at the fault site with a one line comment saying
   what the fault pretends.
5. In the test, define the name for the whole file or use the `XWCTEST_NAMESPACE`
   re-inclusion above, and write the scenario.
6. Commit the regenerated `testMacros.hpp` with the change.

The generator needs Python 3.9 or newer and the `jinja2` module. It does not install
jinja2 itself.

## Where it is used

```
grep -rln 'tests/testMacros.hpp' libMagAOX --include=*.hpp --include=*.cpp   # production sites
grep -rln XWCTEST_NAMESPACE libMagAOX/*/tests libMagAOX/*/*/tests            # tests using re-inclusion
```
