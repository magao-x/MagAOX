Adding An application {#page_module_appadd}
==========

[TOC]

------------------------------------------------------------------------

# Introduction

This document describes how to add an application to the MagAO-X source code.

# 1. Code

Start by copying the folder `template` to a new folder with the name of the app, e.g. `hardwareCtrl`.

The three basic files for any application are the header, the main program file, and the Makefile.  In the new `hardwareCtrl` folder, rename `template.hpp` and `template.cpp` to `hardwareCtrl.hpp` and `hardwareCtrl.cpp` (substituting the appropriate name for the new application for `hardwareCtrl`).  

Now in `Makefile`, `hardwareCtrl.hpp` and `hardwareCtrl.cpp`, change `template` to `hardwareCtrl` (you should be able to use find-all and replace).  

If all replacement is done correctly, the application will build with only warnings if you type `make` on the command line.

Next edit the code in the `.hpp` file to implement the application.  You can also edit the Makefile adding additional libraries, or perhaps another header.  You will typically not need to edit any code in the `.cpp` file other than replacing `template` as above.

# 2. Build System Integration

To cause the new app to be built, add it to the appropriate list of apps in the top level Makefile.  Pay attention to which machine you expect the app to run on.

# 3. Tests

The file `tests/template_test.cpp` should have its name changed to (example) `tests/hradwareCtrl_test.cpp`.  Implement any unit tests in this file.

Add the `*_test.cpp` file to the top level `tests/testMagAOX.cpp` file, and to the `tests/Makefile` `TEST_INCLUDES` dependency list.

# 4. Software Documentation

Document your code with doxygen.  Be sure that `template` was changed to the application name in all documentation blocks in the source code, including the group definitions and `\ingroup` directives.

# 5. Program Documentation

Rename and edit the file `doc/template.md`, in the above example it should become `doc/hardwareCtrl.md`.  Change all instances of `template` to the application name, update the text appropriately, including the app specific options and INDI section.

# 6. Documentation System Integration

Next, follow all of these steps to integrate the documentation:
- in the file `libMagAOX/doc/libMagAOX_doxygen.in` add the application folder to the INPUT directive 
- in the same file, add the application/doc folder to the EXCLUDE directive
- in the file 'apps/doc/magaox_apps_doxygen.in`  add the application `xxxx/doc/` folder to the `INPUT` directive

Now running `make doc` in the top level should build all documentation with your new application integrated into it like all the others.

# 7. Final Steps

Delete this `readme.md` from the new application folder.

Commit all of the new files.

Use your new application to find planets.
