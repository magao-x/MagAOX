/** \file testMagAOX.cpp
  * \brief The MagAO-X library-wide Catch2 main file for testing.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-08-30 created by JRM
  * - 2018-12-27 converted to .cpp by JRM
  */

//------------- Do Not Edit -----------------
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
//------------------------------------------


//===========================================================
// Add relative path to <name>_test.cpp in an include in the
// appropriate section below
//==========================================================


//0 libMagAOX Tests
#include "../libMagAOX/tty/tests/ttyIOUtils_test.cpp"

#include "../libMagAOX/app/dev/tests/outletController_tests.cpp"

//1 App specific Tests

#include "../apps/ocam2KCtrl/tests/ocamUtils_test.cpp"

#include "../apps/siglentSDG/tests/siglentSDG_test.cpp"

#include "../apps/sshDigger/tests/sshDigger_tests.cpp"

#include "../apps/sysMonitor/tests/sysMonitor_test.cpp"

