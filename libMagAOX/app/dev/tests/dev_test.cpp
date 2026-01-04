/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../../../tests/catch2/catch.hpp"

#include "../../MagAOXApp.hpp"
#include "../dmPokeWFS.hpp"
#include "../dssShutter.hpp"
#include "../edtCamera.hpp"
#include "../frameGrabber.hpp"
#include "../ioDevice.hpp"
#include "../semUtilsDerived.hpp"
#include "../shmimMonitor.hpp"
#include "../stdCamera.hpp"
#include "../stdMotionStage.hpp"

using namespace MagAOX::app;

namespace template_test
{

SCENARIO( "xxxx", "[template]" )
{
   GIVEN("xxxxx")
   {
      int rv;

      WHEN("xxxx")
      {
         rv = 0;

         REQUIRE(rv == 0);
      }
   }
}
} //namespace template_test
