/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../tests/catch2/catch.hpp"

#include "../INDI/connect_to.h"
#include "../INDI/fq.h"
#include "../INDI/indiapi.h"
#include "../libcommon/Cmd.hpp"
#include "../libcommon/IndiClient.hpp"
#include "../libcommon/IndiConnection.hpp"
#include "../libcommon/IndiDriver.hpp"
#include "../libcommon/IndiElement.hpp"
#include "../libcommon/IndiMessage.hpp"
#include "../libcommon/IndiProperty.hpp"
#include "../libcommon/IndiPropertyMap.hpp"
#include "../libcommon/MutexLock.hpp"
#include "../libcommon/ReadWriteLock.hpp"
#include "../libcommon/System.hpp"
#include "../libcommon/SystemSocket.hpp"
#include "../libcommon/Thread.hpp"
#include "../libcommon/TimeStamp.hpp"
#include "../liblilxml/base64.h"
#include "../liblilxml/lilxml.h"


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
