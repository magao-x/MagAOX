/** \file kcubeCtrl_test.cpp
  * \brief Catch2 tests for the kcubeCtrl app.
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../kcubeCtrl.hpp"

using namespace MagAOX::app;

namespace KCCTEST 
{

class kcubeCtrl_test : public kcubeCtrl 
{

public:
    kcubeCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(identify);
    }
};

SCENARIO( "INDI Callbacks", "[kcubeCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( kcubeCtrl, identify);
}

/*
SCENARIO( "xxxx", "[kcubeCtrl]" )
{
   GIVEN("xxxxx")
   {
      int rv;

      WHEN("xxxx")
      {
         rv = [some test];

         REQUIRE(rv == 0);
      }
   }
}*/

} //namespace KCCTEST
