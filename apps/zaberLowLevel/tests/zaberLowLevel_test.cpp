/** \file zaberLowLevel_test.cpp
  * \brief Catch2 tests for the zaberLowLevel app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */


//Direct include to avoid having to link separately
extern "C"
{
    #include "../za_serial.c"
}

#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../zaberLowLevel.hpp"

using namespace MagAOX::app;

namespace ZLLTEST
{

class zaberLowLevel_test : public zaberLowLevel 
{

public:
    zaberLowLevel_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_PROP(tgt_pos)
        XWCTEST_SETUP_INDI_PROP(tgt_relpos)
        XWCTEST_SETUP_INDI_PROP(req_home)
        XWCTEST_SETUP_INDI_PROP(req_halt)
        XWCTEST_SETUP_INDI_PROP(req_ehalt)

    }
};

//#define QUOTE(s) #s


SCENARIO( "INDI Callbacks", "[zaberLowLevel]" )
{
    XWCTEST_INDI_CALLBACK( zaberLowLevel, tgt_pos);
    XWCTEST_INDI_CALLBACK( zaberLowLevel, tgt_relpos);
    XWCTEST_INDI_CALLBACK( zaberLowLevel, req_home);
    XWCTEST_INDI_CALLBACK( zaberLowLevel, req_halt);
    XWCTEST_INDI_CALLBACK( zaberLowLevel, req_ehalt);




}


} //namespace zaberLowLevel_test 
