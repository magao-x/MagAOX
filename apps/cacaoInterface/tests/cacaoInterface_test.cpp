/** \file cacaoInterface_test.cpp
  * \brief Catch2 tests for the cacaoInterface app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../cacaoInterface.hpp"

using namespace MagAOX::app;

namespace CACAOITEST
{

class cacaoInterface_test : public cacaoInterface 
{

public:
    cacaoInterface_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(loopState);
        XWCTEST_SETUP_INDI_NEW_PROP(loopGain);
        XWCTEST_SETUP_INDI_NEW_PROP(loopZero);
        XWCTEST_SETUP_INDI_NEW_PROP(multCoeff);
        XWCTEST_SETUP_INDI_NEW_PROP(maxLim);
        
    }
};


SCENARIO( "INDI Callbacks", "[cacaoInterface]" )
{
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopState);
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopGain);
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopZero);
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, multCoeff);
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, maxLim);

}


} //namespace cacaoInterface_test 
