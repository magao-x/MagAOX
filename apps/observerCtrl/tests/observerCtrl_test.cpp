/** \file observerCtrl_test.cpp
  * \brief Catch2 tests for the observerCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../observerCtrl.hpp"

using namespace MagAOX::app;

namespace SMCTEST
{

class observerCtrl_test : public observerCtrl 
{

public:
    observerCtrl_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(observers);
        XWCTEST_SETUP_INDI_NEW_PROP(obsName);
        XWCTEST_SETUP_INDI_NEW_PROP(observing);
        XWCTEST_SETUP_INDI_NEW_PROP(sws);


    }
};


SCENARIO( "INDI Callbacks", "[observerCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observers);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, obsName);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observing);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, sws);


}


} //namespace observerCtrl_test 
