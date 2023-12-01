/** \file adcTracker_test.cpp
  * \brief Catch2 tests for the adcTracker app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */



#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../adcTracker.hpp"

using namespace MagAOX::app;

namespace ADCTTEST
{

class adcTracker_test : public adcTracker 
{

public:
    adcTracker_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(tracking);
        XWCTEST_SETUP_INDI_NEW_PROP(deltaAngle);
        XWCTEST_SETUP_INDI_NEW_PROP(deltaADC1);
        XWCTEST_SETUP_INDI_NEW_PROP(deltaADC2);
        XWCTEST_SETUP_INDI_NEW_PROP(minZD);

        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_teldata, tcsi, zd);
    }
};


SCENARIO( "INDI Callbacks", "[adcTracker]" )
{
    XWCTEST_INDI_NEW_CALLBACK( adcTracker, tracking);
    XWCTEST_INDI_NEW_CALLBACK( adcTracker, deltaAngle);
    XWCTEST_INDI_NEW_CALLBACK( adcTracker, deltaADC1);
    XWCTEST_INDI_NEW_CALLBACK( adcTracker, deltaADC2);
    XWCTEST_INDI_NEW_CALLBACK( adcTracker, minZD);

    XWCTEST_INDI_SET_CALLBACK( adcTracker, m_indiP_teldata, tcsi, zd);

}


} //namespace adcTracker_test 
