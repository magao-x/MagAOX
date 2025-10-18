/** \file observerCtrl_test.cpp
  * \brief Catch2 tests for the observerCtrl app.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  */


/** \defgroup observerCtrl_tests
 *  \brief Tests of the observerCtrl app
 *  \ingroup app_test
 *
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
        XWCTEST_SETUP_INDI_NEW_PROP(operators);
        XWCTEST_SETUP_INDI_NEW_PROP(obsName);
        XWCTEST_SETUP_INDI_NEW_PROP(observing);
        XWCTEST_SETUP_INDI_NEW_PROP(obsDuration);
        XWCTEST_SETUP_INDI_NEW_PROP(sws);
        XWCTEST_SETUP_INDI_NEW_PROP(userlog);
        XWCTEST_SETUP_INDI_NEW_PROP(resetTarget);
        XWCTEST_SETUP_INDI_NEW_PROP(target);
        XWCTEST_SETUP_INDI_NEW_PROP(tcsTarget);

        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_catalog, tcsi, catalog);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_teldata, tcsi, teldata);
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_labMode, tcsi, labMode);

    }
};


/// observerCtrl INDI Callback Input Validation
/**
 * \ingroup observerCtrl_tests
 */
TEST_CASE( "observerCtrl INDI Callback Input Validation", "[observerCtrl]" )
{
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observers);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, operators);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, obsName);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observing);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, sws);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, userlog);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, resetTarget);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, target);
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, tcsTarget);
    XWCTEST_INDI_SET_CALLBACK( observerCtrl, m_indiP_catalog, tcsi, catalog);
    XWCTEST_INDI_SET_CALLBACK( observerCtrl, m_indiP_teldata, tcsi, teldata);
    XWCTEST_INDI_SET_CALLBACK( observerCtrl, m_indiP_labMode, tcsi, labMode);

}



} //namespace observerCtrl_test
