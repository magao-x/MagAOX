/** \file template_test.cpp
  * \brief Catch2 tests for the template app.
  *
  * History:
  */
#include "../../../tests/catch2/catch.hpp"
#include "../../tests/testMacrosINDI.hpp"

#include "../closedLoopIndi.hpp"

using namespace MagAOX::app;

namespace closedLoopIndi_test
{

class closedLoopIndi_test : public closedLoopIndi 
{

public:
    closedLoopIndi_test(const std::string device)
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP(reference0);
        XWCTEST_SETUP_INDI_NEW_PROP(reference1);
        XWCTEST_SETUP_INDI_NEW_PROP(ggain);
        XWCTEST_SETUP_INDI_NEW_PROP(ctrlEnabled);

        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_inputs, "inputdev", "measurement" )
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_ctrl0, "ctrl0dev", "prop0" )
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_ctrl1, "ctrl1dev", "prop1" )
        XWCTEST_SETUP_INDI_ARB_PROP(m_indiP_upstream, "updev", "loop_state" )
    }
};



SCENARIO( "INDI Callbacks", "[closedLoopIndi]" )
{
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, reference0);
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, reference1);
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, ggain);
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, ctrlEnabled);
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_inputs, "inputdev", "measurement")
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_ctrl0, "ctrl0dev", "prop0")
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_ctrl1, "ctrl1dev", "prop1")
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_upstream, "updev", "loop_state")
}


} //namespace closedLoopIndi_test 
