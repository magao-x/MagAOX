/** \file closedLoopIndi_test.cpp
 * \brief Catch2 tests for the closedLoopIndi app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup closedLoopIndi_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../closedLoopIndi.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup closedLoopIndi_unit_test closedLoopIndi Unit Tests
 * \brief Unit tests for the closedLoopIndi application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `closedLoopIndi` unit tests.
/** \ingroup closedLoopIndi_unit_test
 */
namespace closedLoopIndiTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class closedLoopIndi_test : public closedLoopIndi
{
  public:
    closedLoopIndi_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( reference0 );
        XWCTEST_SETUP_INDI_NEW_PROP( reference1 );
        XWCTEST_SETUP_INDI_NEW_PROP( ggain );
        XWCTEST_SETUP_INDI_NEW_PROP( ctrlEnabled );
        XWCTEST_SETUP_INDI_NEW_PROP( counterReset );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_inputs, inputdev, measurement )
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ctrl0_fsm, ctrl0dev, fsm )
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ctrl0, ctrl0dev, prop0 )
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ctrl1_fsm, ctrl1dev, fsm )
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ctrl1, ctrl1dev, prop1 )
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_upstream, updev, loop_state )
    }
};
/// \endcond

/// Verify the closedLoopIndi INDI callback validators accept only the expected properties.
/**
 * \ingroup closedLoopIndi_unit_test
 */
TEST_CASE( "closedLoopIndi INDI callbacks validate device and property names", "[closedLoopIndi]" )
{
    // clang-format off
    #ifdef CLOSEDLOOPINDI_TEST_DOXYGEN_REF
    closedLoopIndi::newCallBack_m_indiP_reference0( pcf::IndiProperty() );
    closedLoopIndi::newCallBack_m_indiP_reference1( pcf::IndiProperty() );
    closedLoopIndi::newCallBack_m_indiP_ggain( pcf::IndiProperty() );
    closedLoopIndi::newCallBack_m_indiP_ctrlEnabled( pcf::IndiProperty() );
    closedLoopIndi::newCallBack_m_indiP_counterReset( pcf::IndiProperty() );
    closedLoopIndi::setCallBack_m_indiP_inputs( pcf::IndiProperty() );
    closedLoopIndi::setCallBack_m_indiP_ctrl0_fsm( pcf::IndiProperty() );
    closedLoopIndi::setCallBack_m_indiP_ctrl0( pcf::IndiProperty() );
    closedLoopIndi::setCallBack_m_indiP_ctrl1_fsm( pcf::IndiProperty() );
    closedLoopIndi::setCallBack_m_indiP_ctrl1( pcf::IndiProperty() );
    closedLoopIndi::setCallBack_m_indiP_upstream( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, reference0 );
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, reference1 );
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, ggain );
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, ctrlEnabled );
    XWCTEST_INDI_NEW_CALLBACK( closedLoopIndi, counterReset );
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_inputs, inputdev, measurement )
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_ctrl0_fsm, ctrl0dev, fsm )
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_ctrl0, ctrl0dev, prop0 )
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_ctrl1_fsm, ctrl1dev, fsm )
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_ctrl1, ctrl1dev, prop1 )
    XWCTEST_INDI_SET_CALLBACK( closedLoopIndi, m_indiP_upstream, updev, loop_state )
}

} // namespace closedLoopIndiTest

} // namespace libXWCTest
