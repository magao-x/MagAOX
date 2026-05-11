/** \file smc100ccCtrl_test.cpp
 * \brief Catch2 tests for the smc100ccCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup smc100ccCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../smc100ccCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup smc100ccCtrl_unit_test smc100ccCtrl Unit Tests
 * \brief Unit tests for the smc100ccCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `smc100ccCtrl` unit tests.
/** \ingroup smc100ccCtrl_unit_test
 */
namespace smc100ccCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class smc100ccCtrl_test : public smc100ccCtrl
{
  public:
    smc100ccCtrl_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( position );

        XWCTEST_SETUP_INDI_NEW_PROP( preset );
        XWCTEST_SETUP_INDI_NEW_PROP( presetName );
        XWCTEST_SETUP_INDI_NEW_PROP( home );
        XWCTEST_SETUP_INDI_NEW_PROP( stop );
    }
};
/// \endcond

/// Verify the smc100ccCtrl INDI callback validators accept only the expected properties.
/**
 * \ingroup smc100ccCtrl_unit_test
 */
TEST_CASE( "smc100ccCtrl INDI callbacks validate device and property names", "[smc100ccCtrl]" )
{
    // clang-format off
    #ifdef SMC100CCCTRL_TEST_DOXYGEN_REF
    smc100ccCtrl::newCallBack_m_indiP_position( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_preset( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_presetName( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_home( pcf::IndiProperty() );
    smc100ccCtrl::newCallBack_m_indiP_stop( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, position );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, preset );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, presetName );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, home );
    XWCTEST_INDI_NEW_CALLBACK( smc100ccCtrl, stop );
}

} // namespace smc100ccCtrlTest

} // namespace libXWCTest
