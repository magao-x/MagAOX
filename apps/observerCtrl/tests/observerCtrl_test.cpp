/** \file observerCtrl_test.cpp
 * \brief Catch2 tests for the observerCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup observerCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../observerCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup observerCtrl_unit_test observerCtrl Unit Tests
 * \brief Unit tests for the observerCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `observerCtrl` unit tests.
/** \ingroup observerCtrl_unit_test
 */
namespace observerCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class observerCtrl_test : public observerCtrl
{
  public:
    observerCtrl_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( observers );
        XWCTEST_SETUP_INDI_NEW_PROP( obsName );
        XWCTEST_SETUP_INDI_NEW_PROP( observing );
        XWCTEST_SETUP_INDI_NEW_PROP( sws );
    }
};
/// \endcond

/// Verify the observerCtrl INDI callback validators accept only the expected properties.
/**
 * \ingroup observerCtrl_unit_test
 */
TEST_CASE( "observerCtrl INDI callbacks validate device and property names", "[observerCtrl]" )
{
    // clang-format off
    #ifdef OBSERVERCTRL_TEST_DOXYGEN_REF
    observerCtrl::newCallBack_m_indiP_observers( pcf::IndiProperty() );
    observerCtrl::newCallBack_m_indiP_obsName( pcf::IndiProperty() );
    observerCtrl::newCallBack_m_indiP_observing( pcf::IndiProperty() );
    observerCtrl::newCallBack_m_indiP_sws( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observers );
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, obsName );
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, observing );
    XWCTEST_INDI_NEW_CALLBACK( observerCtrl, sws );
}

} // namespace observerCtrlTest

} // namespace libXWCTest
