/** \file kTracker_test.cpp
 * \brief Catch2 tests for the kTracker app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup kTracker_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../kTracker.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup kTracker_unit_test kTracker Unit Tests
 * \brief Unit tests for the kTracker application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `kTracker` unit tests.
/** \ingroup kTracker_unit_test
 */
namespace kTrackerTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class kTracker_test : public kTracker
{
  public:
    kTracker_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( tracking );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_teldata, tcsi, zd );
    }
};
/// \endcond

/// Verify the kTracker INDI callback validators accept only the expected properties.
/**
 * \ingroup kTracker_unit_test
 */
TEST_CASE( "kTracker INDI callbacks validate device and property names", "[kTracker]" )
{
    // clang-format off
    #ifdef KTRACKER_TEST_DOXYGEN_REF
    kTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty() );
    kTracker::setCallBack_m_indiP_teldata( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( kTracker, tracking );
    XWCTEST_INDI_SET_CALLBACK( kTracker, m_indiP_teldata, tcsi, zd );
}

} // namespace kTrackerTest

} // namespace libXWCTest
