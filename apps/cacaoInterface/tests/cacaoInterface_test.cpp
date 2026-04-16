/** \file cacaoInterface_test.cpp
 * \brief Catch2 tests for the cacaoInterface app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cacaoInterface_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../cacaoInterface.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup cacaoInterface_unit_test cacaoInterface Unit Tests
 * \brief Unit tests for the cacaoInterface application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `cacaoInterface` unit tests.
/** \ingroup cacaoInterface_unit_test
 */
namespace cacaoInterfaceTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class cacaoInterface_test : public cacaoInterface
{
  public:
    cacaoInterface_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( loopState );
        XWCTEST_SETUP_INDI_NEW_PROP( loopGain );
        XWCTEST_SETUP_INDI_NEW_PROP( loopZero );
        XWCTEST_SETUP_INDI_NEW_PROP( multCoeff );
        XWCTEST_SETUP_INDI_NEW_PROP( maxLim );
    }
};
/// \endcond

/// Verify the cacaoInterface INDI callback validators accept only the expected properties.
/**
 * \ingroup cacaoInterface_unit_test
 */
TEST_CASE( "cacaoInterface INDI callbacks validate device and property names", "[cacaoInterface]" )
{
    // clang-format off
    #ifdef CACAOINTERFACE_TEST_DOXYGEN_REF
    cacaoInterface::newCallBack_m_indiP_loopState( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_loopGain( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_loopZero( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_multCoeff( pcf::IndiProperty() );
    cacaoInterface::newCallBack_m_indiP_maxLim( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopState );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopGain );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, loopZero );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, multCoeff );
    XWCTEST_INDI_NEW_CALLBACK( cacaoInterface, maxLim );
}

} // namespace cacaoInterfaceTest

} // namespace libXWCTest
