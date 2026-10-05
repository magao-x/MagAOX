/** \file kcubeCtrl_test.cpp
 * \brief Catch2 tests for the kcubeCtrl app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup kcubeCtrl_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../kcubeCtrl.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup kcubeCtrl_unit_test kcubeCtrl Unit Tests
 * \brief Unit tests for the kcubeCtrl application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `kcubeCtrl` unit tests.
/** \ingroup kcubeCtrl_unit_test
 */
namespace kcubeCtrlTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class kcubeCtrl_test : public kcubeCtrl
{
  public:
    kcubeCtrl_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( identify );
    }
};
/// \endcond

/// Verify the kcubeCtrl INDI callback validators accept only the expected properties.
/**
 * \ingroup kcubeCtrl_unit_test
 */
TEST_CASE( "kcubeCtrl INDI callbacks validate device and property names", "[kcubeCtrl]" )
{
    // clang-format off
    #ifdef KCUBECTRL_TEST_DOXYGEN_REF
    kcubeCtrl::newCallBack_m_indiP_identify( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( kcubeCtrl, identify );
}

} // namespace kcubeCtrlTest

} // namespace libXWCTest
