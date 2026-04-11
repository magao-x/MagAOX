/** \file kTracker_test.cpp
 * \brief Catch2 tests for the kTracker app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * History:
 */

#include "../../../tests/catch2/catch.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../kTracker.hpp"

using namespace MagAOX::app;

namespace KTRACKERTEST
{

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

SCENARIO( "INDI Callbacks", "[kTracker]" )
{
    XWCTEST_INDI_NEW_CALLBACK( kTracker, tracking );

    XWCTEST_INDI_SET_CALLBACK( kTracker, m_indiP_teldata, tcsi, zd );
}

} // namespace KTRACKERTEST
