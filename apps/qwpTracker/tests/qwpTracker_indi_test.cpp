/** \file qwpTracker_indi_test.cpp
 * \brief Catch2 tests for the qwpTracker INDI callback validation.
 *
 * Kept separate from qwpTracker_test.cpp because testMacrosINDI.hpp makes every validated callback return early.
 *
 * \ingroup qwpTracker_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../qwpTracker.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{
namespace qwpTrackerTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class qwpTracker_test : public qwpTracker
{
  public:
    qwpTracker_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( tracking );
        XWCTEST_SETUP_INDI_NEW_PROP( qwp1 );
        XWCTEST_SETUP_INDI_NEW_PROP( qwp2 );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_stagek, stagek, position );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp1AbsDeg, stageqwprot1, absDeg );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp1Fsm, stageqwprot1, fsm );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp2AbsDeg, stageqwprot2, absDeg );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp2Fsm, stageqwprot2, fsm );
    }
};
/// \endcond

/// Verify the qwpTracker INDI callback validators accept only the expected properties.
/**
 * \ingroup qwpTracker_unit_test
 */
TEST_CASE( "qwpTracker INDI callbacks validate device and property names", "[qwpTracker]" )
{
    // clang-format off
    #ifdef QWPTRACKER_TEST_DOXYGEN_REF
    qwpTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty() );
    qwpTracker::newCallBack_m_indiP_qwp1( pcf::IndiProperty() );
    qwpTracker::newCallBack_m_indiP_qwp2( pcf::IndiProperty() );
    qwpTracker::setCallBack_m_indiP_stagek( pcf::IndiProperty() );
    qwpTracker::setCallBack_m_indiP_qwp1AbsDeg( pcf::IndiProperty() );
    qwpTracker::setCallBack_m_indiP_qwp1Fsm( pcf::IndiProperty() );
    qwpTracker::setCallBack_m_indiP_qwp2AbsDeg( pcf::IndiProperty() );
    qwpTracker::setCallBack_m_indiP_qwp2Fsm( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( qwpTracker, tracking );
    XWCTEST_INDI_NEW_CALLBACK( qwpTracker, qwp1 );
    XWCTEST_INDI_NEW_CALLBACK( qwpTracker, qwp2 );
    XWCTEST_INDI_SET_CALLBACK( qwpTracker, m_indiP_stagek, stagek, position );
    XWCTEST_INDI_SET_CALLBACK( qwpTracker, m_indiP_qwp1AbsDeg, stageqwprot1, absDeg );
    XWCTEST_INDI_SET_CALLBACK( qwpTracker, m_indiP_qwp1Fsm, stageqwprot1, fsm );
    XWCTEST_INDI_SET_CALLBACK( qwpTracker, m_indiP_qwp2AbsDeg, stageqwprot2, absDeg );
    XWCTEST_INDI_SET_CALLBACK( qwpTracker, m_indiP_qwp2Fsm, stageqwprot2, fsm );
}

} // namespace qwpTrackerTest

} // namespace libXWCTest
