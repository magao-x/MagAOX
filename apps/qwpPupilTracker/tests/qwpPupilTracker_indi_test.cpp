/** \file qwpPupilTracker_indi_test.cpp
 * \brief Catch2 tests for the qwpPupilTracker INDI callback validation.
 *
 * Kept separate from qwpPupilTracker_test.cpp because testMacrosINDI.hpp makes every validated callback return early.
 *
 * \ingroup qwpPupilTracker_files
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../qwpPupilTracker.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{
namespace qwpPupilTrackerTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
class qwpPupilTracker_test : public qwpPupilTracker
{
  public:
    qwpPupilTracker_test( const std::string device )
    {
        m_configName = device;

        XWCTEST_SETUP_INDI_NEW_PROP( tracking );

        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp1Position, stageqwprot1, position );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp1Fsm, stageqwprot1, fsm );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp2Position, stageqwprot2, position );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_qwp2Fsm, stageqwprot2, fsm );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ttmPos1, ttmperi, pos_1 );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ttmPos2, ttmperi, pos_2 );
        XWCTEST_SETUP_INDI_ARB_PROP( m_indiP_ttmFsm, ttmperi, fsm );
    }
};
/// \endcond

/// Verify the qwpPupilTracker INDI callback validators accept only the expected properties.
/**
 * \ingroup qwpPupilTracker_unit_test
 */
TEST_CASE( "qwpPupilTracker INDI callbacks validate device and property names", "[qwpPupilTracker]" )
{
    // clang-format off
    #ifdef QWPPUPILTRACKER_TEST_DOXYGEN_REF
    qwpPupilTracker::newCallBack_m_indiP_tracking( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_qwp1Position( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_qwp1Fsm( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_qwp2Position( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_qwp2Fsm( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_ttmPos1( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_ttmPos2( pcf::IndiProperty() );
    qwpPupilTracker::setCallBack_m_indiP_ttmFsm( pcf::IndiProperty() );
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( qwpPupilTracker, tracking );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_qwp1Position, stageqwprot1, position );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_qwp1Fsm, stageqwprot1, fsm );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_qwp2Position, stageqwprot2, position );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_qwp2Fsm, stageqwprot2, fsm );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_ttmPos1, ttmperi, pos_1 );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_ttmPos2, ttmperi, pos_2 );
    XWCTEST_INDI_SET_CALLBACK( qwpPupilTracker, m_indiP_ttmFsm, ttmperi, fsm );
}

} // namespace qwpPupilTrackerTest

} // namespace libXWCTest
