/** \file trippLitePDU_sim_test.cpp
 * \brief Offline execution of the actual Tripp Lite simulator branches.
 * \ingroup trippLitePDU_files
 */
#include "../../../tests/outletAppTest.hpp"
#ifndef XWC_SIM_MODE
#define XWC_SIM_MODE
#endif
#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#include "../trippLitePDU.hpp"
#undef telemeter
#undef MagAOXApp
#undef XWC_SIM_MODE
namespace libXWCTest
{
/** \addtogroup trippLitePDU_unit_test
 * @{ */
namespace trippLitePDUTest
{
/// Verify that production simulator dispatch observes state and propagates control errors.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU simulator production dispatch", "[trippLitePDU][simulator]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    MagAOX::app::trippLitePDU::appStartup(); MagAOX::app::trippLitePDU::devConnect();
    MagAOX::app::trippLitePDU::devLogin(); MagAOX::app::trippLitePDU::devPostLogin();
    MagAOX::app::trippLitePDU::devStatus(); MagAOX::app::trippLitePDU::turnOutletOn(); MagAOX::app::trippLitePDU::turnOutletOff();
    #endif
    // clang-format on
    outletHarness::g_faults={};
    outletHarness::Controller<MagAOX::app::trippLitePDU> app;
    app.configText("[camera]\noutlet=1\n"); app.loadConfig(); REQUIRE(!app.m_shutdown);
    REQUIRE(app.appStartup()==0); REQUIRE(app.appLogic()==0);
    REQUIRE(app.state()==MagAOX::app::stateCodes::READY);
    REQUIRE(app.outletState(0)==OUTLET_STATE_OFF);
    REQUIRE(app.turnOutletOn(0)==0); REQUIRE(app.appLogic()==0);
    REQUIRE(app.outletState(0)==OUTLET_STATE_ON);
    REQUIRE(app.turnOutletOff(0)==0); REQUIRE(app.appLogic()==0);
    REQUIRE(app.outletState(0)==OUTLET_STATE_OFF);
    REQUIRE(app.turnOutletOn(8)<0); REQUIRE(app.turnOutletOff(8)<0);
    REQUIRE(app.appShutdown()==0);
}
} // namespace trippLitePDUTest
///@}
} // namespace libXWCTest
