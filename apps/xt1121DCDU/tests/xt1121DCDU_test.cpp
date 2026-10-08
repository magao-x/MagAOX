/** \file xt1121DCDU_test.cpp
 * \brief Offline configuration, FSM, numeric INDI traffic, and telemetry tests for the DCDU.
 * \ingroup xt1121DCDU_files
 */
#include "../../../tests/outletAppTest.hpp"
#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#include "../xt1121DCDU.hpp"
#undef telemeter
#undef MagAOXApp

namespace libXWCTest
{
/** \defgroup xt1121DCDU_unit_test xt1121DCDU Unit Tests
 * \ingroup application_unit_test
 */
namespace xt1121DCDUTest
{
using namespace MagAOX::app;
using namespace outletHarness;
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose mapping state while preserving the real app behavior.
struct Fixture : Controller<xt1121DCDU>
{
    /// Access the configured source name.
    using xt1121DCDU::m_deviceName;
    /// Access configured channel numbers.
    using xt1121DCDU::m_channelNumbers;
    /// Access the mapping helpers actually under test.
    using xt1121DCDU::xtChannelName;
    using xt1121DCDU::xtChannelProperty;
    /// Expose the shared source-update failure contract.
    using xt1121DCDU::processSourceUpdate;
    /// Inject a failure from the inherited all-outlet update.
    bool m_failUpdate {false};
    /// Preserve real updates unless their result is selected to fail.
    int updateOutletStates() override;
    /// Load a default-backed single channel.
    void configure();
};
int Fixture::updateOutletStates()
{
    if(m_failUpdate) return -1;
    return dev::outletController<xt1121DCDU>::updateOutletStates();
}
void Fixture::configure()
{
    configText("[device]\nname=xt\n[channel]\noutlets=1,2\n");
    loadConfig();
    REQUIRE(!m_shutdown);
}
/// \endcond

/// Verify real configuration defaults, overrides, mapping, and fatal load errors.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("DCDU configuration and outlet/channel mappings", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::xt1121DCDU(); xt1121DCDU::~xt1121DCDU(); xt1121DCDU::setupConfig();
    xt1121DCDU::loadConfig(); xt1121DCDU::loadConfigImpl(); xt1121DCDU::xtChannelName(); xt1121DCDU::xtChannelProperty();
    #endif
    // clang-format on
    g_faults={};
    Fixture app;
    app.configure();
    REQUIRE(app.m_deviceName=="xt");
    REQUIRE(app.m_channelNumbers==std::vector<int>{0,1,2,3,4,5,6,7});
    REQUIRE(app.channelOutlets("channel")==std::vector<size_t>{0,1});
    for(int number=0;number<=16;++number)
        REQUIRE(app.xtChannelName(number)==std::format("ch{:02}",number));
    REQUIRE(app.xtChannelName(-1).empty());
    REQUIRE(app.xtChannelName(17).empty());
    for(int number=0;number<8;++number) REQUIRE(app.xtChannelProperty(number)!=nullptr);
    REQUIRE(app.xtChannelProperty(-1)==nullptr);
    REQUIRE(app.xtChannelProperty(8)==nullptr);
    SECTION("explicit source channels")
    {
        app.m_channelNumbers={8,9,10,11,12,13,14,16};
        REQUIRE(app.appStartup()==0);
        REQUIRE(app.m_indiSetCallBacks.count("xt.ch16")==1);
    }
}

/// Propagate app/helper configuration and startup failures without starting hardware access.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("DCDU configuration and lifecycle failure contracts", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::loadConfig(); xt1121DCDU::loadConfigImpl(); xt1121DCDU::appStartup(); xt1121DCDU::appShutdown();
    #endif
    // clang-format on
    for(size_t call : {0,1,2,4})
    {
        g_faults={}; Fixture app;
        g_faults.m_telemResults[call]=-1;
        app.configText("[device]\nname=xt\n[channel]\noutlet=1\n");
        if(call==0) REQUIRE(app.m_shutdown);
        else if(call==1) { app.loadConfig(); REQUIRE(app.m_shutdown); }
        else
        {
            REQUIRE(app.loadConfigImpl(app.config)==0);
            if(call==2) REQUIRE(app.appStartup()<0);
            else REQUIRE(app.appShutdown()==0);
        }
    }
    for(unsigned registration=1;registration<=14;++registration)
    {
        g_faults={}; Fixture app; app.configure();
        g_faults.m_failRegistration=registration;
        REQUIRE(app.appStartup()<0);
    }
    g_faults={};
    Fixture app;
    app.configText("[device]\nname=xt\nchannelNumbers=0,1\n");
    app.loadConfig();
    REQUIRE(app.m_shutdown);
    app.m_channelNumbers={0,1};
    REQUIRE(app.appStartup()<0);
}

/// Run each registered callback and verify numeric On/Off traffic with actual XML.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("DCDU callbacks observe numeric state and issue correct commands", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::processSourceUpdate(); xt1121DCDU::updateOutletState(); xt1121DCDU::turnOutletOn(); xt1121DCDU::turnOutletOff();
    xt1121DCDU::setCallBack_m_indiP_ch0(); xt1121DCDU::setCallBack_m_indiP_ch1();
    xt1121DCDU::setCallBack_m_indiP_ch2(); xt1121DCDU::setCallBack_m_indiP_ch3();
    xt1121DCDU::setCallBack_m_indiP_ch4(); xt1121DCDU::setCallBack_m_indiP_ch5();
    xt1121DCDU::setCallBack_m_indiP_ch6(); xt1121DCDU::setCallBack_m_indiP_ch7();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure(); REQUIRE(app.appStartup()==0); app.driver();
    for(int number=0;number<8;++number)
    {
        for(int value : {0,1,2})
        {
            pcf::IndiProperty observation(pcf::IndiProperty::Number,"xt",app.xtChannelName(number));
            observation.add(pcf::IndiElement("current",value));
            observation.add(pcf::IndiElement("target",value));
            app.handleDefProperty(observation);
            REQUIRE(app.outletState(number)==(value==0 ? OUTLET_STATE_OFF : value==1 ? OUTLET_STATE_ON : OUTLET_STATE_UNKNOWN));
        }
        REQUIRE(app.turnOutletOn(number)==0);
        REQUIRE(app.turnOutletOff(number)==0);
    }
    unsigned commands=0;
    for(auto &message : app.messages())
    {
        if(message.getType()!=pcf::IndiMessage::NewProperty) continue;
        auto command=message.getProperty();
        REQUIRE(command.getDevice()=="xt");
        REQUIRE(command.getName()==std::format("ch{:02}",commands/2));
        REQUIRE(command.getType()==pcf::IndiProperty::Number);
        REQUIRE(command["target"].get<int>()==(commands%2==0 ? 1 : 0));
        ++commands;
    }
    REQUIRE(commands==16);
    *app.xtChannelProperty(0)=pcf::IndiProperty(pcf::IndiProperty::Number,"xt","ch00");
    REQUIRE(app.updateOutletState(0)==0);
    REQUIRE(app.outletState(0)==OUTLET_STATE_UNKNOWN);
    REQUIRE(app.updateOutletState(-1)<0);
    REQUIRE(app.processSourceUpdate(*app.xtChannelProperty(0),property("xt","ch00","current","0"),-1)<0);
    REQUIRE(app.turnOutletOn(-1)<0);
    REQUIRE(app.turnOutletOff(8)<0);
    g_faults.m_failSend=g_faults.m_sends+1;
    REQUIRE(app.turnOutletOn(1)<0);
    g_faults.m_failSend=g_faults.m_sends+1;
    REQUIRE(app.turnOutletOff(1)<0);
}

/// Exercise the real FSM, contention, observed telemetry, interval recording, and error returns.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("DCDU FSM and telemetry lifecycle", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::appLogic(); xt1121DCDU::checkRecordTimes(); xt1121DCDU::recordTelem(); xt1121DCDU::appShutdown();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure(); REQUIRE(app.appStartup()==0);
    app.state(stateCodes::POWERON);
    REQUIRE(app.appLogic()==0);
    REQUIRE(app.state()==stateCodes::READY);
    REQUIRE(g_faults.m_records.size()==1);
    REQUIRE(app.appLogic()==0);
    REQUIRE(g_faults.m_records.size()==1);
    g_faults.m_due=true;
    REQUIRE(app.appLogic()==0);
    REQUIRE(g_faults.m_records.size()==2);
    g_faults.m_due=false;
    g_faults.m_recordResult=-1;
    REQUIRE(app.recordTelem(static_cast<const MagAOX::logger::telem_outlet *>(nullptr))<0);
    app.m_outletTelemRecorded=false;
    REQUIRE(app.appLogic()<0);
    g_faults.m_recordResult=0;
    g_faults.m_telemResults[3]=-1;
    REQUIRE(app.appLogic()<0);
    g_faults.m_telemResults[3]=0;
    app.m_failUpdate=true;
    REQUIRE(app.appLogic()<0);
    app.m_failUpdate=false;
    contended(app.m_indiMutex,[&] {REQUIRE(app.appLogic()==0);});
    app.state(stateCodes::NOTCONNECTED);
    REQUIRE(app.appLogic()<0);
    REQUIRE(app.state()==stateCodes::FAILURE);
    REQUIRE(app.appShutdown()==0);
}
/// Keep observed-state validity and periodic recording correct across power-off.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("DCDU power-off invalidates observations and retains telemetry", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::onPowerOff(); xt1121DCDU::whilePowerOff();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure(); REQUIRE(app.appStartup()==0);
    app.setAllOutletStates(OUTLET_STATE_ON);
    REQUIRE(app.onPowerOff()==0);
    for(int n=0;n<8;++n) REQUIRE(app.outletState(n)==OUTLET_STATE_UNKNOWN);
    g_faults.m_due=true;
    REQUIRE(app.whilePowerOff()==0);
    REQUIRE(g_faults.m_records.size()==2);
    g_faults.m_telemResults[3]=-1;
    REQUIRE(app.whilePowerOff()<0);
    REQUIRE(g_faults.m_sends==0);
}
/// Reject the one-past-end physical outlet configuration before control or telemetry can index it.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("xt1121DCDU rejects a one-past-end configured outlet", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::loadConfig(); xt1121DCDU::loadConfigImpl();
    #endif
    // clang-format on
    outletHarness::g_faults={};
    Fixture app;
    app.configText("[invalid]\noutlet=9\n");
    app.loadConfig(); REQUIRE(app.m_shutdown);
}
/// Load explicit source-channel mapping through the real configurator.
/** \ingroup xt1121DCDU_unit_test */
TEST_CASE("DCDU loads explicit source channel numbers", "[xt1121DCDU]")
{
    // clang-format off
    #ifdef XT1121DCDU_TEST_DOXYGEN_REF
    xt1121DCDU::loadConfigImpl(); xt1121DCDU::appStartup();
    #endif
    // clang-format on
    g_faults={}; Fixture app;
    app.configText("[device]\nname=other-xt\nchannelNumbers=8,9,10,11,12,13,14,16\n[x]\noutlet=1\n");
    app.loadConfig(); REQUIRE(!app.m_shutdown);
    REQUIRE(app.m_deviceName=="other-xt");
    REQUIRE(app.m_channelNumbers==std::vector<int>{8,9,10,11,12,13,14,16});
    REQUIRE(app.appStartup()==0);
    REQUIRE(app.m_indiSetCallBacks.count("other-xt.ch16")==1);
}
} // namespace xt1121DCDUTest
} // namespace libXWCTest
