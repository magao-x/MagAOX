/** \file trippLitePDU_test.cpp
 * \brief Production-path offline PDU parsing, transport, FSM, alarm, and telemetry tests.
 * \ingroup trippLitePDU_files
 */
#include "../../../tests/outletAppTest.hpp"
#ifdef XWC_SIM_MODE
#undef XWC_SIM_MODE
#endif
#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#define telnetConn outletTestTelnet
#define ioDevice outletTestIODevice
#include "../trippLitePDU.hpp"
#undef ioDevice
#undef telnetConn
#undef telemeter
#undef MagAOXApp

namespace libXWCTest
{
/** \defgroup trippLitePDU_unit_test trippLitePDU Unit Tests
 * \ingroup application_unit_test
 */
namespace trippLitePDUTest
{
using namespace MagAOX::app;
using namespace outletHarness;
/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Expose measured/configured state while retaining actual production methods.
struct Fixture : Controller<trippLitePDU>
{
    /// Access electrical measurements for alarm threshold assertions.
    using trippLitePDU::m_frequency;
    using trippLitePDU::m_voltage;
    using trippLitePDU::m_current;
    /// Access protocol configuration and its scripted transport.
    using trippLitePDU::m_deviceVersion;
    using trippLitePDU::m_telnetConn;
    using trippLitePDU::m_deviceAddr;
    using trippLitePDU::m_devicePort;
    using trippLitePDU::m_deviceUsername;
    using trippLitePDU::m_devicePassFile;
    /// Access complete-sample telemetry validity/cache for exact failure injection.
    using trippLitePDU::m_pduValid;
    using trippLitePDU::m_pduSample;
    using trippLitePDU::m_pduTelemRecorded;
    /// Inject an all-outlet update failure into appLogic.
    bool m_failUpdate {false};
    /// Preserve the real update unless its failure is selected.
    int updateOutletStates() override;
    /// Load the standard one-based physical outlet example.
    void configure();
};
int Fixture::updateOutletStates()
{
    if(m_failUpdate) return -1;
    return trippLitePDU::updateOutletStates();
}
void Fixture::configure()
{
    configText("[device]\naddress=test-address\nport=23\nusername=test-user\npassfile=test-secret\n"
               "[camera]\noutlets=1,8\n");
    loadConfig(); REQUIRE(!m_shutdown);
}
/// Form a complete, successful devstatus report with known outlets and measured values.
std::string status( float frequency=60 /**< [in] Hz */, float voltage=120 /**< [in] V */,
                    float current=4 /**< [in] A */ );
std::string status( float frequency, float voltage, float current )
{
    return std::format("header\nInput Voltage: {} V\nInput Frequency: {} Hz\nOutput Current: {} A\nOutlets On: 1 8\n",
                        voltage,frequency,current);
}
/// \endcond

/// Cover actual app/helper configuration, overrides, and every lifecycle failure.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU configuration and lifecycle failure contracts", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::trippLitePDU(); trippLitePDU::~trippLitePDU(); trippLitePDU::setupConfig();
    trippLitePDU::loadConfig(); trippLitePDU::loadConfigImpl(); trippLitePDU::appStartup(); trippLitePDU::appShutdown();
    #endif
    // clang-format on
    g_faults={};
    {
        Fixture app; app.configure();
        REQUIRE(app.m_deviceAddr=="test-address"); REQUIRE(app.m_devicePort=="23");
        REQUIRE(app.m_deviceUsername=="test-user"); REQUIRE(app.m_devicePassFile=="test-secret");
        REQUIRE(app.channelOutlets("camera")==std::vector<size_t>{0,7});
        REQUIRE(app.appStartup()==0); REQUIRE(app.state()==stateCodes::NOTCONNECTED);
        REQUIRE(app.appShutdown()==0);
    }
    for(unsigned registration=1;registration<=8;++registration)
    {
        g_faults={}; Fixture app; app.configure();
        g_faults.m_failRegistration=registration;
        REQUIRE(app.appStartup()<0);
    }
    for(size_t call : {0,1,2,4})
    {
        g_faults={}; Fixture app; g_faults.m_telemResults[call]=-1;
        app.configText("[camera]\noutlet=1\n");
        if(call==0) REQUIRE(app.m_shutdown);
        else if(call==1) {app.loadConfig(); REQUIRE(app.m_shutdown);}
        else
        {
            REQUIRE(app.loadConfigImpl(app.config)==0);
            if(call==2) REQUIRE(app.appStartup()<0);
            else REQUIRE(app.appShutdown()==0);
        }
    }
    g_faults={}; Fixture app; app.configText("[device]\naddress=test\n");
    app.loadConfig(); REQUIRE(app.m_shutdown);
}

/// Verify connection/login/version FSM outcomes without any network calls.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU production transport FSM", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::appLogic(); trippLitePDU::devConnect(); trippLitePDU::devLogin(); trippLitePDU::devPostLogin();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure(); REQUIRE(app.appStartup()==0);
    g_faults.m_status=status();
    g_faults.m_transportResults={-1}; errno=EIO;
    REQUIRE(app.appLogic()==0); REQUIRE(app.state()==stateCodes::NOTCONNECTED);
    g_faults.m_transportResults={-1}; REQUIRE(app.appLogic()==0);
    g_faults.m_transportResults={-2}; errno=ENOSPC; REQUIRE(app.appLogic()==0);
    g_faults.m_transportResults={0,TELNET_E_LOGINTIMEOUT};
    REQUIRE(app.appLogic()==0); REQUIRE(app.state()==stateCodes::NOTCONNECTED);
    g_faults.m_transportResults={0,-1};
    REQUIRE(app.appLogic()<0); REQUIRE(app.state()==stateCodes::FAILURE);
    app.state(stateCodes::NOTCONNECTED);
    REQUIRE(app.appLogic()==0); REQUIRE(app.state()==stateCodes::READY);
    REQUIRE(app.m_pduValid); REQUIRE(app.outletState(0)==OUTLET_STATE_ON);
    REQUIRE(g_faults.m_loginPrompt.empty());
    app.m_deviceVersion=1;
    app.state(stateCodes::NOTCONNECTED);
    REQUIRE(app.appLogic()==0); REQUIRE(app.state()==stateCodes::READY);
    REQUIRE(g_faults.m_loginPrompt=="login:");
    REQUIRE(app.m_telnetConn.m_prompt=="$> ");
    REQUIRE(std::find(g_faults.m_commands.begin(),g_faults.m_commands.end(),"E\n")!=g_faults.m_commands.end());
    app.state(stateCodes::CONNECTED); app.stateLogged();
    REQUIRE(app.appLogic()==0);
    app.m_failUpdate=true; REQUIRE(app.appLogic()<0); app.m_failUpdate=false;
    contended(app.m_indiMutex,[&] {REQUIRE(app.appLogic()==0);});
    app.state(stateCodes::POWERON); REQUIRE(app.appLogic()<0);
    REQUIRE(app.state()==stateCodes::FAILURE);
}

/// Assert correct one-based wire commands and all status read/re-read outcomes.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU production control traffic and status retries", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::turnOutletOn(); trippLitePDU::turnOutletOff(); trippLitePDU::devStatus();
    trippLitePDU::updateOutletState(); trippLitePDU::updateOutletStates();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure(); REQUIRE(app.appStartup()==0); app.driver();
    REQUIRE(app.turnOutletOn(0)==0); REQUIRE(app.turnOutletOff(7)==0);
    REQUIRE(g_faults.m_commands[0]=="loadctl on -o 1 --force\r");
    REQUIRE(g_faults.m_commands[1]=="loadctl off -o 8 --force\r");
    g_faults.m_transportResults={-1}; REQUIRE(app.turnOutletOn(0)<0);
    g_faults.m_transportResults={-1}; REQUIRE(app.turnOutletOff(7)<0);
    g_faults.m_status=status();
    REQUIRE(app.updateOutletState(3)==0);
    REQUIRE(app.m_pduValid);
    for(int timeout : {TTY_E_TIMEOUTONREAD,TTY_E_TIMEOUTONREADPOLL})
    {
        std::string text;
        g_faults.m_transportResults={timeout,0}; REQUIRE(app.devStatus(text)==1);
        g_faults.m_transportResults={timeout,-1}; REQUIRE(app.devStatus(text)<0);
    }
    g_faults.m_transportResults={-1}; REQUIRE(app.updateOutletStates()==0);
    REQUIRE(!app.m_pduValid); REQUIRE(app.state()==stateCodes::NOTCONNECTED);
    REQUIRE(app.outletState(0)==OUTLET_STATE_UNKNOWN);
    g_faults.m_transportResults={TTY_E_TIMEOUTONREAD,0}; REQUIRE(app.updateOutletStates()==0);
    REQUIRE(!app.m_pduValid);
    g_faults.m_status="header\nInvalid\n";
    REQUIRE(app.updateOutletStates()==0); REQUIRE(!app.m_pduValid);
    g_faults.m_recordResult=-1; g_faults.m_transportResults={-1};
    app.m_outletTelemRecorded=false;
    REQUIRE(app.updateOutletStates()<0);
}

/// Verify all parser errors, ignored lines, final-line handling, and complete sample validity.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU parser rejects malformed samples without valid partial telemetry", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::parsePDUStatus();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure();
    std::string report=status(); REQUIRE(app.parsePDUStatus(report)==0);
    REQUIRE(app.m_pduSample==std::vector<float>{60,120,4});
    REQUIRE(app.m_pduValid);
    REQUIRE(app.outletState(0)==OUTLET_STATE_ON); REQUIRE(app.outletState(7)==OUTLET_STATE_ON);
    REQUIRE(app.outletState(1)==OUTLET_STATE_OFF);
    for(const auto &[line,error] : std::vector<std::pair<std::string,int>>{
        {"Input Voltage:123V",-1},{"Input Voltage:    ",-2},{"Input Voltage: 123",-3},
        {"Input Frequency:60Hz",-4},{"Input Frequency:   ",-5},{"Input Frequency: 60z",-6},
        {"Output Current:3A",-7},{"Output Current:    ",-8},{"Output Current: 3",-9},
        {"Outlets On:1",-10},{"Outlets On:    ",-11},{"Output XXXXX",-12},{"Qxxxxxxx",-13},
        {"I",-14},{"O",-15},{"Input Voltage: junkV",-16},{"Input Frequency: junkH",-17},
        {"Output Current: junkA",-18},{"Input Voltage: 12junkV",-16},{"Input Frequency: 12junkH",-17},
        {"Output Current: 12junkA",-18},{"Input XXXXXXX",-1} })
    {
        report="header\nInput Voltage: 100 V\n"+line+"\n";
        INFO(line); REQUIRE(app.parsePDUStatus(report)==error);
        REQUIRE(!app.m_pduValid);
        REQUIRE(app.m_pduSample==std::vector<float>{60,120,4});
    }
    report="header\r\n\n----------------\n01: model\nLow Transfer Voltage: 70 V\n ignored\nDevice Status: OK\n$> \n"
           "Output Voltage: 120 V\nOutput Frequency: 60 Hz\nInput Voltage: 120 V\n"
           "Input Frequency: 60 Hz\nOutput Current: 4 A\nOutlets On: NONE";
    REQUIRE(app.parsePDUStatus(report)==0); REQUIRE(app.m_pduValid);
    for(int n=0;n<8;++n) REQUIRE(app.outletState(n)==OUTLET_STATE_OFF);
    report="header\nOutlets On: 0 9 bad 2\n";
    REQUIRE(app.parsePDUStatus(report)==0); REQUIRE(!app.m_pduValid);
    REQUIRE(app.outletState(1)==OUTLET_STATE_ON);
    report="header without newline";
    REQUIRE(app.parsePDUStatus(report)==0); REQUIRE(!app.m_pduValid);
    report="header\nInput Voltage: 120 V\n";
    REQUIRE(app.parsePDUStatus(report)==0); REQUIRE(!app.m_pduValid);
}

/// Verify every warning/alert/emergency threshold and exact boundary priorities.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU alarms use measured values and configured thresholds", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::updateAlarmsAndWarnings();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure();
    app.m_frequency=60; app.m_voltage=120; app.m_current=4;
    auto normal=[&] {app.m_frequency=60;app.m_voltage=120;app.m_current=4;g_faults.m_logs.clear();};
    for(const auto &[value,priority] : std::vector<std::pair<float,flatlogs::logPrioT>>{
        {57,flatlogs::logPrio::LOG_EMERGENCY},{63,flatlogs::logPrio::LOG_EMERGENCY},
        {58,flatlogs::logPrio::LOG_ALERT},{62,flatlogs::logPrio::LOG_ALERT},
        {59,flatlogs::logPrio::LOG_WARNING},{61,flatlogs::logPrio::LOG_WARNING}})
    {
        normal(); app.m_frequency=value; app.updateAlarmsAndWarnings();
        REQUIRE(g_faults.m_logs.size()==1); REQUIRE(g_faults.m_logs[0].m_priority==priority);
        REQUIRE(g_faults.m_logs[0].m_message.find("Hz")!=std::string::npos);
    }
    for(const auto &[value,priority] : std::vector<std::pair<float,flatlogs::logPrioT>>{
        {99,flatlogs::logPrio::LOG_EMERGENCY},{128,flatlogs::logPrio::LOG_EMERGENCY},
        {101,flatlogs::logPrio::LOG_ALERT},{126,flatlogs::logPrio::LOG_ALERT},
        {105,flatlogs::logPrio::LOG_WARNING},{125,flatlogs::logPrio::LOG_WARNING}})
    {
        normal(); app.m_voltage=value; app.updateAlarmsAndWarnings();
        REQUIRE(g_faults.m_logs.size()==1); REQUIRE(g_faults.m_logs[0].m_priority==priority);
        REQUIRE(g_faults.m_logs[0].m_message.find("V")!=std::string::npos);
    }
    for(const auto &[value,priority] : std::vector<std::pair<float,flatlogs::logPrioT>>{
        {20,flatlogs::logPrio::LOG_EMERGENCY},{16,flatlogs::logPrio::LOG_ALERT},{15,flatlogs::logPrio::LOG_WARNING}})
    {
        normal(); app.m_current=value; app.updateAlarmsAndWarnings();
        REQUIRE(g_faults.m_logs.size()==1); REQUIRE(g_faults.m_logs[0].m_priority==priority);
    }
    normal(); app.updateAlarmsAndWarnings(); REQUIRE(g_faults.m_logs.empty());
}

/// Check actual initial/change/forced electrical/outlet payloads and record/schedule failures.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU telemetry distinguishes complete samples and invalidation", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::recordPDU(); trippLitePDU::recordTelem(); trippLitePDU::checkRecordTimes(); trippLitePDU::appLogic();
    #endif
    // clang-format on
    g_faults={}; Fixture app; app.configure(); REQUIRE(app.appStartup()==0);
    REQUIRE(app.recordPDU()==0);
    REQUIRE(!MagAOX::logger::telem_pdu::valid(g_faults.m_records.back().m_payload.data()));
    g_faults.m_status=status(); REQUIRE(app.appLogic()==0);
    REQUIRE(app.m_pduValid);
    auto &record=g_faults.m_records.back();
    REQUIRE(MagAOX::logger::telem_pdu::frequency(record.m_payload.data())==60);
    REQUIRE(MagAOX::logger::telem_pdu::voltage(record.m_payload.data())==120);
    REQUIRE(MagAOX::logger::telem_pdu::current(record.m_payload.data())==4);
    REQUIRE(MagAOX::logger::telem_pdu::valid(record.m_payload.data()));
    size_t count=g_faults.m_records.size();
    REQUIRE(app.appLogic()==0); REQUIRE(g_faults.m_records.size()==count);
    g_faults.m_due=true; REQUIRE(app.appLogic()==0); REQUIRE(g_faults.m_records.size()==count+2);
    g_faults.m_due=false;
    g_faults.m_status=status(61,121,5); REQUIRE(app.appLogic()==0);
    REQUIRE(MagAOX::logger::telem_pdu::current(g_faults.m_records.back().m_payload.data())==5);
    g_faults.m_recordResult=-1;
    REQUIRE(app.recordPDU(true)<0);
    REQUIRE(app.recordTelem(static_cast<const MagAOX::logger::telem_outlet *>(nullptr))<0);
    REQUIRE(app.recordTelem(static_cast<const MagAOX::logger::telem_pdu *>(nullptr))<0);
    app.m_pduTelemRecorded=false;
    REQUIRE(app.appLogic()<0);
    g_faults.m_recordResult=0; g_faults.m_telemResults[3]=-1;
    REQUIRE(app.appLogic()<0);
}
/// Reject the one-past-end physical outlet configuration before control or telemetry can index it.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("trippLitePDU rejects a one-past-end configured outlet", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::loadConfig(); trippLitePDU::loadConfigImpl();
    #endif
    // clang-format on
    outletHarness::g_faults={};
    Fixture app;
    app.configText("[invalid]\noutlet=9\n");
    app.loadConfig(); REQUIRE(app.m_shutdown);
}
/// Verify actual threshold, protocol, and timeout overrides and I/O configuration failure.
/** \ingroup trippLitePDU_unit_test */
TEST_CASE("PDU real configuration overrides limits and timeouts", "[trippLitePDU]")
{
    // clang-format off
    #ifdef TRIPPLITEPDU_TEST_DOXYGEN_REF
    trippLitePDU::setupConfig(); trippLitePDU::loadConfigImpl(); trippLitePDU::updateAlarmsAndWarnings();
    #endif
    // clang-format on
    g_faults={}; Fixture app;
    app.configText("[device]\npowerAlertVersion=1\n[limits]\nfreqLowWarn=58.5\nfreqHighWarn=61.5\n"
                   "freqLowAlert=57.5\nfreqHighAlert=62.5\nfreqLowEmerg=56.5\nfreqHighEmerg=63.5\n"
                   "voltLowWarn=104.5\nvoltHighWarn=125.5\nvoltLowAlert=100.5\nvoltHighAlert=126.5\n"
                   "voltLowEmerg=98.5\nvoltHighEmerg=128.5\ncurrWarn=14.5\ncurrAlert=15.5\ncurrEmerg=19.5\n"
                   "[x]\noutlet=1\n");
    REQUIRE(app.loadConfigImpl(app.config)==0);
    REQUIRE(app.m_deviceVersion==1);
    app.m_frequency=58.75; app.m_voltage=120; app.m_current=4;
    g_faults.m_logs.clear(); app.updateAlarmsAndWarnings(); REQUIRE(g_faults.m_logs.empty());
    app.m_current=14.75; app.updateAlarmsAndWarnings();
    REQUIRE(g_faults.m_logs.size()==1); REQUIRE(g_faults.m_logs[0].m_priority==flatlogs::logPrio::LOG_WARNING);
    g_faults.m_ioLoad=-1;
    REQUIRE(app.loadConfigImpl(app.config)<0);
}
} // namespace trippLitePDUTest
} // namespace libXWCTest
