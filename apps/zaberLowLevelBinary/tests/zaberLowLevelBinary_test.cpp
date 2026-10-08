/** \file zaberLowLevelBinary_test.cpp
 * \brief Catch2 tests for the zaberLowLevelBinary app.
 *
 * \ingroup zaberLowLevelBinary_files
 */

#include <filesystem>
#include <fstream>

extern "C"
{
#include "../zb_serial.c"
}

#include "../../../tests/testXWC.hpp"
#include "../../../tests/testMacrosINDI.hpp"

#include "../zaberLowLevelBinary.hpp"

using namespace MagAOX::app;

namespace libXWCTest
{

/** \defgroup zaberLowLevelBinary_unit_test zaberLowLevelBinary Unit Tests
 * \brief Unit tests for the zaberLowLevelBinary application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `zaberLowLevelBinary` unit tests.
/** \ingroup zaberLowLevelBinary_unit_test
 */
namespace zaberLowLevelBinaryTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
/// Production controller fixture with private FIFOs and retained state files.
class zaberLowLevelBinary_test : public zaberLowLevelBinary
{
  public:
    /// Construct the test harness and set up INDI callback fixtures.
    zaberLowLevelBinary_test( const std::string &device /**< [in] Configured test device name. */ );

    /// Set up a single staged snapshot and INDI transport for power-off tests.
    int setupPowerOffSnapshot( const std::string &stageName /**< [in] Configured stage name. */, long rawPos /**< [in] Retained raw position in microsteps. */, bool parked /**< [in] Retained parked flag. */, long maxPos /**< [in] Retained maximum position in microsteps. */, time_t lastHomed /**< [in] Retained last-home time in seconds. */ );

    /// Configure a stage entry for discovery and recovery tests.
    int addConfiguredStage( const std::string &stageName /**< [in] Configured stage name. */, const std::string &serial /**< [in] Configured stage serial number. */, int deviceAddress = -1 /**< [in] Cached binary device address, or -1 if absent. */ );

    /// Load a discovery snapshot through the production mapping code.
    int loadDiscoverySnapshot( const std::vector<int> &addresses /**< [in] Discovered device addresses. */, const std::vector<std::string> &serials /**< [in] Discovered serial numbers. */ );

    /// Set the cached device address for a configured stage.
    int setDeviceAddressFor( size_t stageIndex /**< [in] Configured stage index. */, int deviceAddress /**< [in] Cached binary device address, or -1 if absent. */ );

    /// Get the cached device address for a configured stage.
    int deviceAddressFor( size_t stageIndex /**< [in] Configured stage index. */ );

    /// Drive the recoverable error handler under test.
    int recoverTransportError( bool devicePresent /**< [in] Whether the USB tty remains available. */ );

    /// Set the FSM state for recovery tests.
    int setAppState( stateCodes::stateCodeT newState /**< [in] New application FSM state. */ );

    /// Get the FSM state for recovery tests.
    stateCodes::stateCodeT appState();

    /// Read the value of a text, number, or switch element from a test property.
    std::string propertyValue( const pcf::IndiProperty &property /**< [in] Property containing the requested element. */, const std::string &element /**< [in] Name of the element to read. */ ) const;

    /// Get the current-position property value for a stage.
    std::string currPosValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Get the target-position property value for a stage.
    std::string tgtPosValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Get the parked-state property value for a stage.
    std::string parkedValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Get the last-homed property value for a stage.
    std::string lastHomedValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Get the max-position property value for a stage.
    std::string maxPosValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Get the current-state property value for a stage.
    std::string currStateValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Get the warning-switch property value for a stage.
    pcf::IndiElement::SwitchStateType warnValue( const std::string &stageName /**< [in] Configured stage name. */ ) const;

    /// Invoke the power-off handling under test.
    int doOnPowerOff();

    /// Stop the private driver and remove only fixture-owned files.
    ~zaberLowLevelBinary_test() noexcept;

  private:
    std::filesystem::path m_testRoot; ///< Temporary directory backing the test FIFOs and state snapshot.
};

/// Stage helper fixture exposing only homing bookkeeping under test.
class zaberBinaryStage_test : public zaberBinaryStage<zaberLowLevelBinary_test>
{
  public:
    /// Construct a test binary-stage helper.
    zaberBinaryStage_test( zaberLowLevelBinary_test *parent /**< [in] Non-owning test application instance. */ );

    /// Set the fields used to detect homing completion.
    void setHomeState( bool homing /**< [in] Whether homing is active. */, bool warnWR /**< [in] Whether the stage still requires homing. */, long tgtPos /**< [in] Requested position in microsteps. */, long rawPos /**< [in] Retained raw position in microsteps. */, time_t lastHomed /**< [in] Retained last-home time in seconds. */ );

    /// Invoke the last-home timestamp refresh logic under test.
    int refreshLastHomed( bool wasHoming /**< [in] Whether the previous status was homing. */ );

    /// Get the stored last-home seconds value.
    time_t lastHomedSec() const;
};
inline zaberLowLevelBinary_test::zaberLowLevelBinary_test( const std::string &device )
{
    m_configName = device;

    XWCTEST_SETUP_INDI_NEW_PROP( tgt_pos );
    XWCTEST_SETUP_INDI_NEW_PROP( req_home );
    XWCTEST_SETUP_INDI_NEW_PROP( req_home_all );
    XWCTEST_SETUP_INDI_NEW_PROP( req_halt );
    XWCTEST_SETUP_INDI_NEW_PROP( req_ehalt );
    XWCTEST_SETUP_INDI_NEW_PROP( knob_enable );
}

inline int zaberLowLevelBinary_test::setupPowerOffSnapshot( const std::string &stageName, long rawPos, bool parked, long maxPos, time_t lastHomed )
{
    std::error_code ec;

    m_testRoot = std::filesystem::temp_directory_path() / ( "zaberLowLevelBinary_test_" + m_configName );
    std::filesystem::remove_all( m_testRoot, ec );

    m_basePath = m_testRoot.string();
    m_sysPath  = ( m_testRoot / "sys" ).string();

    std::filesystem::create_directories( m_testRoot / MAGAOX_driverFIFORelPath );
    std::filesystem::create_directories( std::filesystem::path( m_sysPath ) / m_configName );

    m_stages.emplace_back( this );
    m_stages.back().name( stageName );
    m_stages.back().serial( "serial0" );

    {
        std::ofstream stateOut( std::filesystem::path( m_sysPath ) / m_configName / stageName );
        stateOut << rawPos << '\n' << parked << '\n' << maxPos << '\n' << lastHomed << '\n';
    }

    state( stateCodes::INITIALIZED );
    if( appStartup() < 0 )
    {
        return -1;
    }

    if( createINDIFIFOS() < 0 )
    {
        return -1;
    }

    m_indiP_state = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_state.setDevice( m_configName );
    m_indiP_state.setName( "fsm" );
    m_indiP_state.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_state.add( pcf::IndiElement( "state" ) );

    m_indiDriver = new indiDriver<MagAOXAppT>( this, m_configName, "0", "0" );

    return ( m_indiDriver && m_indiDriver->good() ) ? 0 : -1;
}

inline int zaberLowLevelBinary_test::addConfiguredStage( const std::string &stageName, const std::string &serial, int deviceAddress )
{
    m_stages.emplace_back( this );
    m_stages.back().name( stageName );
    m_stages.back().serial( serial );
    m_stages.back().deviceAddress( deviceAddress );

    const size_t idx = m_stages.size() - 1;

    m_stageName.insert( { stageName, idx } );
    m_stageSerial.insert( { serial, idx } );

    return 0;
}

inline int zaberLowLevelBinary_test::loadDiscoverySnapshot( const std::vector<int> &addresses, const std::vector<std::string> &serials )
{
    return loadStages( addresses, serials );
}

inline int zaberLowLevelBinary_test::setDeviceAddressFor( size_t stageIndex, int deviceAddress )
{
    m_stages.at( stageIndex ).deviceAddress( deviceAddress );
    return 0;
}

inline int zaberLowLevelBinary_test::deviceAddressFor( size_t stageIndex )
{
    return m_stages.at( stageIndex ).deviceAddress();
}

inline int zaberLowLevelBinary_test::recoverTransportError( bool devicePresent )
{
    return recoverFromError( devicePresent );
}

inline int zaberLowLevelBinary_test::setAppState( stateCodes::stateCodeT newState )
{
    state( newState );
    return 0;
}

inline stateCodes::stateCodeT zaberLowLevelBinary_test::appState()
{
    return state();
}

inline std::string zaberLowLevelBinary_test::propertyValue( const pcf::IndiProperty &property, const std::string &element ) const
{
    return property[element].getValue();
}

inline std::string zaberLowLevelBinary_test::currPosValue( const std::string &stageName ) const
{
    return propertyValue( m_indiP_curr_pos, stageName );
}

inline std::string zaberLowLevelBinary_test::tgtPosValue( const std::string &stageName ) const
{
    return propertyValue( m_indiP_tgt_pos, stageName );
}

inline std::string zaberLowLevelBinary_test::parkedValue( const std::string &stageName ) const
{
    return propertyValue( m_indiP_parked, stageName );
}

inline std::string zaberLowLevelBinary_test::lastHomedValue( const std::string &stageName ) const
{
    return propertyValue( m_indiP_lastHomed, stageName );
}

inline std::string zaberLowLevelBinary_test::maxPosValue( const std::string &stageName ) const
{
    return propertyValue( m_indiP_max_pos, stageName );
}

inline std::string zaberLowLevelBinary_test::currStateValue( const std::string &stageName ) const
{
    return propertyValue( m_indiP_curr_state, stageName );
}

inline pcf::IndiElement::SwitchStateType zaberLowLevelBinary_test::warnValue( const std::string &stageName ) const
{
    return m_indiP_warn[stageName].getSwitchState();
}

inline int zaberLowLevelBinary_test::doOnPowerOff()
{
    return onPowerOff();
}

inline zaberLowLevelBinary_test::~zaberLowLevelBinary_test() noexcept
{
    std::error_code ec;

    delete m_indiDriver;
    m_indiDriver = nullptr;
    std::filesystem::remove_all( m_testRoot, ec );
}

inline zaberBinaryStage_test::zaberBinaryStage_test( zaberLowLevelBinary_test *parent ) : zaberBinaryStage<zaberLowLevelBinary_test>( parent )
{
}

inline void zaberBinaryStage_test::setHomeState( bool homing, bool warnWR, long tgtPos, long rawPos, time_t lastHomed )
{
    m_homing            = homing;
    m_warnWR            = warnWR;
    m_tgtPos            = tgtPos;
    m_rawPos            = rawPos;
    m_lastHomed.tv_sec  = lastHomed;
    m_lastHomed.tv_nsec = 0;
}

inline int zaberBinaryStage_test::refreshLastHomed( bool wasHoming )
{
    return updateLastHomed( wasHoming );
}

inline time_t zaberBinaryStage_test::lastHomedSec() const
{
    return m_lastHomed.tv_sec;
}
/// \endcond

/// Verify registered command property validation.
/**
 * \ingroup zaberLowLevelBinary_unit_test
 */
SCENARIO( "INDI Callbacks", "[zaberLowLevelBinary]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::newCallBack_m_indiP_tgt_pos( pcf::IndiProperty() );
    zaberLowLevelBinary::newCallBack_m_indiP_req_home(); zaberLowLevelBinary::newCallBack_m_indiP_req_home_all();
    zaberLowLevelBinary::newCallBack_m_indiP_req_halt(); zaberLowLevelBinary::newCallBack_m_indiP_req_ehalt();
    zaberLowLevelBinary::newCallBack_m_indiP_knob_enable();
    zaberLowLevelBinary::st_newCallBack_m_indiP_tgt_pos(); zaberLowLevelBinary::st_newCallBack_m_indiP_req_home();
    zaberLowLevelBinary::st_newCallBack_m_indiP_req_home_all(); zaberLowLevelBinary::st_newCallBack_m_indiP_req_halt();
    zaberLowLevelBinary::st_newCallBack_m_indiP_req_ehalt(); zaberLowLevelBinary::st_newCallBack_m_indiP_knob_enable();
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, tgt_pos );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_home );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_home_all );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_halt );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, req_ehalt );
    XWCTEST_INDI_NEW_CALLBACK( zaberLowLevelBinary, knob_enable );
}

/// Verify observed Off preserves retained position and parked metadata while clearing warnings.
/** \ingroup zaberLowLevelBinary_unit_test */
SCENARIO( "Power-off INDI snapshot retains stage state", "[zaberLowLevelBinary]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::appStartup(); zaberLowLevelBinary::onPowerOff();
    #endif
    // clang-format on
    zaberLowLevelBinary_test zllbt( "zllbtest" );

    REQUIRE( zllbt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );

    REQUIRE( zllbt.doOnPowerOff() == 0 );

    REQUIRE( zllbt.currPosValue( "stageA" ) == "12345" );
    REQUIRE( zllbt.tgtPosValue( "stageA" ) == "12345" );
    REQUIRE( zllbt.parkedValue( "stageA" ) == "1" );
    REQUIRE( zllbt.lastHomedValue( "stageA" ) == "77" );
    REQUIRE( zllbt.maxPosValue( "stageA" ) == "54321" );
    REQUIRE( zllbt.currStateValue( "stageA" ) == "POWEROFF" );
    REQUIRE( zllbt.warnValue( "stageA" ) == pcf::IndiElement::Off );
}

/// Verify last-home timestamps advance on homing completion while preserving idle timestamps.
/** \ingroup zaberLowLevelBinary_unit_test */
SCENARIO( "Binary last-home timestamps refresh after homing completes", "[zaberLowLevelBinary]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberBinaryStage<zaberLowLevelBinary>::updateLastHomed();
    #endif
    // clang-format on
    zaberLowLevelBinary_test zllbt( "zllbtest" );
    zaberBinaryStage_test    stage( &zllbt );

    WHEN( "a homing sequence completes with a stale stored timestamp" )
    {
        stage.setHomeState( false, false, 0, 0, 77 );

        REQUIRE( stage.refreshLastHomed( true ) == 0 );
        REQUIRE( stage.lastHomedSec() != 77 );
    }

    WHEN( "the stage is merely idle at home with an existing timestamp" )
    {
        stage.setHomeState( false, false, 0, 0, 77 );

        REQUIRE( stage.refreshLastHomed( false ) == 0 );
        REQUIRE( stage.lastHomedSec() == 77 );
    }
}

/// Verify discovery clears stale addresses and reports missing configured stages safely.
/**
 * \ingroup zaberLowLevelBinary_unit_test
 */
SCENARIO( "Binary discovery resets stale device addresses", "[zaberLowLevelBinary]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::loadStages( std::declval<const std::vector<int> &>(), std::declval<const std::vector<std::string> &>() );
    #endif
    // clang-format on

    zaberLowLevelBinary_test zllbt( "zllbtest" );

    REQUIRE( zllbt.addConfiguredStage( "stagebs", "64040", 1 ) == 0 );
    REQUIRE( zllbt.addConfiguredStage( "stageirf", "122400", 2 ) == 0 );

    REQUIRE( zllbt.loadDiscoverySnapshot( { 1 }, { "64040" } ) == ZBC_CONNECTED );
    REQUIRE( zllbt.deviceAddressFor( 0 ) == 1 );
    REQUIRE( zllbt.deviceAddressFor( 1 ) < 1 );
}

/// Verify a later discovery pass can find a stage that was missing initially.
/**
 * \ingroup zaberLowLevelBinary_unit_test
 */
SCENARIO( "Binary discovery can find devices that appear later", "[zaberLowLevelBinary]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::loadStages();
    #endif
    // clang-format on
    zaberLowLevelBinary_test zllbt( "zllbtest_rediscover" );

    REQUIRE( zllbt.addConfiguredStage( "stagebs", "64040" ) == 0 );
    REQUIRE( zllbt.addConfiguredStage( "stageirf", "122400" ) == 0 );

    REQUIRE( zllbt.loadDiscoverySnapshot( { 1 }, { "64040" } ) == ZBC_CONNECTED );
    REQUIRE( zllbt.deviceAddressFor( 0 ) == 1 );
    REQUIRE( zllbt.deviceAddressFor( 1 ) < 1 );

    REQUIRE( zllbt.loadDiscoverySnapshot( { 1, 2 }, { "64040", "122400" } ) == ZBC_CONNECTED );
    REQUIRE( zllbt.deviceAddressFor( 0 ) == 1 );
    REQUIRE( zllbt.deviceAddressFor( 1 ) == 2 );
}

/// Verify communication failures drop the binary app back into reconnectable states.
/**
 * \ingroup zaberLowLevelBinary_unit_test
 */
SCENARIO( "Recoverable binary transport errors transition to reconnect states", "[zaberLowLevelBinary]" )
{
    // clang-format off
    #ifdef ZABERLOWLEVELBINARY_TEST_DOXYGEN_REF
    zaberLowLevelBinary::resetConnection();
    zaberLowLevelBinary::recoverFromError( true );
    #endif
    // clang-format on

    SECTION( "A present tty returns the app to NOTCONNECTED" )
    {
        zaberLowLevelBinary_test zllbt( "zllbtest_present" );

        REQUIRE( zllbt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );
        REQUIRE( zllbt.setDeviceAddressFor( 0, 1 ) == 0 );
        REQUIRE( zllbt.setAppState( stateCodes::ERROR ) == 0 );

        REQUIRE( zllbt.recoverTransportError( true ) == 0 );
        REQUIRE( zllbt.appState() == stateCodes::NOTCONNECTED );
        REQUIRE( zllbt.currStateValue( "stageA" ) == "NOTCONNECTED" );
    }

    SECTION( "A missing tty returns the app to NODEVICE" )
    {
        zaberLowLevelBinary_test zllbt( "zllbtest_missing" );

        REQUIRE( zllbt.setupPowerOffSnapshot( "stageA", 12345, true, 54321, 77 ) == 0 );
        REQUIRE( zllbt.setDeviceAddressFor( 0, 1 ) == 0 );
        REQUIRE( zllbt.setAppState( stateCodes::ERROR ) == 0 );

        REQUIRE( zllbt.recoverTransportError( false ) == 0 );
        REQUIRE( zllbt.appState() == stateCodes::NODEVICE );
        REQUIRE( zllbt.currStateValue( "stageA" ) == "NODEVICE" );
    }
}

} // namespace zaberLowLevelBinaryTest

} // namespace libXWCTest
