/** \file accelFFCtrlShadowMonitor_test.cpp
 * \brief File-only native shadow gates, exact timing records and exclusive log ownership.
 * \ingroup accelFFCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../accelFFCtrlShadowMonitor.hpp"

#include <filesystem>
#include <fstream>

namespace libXWCTest
{
namespace accelFFCtrlTest
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// \cond ACCELFFCTRL_SHADOW_MONITOR_FIXTURE
/// Own only a private diagnostic directory, never a shared-memory namespace.
struct monitorFixture
{
    std::string      m_path;     ///< Private directory removed when the fixture is destroyed.
    shadowSettings   m_settings; ///< Explicit file-only opt-in, not an application or actuator enable.
    inputDiagnostics m_input;    ///< Deterministic immutable ready snapshot.
    inputClock       m_clock{ 1700000001000000000, 1000000000, 10 }; ///< Bracketed independent clock epochs.
    wfsTiming        m_wfs{ 17, 2, 1700000000999500000, 1700000000999600000, true }; ///< Fresh WFS metadata.

    /// Prepare positive clocks and a future target without constructing a framework application.
    monitorFixture();

    /// Remove only the fixture's own files.
    ~monitorFixture();
};

monitorFixture::monitorFixture()
{
    char path[] = "/tmp/accelFFCtrl-monitor-XXXXXX";
    REQUIRE( mkdtemp( path ) != nullptr );
    m_path                              = path;
    m_settings.m_enabled                = true;
    m_settings.m_wfsName                = "fixture-wfs";
    m_settings.m_logFile                = m_path + "/diagnostics.csv";
    m_input.m_counter                   = 9007199254740993ULL;
    m_input.m_generation                = 3;
    m_input.m_prediction.m_state        = estimatorState::ready;
    m_input.m_prediction.m_sampleTimeNs = m_clock.m_mono - 125000;
    m_input.m_prediction.m_targetTimeNs = m_clock.m_mono + 1000000;
    m_input.m_prediction.m_pointing     = { 2e-8, -3e-8 };
}

monitorFixture::~monitorFixture()
{
    std::filesystem::remove_all( m_path );
}

/// Split a simple numeric CSV record, preserving exact integer text.
std::vector<std::string> monitorColumns( const std::string &line /**< [in] one numeric CSV row */ )
{
    std::istringstream       stream( line );
    std::string              value;
    std::vector<std::string> result;
    while( std::getline( stream, value, ',' ) )
        result.push_back( value );
    return result;
}
/// \endcond

/** \brief Startup opt-in requires a separate non-DM timing source and a new file; default gain is zero.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl file-only shadow defaults and settings", "[accelFFCtrl][shadow-monitor]" )
{
    shadowSettings defaults;
    REQUIRE( defaults.valid() );
    REQUIRE_FALSE( defaults.m_enabled );
    REQUIRE( defaults.m_gain == 0 );
    defaults.m_enabled = true;
    REQUIRE_FALSE( defaults.valid() );
    monitorFixture f;
    REQUIRE( f.m_settings.valid() );
    for( const std::string name : { "dm01disp00", "../fixture-wfs", "" } )
    {
        auto invalid      = f.m_settings;
        invalid.m_wfsName = name;
        REQUIRE_FALSE( invalid.valid() );
    }
    for( double gain : { -1., 1.01, std::numeric_limits<double>::quiet_NaN() } )
    {
        auto invalid   = f.m_settings;
        invalid.m_gain = gain;
        REQUIRE_FALSE( invalid.valid() );
    }
}

/** \brief New absolute proposals oppose both model axes; disabled and zero gain never retain a command.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl file-only model sign and gain", "[accelFFCtrl][shadow-monitor]" )
{
    monitorFixture f;
    auto           zero = evaluateShadow( f.m_settings, f.m_input, f.m_clock, f.m_wfs );
    REQUIRE( zero.m_valid );
    REQUIRE( zero.m_proposed == std::array<double, 2>{} );
    REQUIRE( zero.m_unitGain == std::array<double, 2>{ -2e-8, 3e-8 } );
    f.m_settings.m_gain = .5;
    auto half           = evaluateShadow( f.m_settings, f.m_input, f.m_clock, f.m_wfs );
    REQUIRE( half.m_proposed == std::array<double, 2>{ -1e-8, 1.5e-8 } );
    f.m_settings.m_enabled = false;
    auto disabled          = evaluateShadow( f.m_settings, f.m_input, f.m_clock, f.m_wfs );
    REQUIRE_FALSE( disabled.m_valid );
    REQUIRE( disabled.m_proposed == std::array<double, 2>{} );
}

/** \brief Stale or untrusted metadata, pending resets and invalid targets inhibit without changing numerical state.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl file-only shadow integrity gates", "[accelFFCtrl][shadow-monitor]" )
{
    monitorFixture f;
    f.m_settings.m_gain = 1;
    for( unsigned fault = 0; fault < 13; ++fault )
    {
        auto input = f.m_input;
        auto clock = f.m_clock;
        auto wfs   = f.m_wfs;
        switch( fault )
        {
        case 0:
            wfs.m_valid = false;
            break;
        case 1:
            wfs.m_acquired -= 3000000;
            break;
        case 2:
            wfs.m_written = wfs.m_acquired - 1;
            break;
        case 3:
            wfs.m_acquired = clock.m_real + 1;
            break;
        case 4:
            wfs.m_counter = 0;
            break;
        case 5:
            input.m_prediction.m_sampleTimeNs -= 3000000;
            break;
        case 6:
            input.m_prediction.m_targetTimeNs = clock.m_mono - 1;
            break;
        case 7:
            input.m_prediction.m_state = estimatorState::warming;
            break;
        case 8:
            input.m_prediction.m_pointing[1] = std::numeric_limits<double>::quiet_NaN();
            break;
        case 9:
            clock.m_real = 0;
            break;
        case 10:
            clock.m_uncertainty = -1;
            break;
        case 11:
            input.m_prediction.m_targetTimeNs += input.m_config.m_maxPredictionNs;
            break;
        case 12:
            input.m_prediction.m_sampleTimeNs = clock.m_mono + 1;
            break;
        }
        auto result = evaluateShadow( f.m_settings, input, clock, wfs );
        REQUIRE_FALSE( result.m_valid );
        REQUIRE( result.m_proposed == std::array<double, 2>{} );
    }
    const auto pending = evaluateShadow( f.m_settings, f.m_input, f.m_clock, f.m_wfs, true );
    REQUIRE_FALSE( pending.m_valid );
    REQUIRE( pending.m_proposed == std::array<double, 2>{} );
    REQUIRE( f.m_input.m_prediction.m_targetTimeNs == 1001000000 );
}

/** \brief Logs retain exact counters/timestamps, carry no DM enable, refuse overwrites and end explicitly inhibited.
 * \ingroup accelFFCtrl_unit_test
 */
TEST_CASE( "accelFFCtrl exclusive diagnostic CSV lifecycle", "[accelFFCtrl][shadow-monitor]" )
{
    monitorFixture f;
    shadowLog      log;
    REQUIRE( log.open( f.m_settings.m_logFile ) );
    shadowLog other;
    REQUIRE_FALSE( other.open( f.m_settings.m_logFile ) );
    const auto link = f.m_path + "/symlink.csv";
    REQUIRE( symlink( f.m_settings.m_logFile.c_str(), link.c_str() ) == 0 );
    REQUIRE_FALSE( other.open( link ) );
    REQUIRE_FALSE( other.open( "relative.csv" ) );
    auto status = evaluateShadow( f.m_settings, f.m_input, f.m_clock, f.m_wfs );
    REQUIRE( log.write( f.m_input, f.m_clock, f.m_wfs, status ) );
    REQUIRE( log.write( f.m_input, f.m_clock, f.m_wfs, status, true ) );
    log.close();
    std::ifstream file( f.m_settings.m_logFile );
    std::string   line;
    REQUIRE( bool( std::getline( file, line ) ) );
    const auto header = monitorColumns( line );
    REQUIRE( header.size() == 32 + parameterNames.size() );
    REQUIRE( bool( std::getline( file, line ) ) );
    const auto row = monitorColumns( line );
    REQUIRE( row.size() == header.size() );
    REQUIRE( row[0] == "1700000001000000000" );
    REQUIRE( row[2] == "9007199254740993" );
    REQUIRE( row[6] == "1001000000" );
    REQUIRE( row[10] == "1700000000999500000" );
    REQUIRE( row[23] == "0" );
    REQUIRE( bool( std::getline( file, line ) ) );
    const auto final = monitorColumns( line );
    REQUIRE( final[8] == "17" );
    REQUIRE( final[15] == "0" );
    for( size_t n = 17; n <= 23; ++n )
        REQUIRE( std::stod( final[n] ) == 0 );
    REQUIRE( final[24] == "1" );
}

} // namespace accelFFCtrlTest
} // namespace libXWCTest
