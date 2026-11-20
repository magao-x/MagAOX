/** \file autoJay_test.cpp
 * \brief Offline application setup, capture, cancellation, and publication tests.
 * \ingroup autoJay_files
 */

#include <chrono>
#include <thread>

#include "../../../tests/testXWC.hpp"
#include "../autoJay.hpp"

namespace libXWCTest
{
namespace autoJayTest
{

/// \cond AUTOJAY_APP_TEST_HELPERS
/// An application fixture whose channel interface has no live-stream implementation.
class autoJayHarness : public MagAOX::app::autoJay
{
  public:
    using imageT      = MagAOX::app::autoJayDetail::imageT;
    using calibration = MagAOX::app::autoJayDetail::calibration;
    using geometry    = MagAOX::app::autoJayDetail::geometry;

    std::filesystem::path m_directory;             ///< Owned scratch files for patterns and calibration bundles.
    unsigned              m_openCount{ 0 };        ///< Calls into the fake channel-open interface.
    unsigned              m_clearCount{ 0 };       ///< Calls into the fake cleanup interface.
    bool                  m_open{ false };         ///< Whether this fake transaction owns channels.
    bool                  m_failClear{ false };    ///< Inject a cleanup failure without touching a stream.
    bool                  m_changeOnOpen{ false }; ///< Inject a setup-generation change while opening fake channels.
    std::vector<std::array<float, 2>> m_commands;  ///< Ordered fake command scales.
    std::array<uint64_t, 2>           m_fakeCounters{}; ///< Fake combined-DM acknowledgments.

    using autoJay::begin;
    using autoJay::controlSettled;
    using autoJay::finish;
    using autoJay::loadConfigImpl;
    using autoJay::m_acquisition;
    using autoJay::m_active;
    using autoJay::m_calibration;
    using autoJay::m_captureConfig;
    using autoJay::m_captureRequested;
    using autoJay::m_continuous;
    using autoJay::m_controlDeadline;
    using autoJay::m_controllerEnabled;
    using autoJay::m_controlSettle;
    using autoJay::m_counter;
    using autoJay::m_cycle;
    using autoJay::m_cycleGeneration;
    using autoJay::m_generation;
    using autoJay::m_lastFrameTime;
    using autoJay::m_observed;
    using autoJay::m_patternFiles;
    using autoJay::m_quality;
    using autoJay::m_session;
    using autoJay::m_startRequested;
    using autoJay::m_status;
    using autoJay::m_stopRequested;
    using autoJay::m_waitingControl;
    using autoJay::monotonic;
    using autoJay::recall;
    using autoJay::stop;

    /// Build valid defaults and synthetic external properties without starting application threads.
    autoJayHarness();

    /// Remove fixture files after discarding fake channel ownership.
    ~autoJayHarness();

    /// Implement a channel open using only fixture state.
    bool openChannels( calibration &cal /**< [in,out] Frozen channel configuration and effective maps */,
                       std::string &error /**< [out] Failure reason */ ) override;

    /// Report fake ownership without inspecting shared memory.
    bool channelsValid( std::string &error /**< [out] Failure reason */ ) const override;

    /// Return the fixture's monotonically advanced trigger counters.
    std::array<uint64_t, 2> triggerCounters() const override;

    /// Record a command and acknowledge both fake streams.
    bool writeChannels( const std::array<float, 2> &scales /**< [in] Per-DM command scales */,
                        std::string                &error /**< [out] Failure reason */ ) override;

    /// Release fake channels or inject a cleanup failure.
    bool clearChannels( std::string &error /**< [out] Failure reason */ ) override;

    /// Deliver a complete property through the real callback, adding only its local subscription slot.
    void deliver( const pcf::IndiProperty &value /**< [in] Synthetic property */ );

    /// Return one current property for a test mutation.
    pcf::IndiProperty observedProperty( const std::string &key /**< [in] Device.property key */ ) const;

    /// Seed a numeric property with one or more values.
    void numbers( const std::string                                    &device /**< [in] Synthetic publisher */,
                  const std::string                                    &name /**< [in] Property name */,
                  std::initializer_list<std::pair<std::string, double>> values /**< [in] Named values */ );

    /// Seed one text element.
    void text( const std::string &device /**< [in] Synthetic publisher */,
               const std::string &name /**< [in] Property name */,
               const std::string &element /**< [in] Element name */,
               const std::string &value /**< [in] Element value */ );

    /// Seed one switch element.
    void toggle( const std::string &device /**< [in] Synthetic publisher */,
                 const std::string &name /**< [in] Property name */,
                 const std::string &element /**< [in] Element name */,
                 bool               on /**< [in] Reported switch state */ );

    /// Set the shared synthetic ROI and stream dimensions.
    void liveGeometry( int width /**< [in] Sampled horizontal pixels */,
                       int height /**< [in] Sampled vertical pixels */,
                       int bin = 1 /**< [in] Detector binning on both axes */ );

    /// Allocate the camera conversion for the fixture's stream dimensions.
    int allocateCamera( uint8_t type /**< [in] Synthetic ImageStreamIO datatype */ );

    /// Pass a synthetic frame and its metadata through the camera callback.
    int cameraFrame( void           *source, /**< [in] Synthetic pixels */
                     IMAGE_METADATA &metadata /**< [in] Metadata for the synthetic frame */ );

    /// Validate a copy of the capture or loaded configuration and return the resolved snapshot.
    bool setupValid( bool         capture /**< [in] Resolve capture wildcards */,
                     std::string &error /**< [out] Setup rejection */,
                     calibration *resolved = nullptr /**< [out] Optional checked snapshot */,
                     bool         settling = false /**< [in] Permit an expected correction in progress */ );

    /// Load application settings from a private test configuration file.
    int configure( const std::string &contents /**< [in] INI configuration text */ );

    /// Generate distinct frames through every real acquisition phase, without application threads.
    void completeSequence();

    /// Configure a controller acknowledgment for the current sample.
    void acknowledgment( double session /**< [in] Producer identity */,
                         double counter /**< [in] Processed sample counter */,
                         int    status /**< [in] Controller outcome */ );

    /// Evaluate an asymmetric synthetic response for one DM.
    imageT response( unsigned dm /**< [in] Select NCPC or tweeter response */ ) const;

    /// Return this fixture's application calibration directory.
    std::string calibrationRoot() const;
};

autoJayHarness::autoJayHarness()
{
    char directory[] = "/tmp/autoJay-app-XXXXXX";
    REQUIRE( ::mkdtemp( directory ) != nullptr );
    m_directory      = directory;
    m_configName     = "autoJay-test";
    m_basePath       = m_directory;
    m_calibDir       = m_directory / "calib";
    m_diagnosticsDir = m_directory / "diagnostics";
    m_tel.logPath( m_directory / "telem" );
    m_tel.logName( m_configName );
    m_log.logPath( m_directory / "log" );
    setupConfig();
    REQUIRE( loadConfigImpl( config ) == 0 );
    m_captureConfig.stream       = "camflowfs";
    m_captureConfig.frames       = 2;
    m_captureConfig.settleFrames = 0;
    m_captureConfig.dmSleep      = 0;
    m_captureConfig.phaseTimeout = 5;
    m_session                    = 123456;
    m_controllerEnabled          = true;
    for( unsigned i = 0; i < 2; ++i )
    {
        m_captureConfig.dm[i].amplitude       = 0.1F;
        m_captureConfig.dm[i].pattern         = imageT::Zero( 4, 4 );
        m_captureConfig.dm[i].pattern( 1, 1 ) = 1;
        m_patternFiles[i] = ( m_directory / ( "pattern" + std::to_string( i ) + ".fits" ) ).string();
        std::string error;
        REQUIRE( MagAOX::app::autoJayDetail::calibrationIO::writeImage(
            m_patternFiles[i], m_captureConfig.dm[i].pattern, error ) );
    }
    for( const auto &device :
         { "stagelosel", "stageflowfs", "holoop", "twAlign-camwfs-ctrl", "autoJay-ctrl", "ttmperi" } )
        text( device, "fsm", "state", "READY" );
    text( "camflowfs", "fsm", "state", "OPERATING" );
    text( "ttmperi", "fsm", "state", "OPERATING" );
    toggle( "stagelosel", "presetName", "pupil", true );
    numbers( "stagelosel", "position", { { "current", 3.5 } } );
    numbers( "stageflowfs", "position", { { "current", 32.0 } } );
    numbers( "camflowfs", "exptime", { { "current", 0.001 } } );
    toggle( "holoop", "loop_state", "toggle", false );
    toggle( "twAlign-camwfs-ctrl", "loop_state", "toggle", false );
    toggle( "autoJay-ctrl", "loop_state", "toggle", false );
    for( const auto &axis : { "axis1_voltage", "axis2_voltage" } )
        numbers( "ttmperi", axis, { { "current", 50 }, { "target", 50 } } );
    liveGeometry( 44, 40 );
    calibration aligned;
    std::string error;
    REQUIRE( setupValid( true, error, &aligned ) );
    aligned.baseline        = imageT::Constant( 44, 40, 500 );
    aligned.dm[0].reference = response( 0 );
    aligned.dm[1].reference = response( 1 );
    aligned.id              = "test-reference";
    m_calibration           = std::make_shared<const calibration>( aligned );
    const std::array<std::pair<pcf::IndiProperty *, std::string>, 6> requests = {
        std::make_pair( &m_indiP_measure, "measure" ),
        { &m_indiP_capture, "capture" },
        { &m_indiP_reload, "reload" },
        { &m_indiP_stop, "stop" },
        { &m_indiP_diagnostics, "save_diagnostics" },
        { &m_indiP_continuous, "continuous" } };
    for( const auto &entry : requests )
    {
        entry.first->setDevice( m_configName );
        entry.first->setName( entry.second );
    }
}

autoJayHarness::~autoJayHarness()
{
    m_open = false;
    std::error_code ignored;
    std::filesystem::remove_all( m_directory, ignored );
}

bool autoJayHarness::openChannels( calibration &cal, std::string &error )
{
    ++m_openCount;

    for( size_t dm = 0; dm < cal.dm.size(); ++dm )
    {
        if( !resolvePattern( cal, dm, 4, 4, error ) )
        {
            return false;
        }
    }

    m_open = true;
    if( m_changeOnOpen )
    {
        ++m_generation;
    }

    return true;
}

bool autoJayHarness::channelsValid( std::string &error ) const
{
    if( !m_open )
        error = "fake channels are closed";
    return m_open;
}

std::array<uint64_t, 2> autoJayHarness::triggerCounters() const
{
    return m_fakeCounters;
}

bool autoJayHarness::writeChannels( const std::array<float, 2> &scales, std::string &error )
{
    if( !channelsValid( error ) )
        return false;
    m_commands.push_back( scales );
    ++m_fakeCounters[0];
    if( !m_cycle.dm[1].trigger.empty() )
    {
        ++m_fakeCounters[1];
    }

    return true;
}

bool autoJayHarness::clearChannels( std::string &error )
{
    ++m_clearCount;
    m_open = false;
    if( m_failClear )
        error = "injected cleanup failure";
    return !m_failClear;
}

void autoJayHarness::deliver( const pcf::IndiProperty &value )
{
    m_observed.try_emplace( value.getDevice() + "." + value.getName() );
    REQUIRE( receive( value ) == 0 );
}

pcf::IndiProperty autoJayHarness::observedProperty( const std::string &key ) const
{
    return m_observed.at( key ).value.value();
}

void autoJayHarness::numbers( const std::string                                    &device,
                              const std::string                                    &name,
                              std::initializer_list<std::pair<std::string, double>> values )
{
    pcf::IndiProperty property( pcf::IndiProperty::Number, device, name );
    property.setState( pcf::IndiProperty::Ok );
    for( const auto &entry : values )
        property.add( pcf::IndiElement( entry.first, entry.second ) );
    deliver( property );
}

void autoJayHarness::text( const std::string &device,
                           const std::string &name,
                           const std::string &element,
                           const std::string &value )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text, device, name );
    property.setState( pcf::IndiProperty::Ok );
    property.add( pcf::IndiElement( element, value ) );
    deliver( property );
}

void autoJayHarness::toggle( const std::string &device, const std::string &name, const std::string &element, bool on )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch, device, name );
    property.setState( pcf::IndiProperty::Ok );
    property.add( pcf::IndiElement( element ) );
    property[element].setSwitchState( on ? pcf::IndiElement::On : pcf::IndiElement::Off );
    deliver( property );
}

void autoJayHarness::liveGeometry( int width, int height, int bin )
{
    m_streamWidth                                                  = width;
    m_streamHeight                                                 = height;
    MagAOX::app::dev::shmimMonitor<MagAOX::app::autoJay>::m_width  = width;
    MagAOX::app::dev::shmimMonitor<MagAOX::app::autoJay>::m_height = height;
    numbers( "camflowfs",
             "roi_region_state",
             { { "x", 21.5 },
               { "y", 19.5 },
               { "w", width * bin },
               { "h", height * bin },
               { "bin_x", bin },
               { "bin_y", bin },
               { "units", 0 } } );
    m_lastFrameTime = monotonic();
}

int autoJayHarness::allocateCamera( uint8_t type )
{
    MagAOX::app::dev::shmimMonitor<MagAOX::app::autoJay>::m_dataType = type;
    return allocate( MagAOX::app::dev::shmimT{} );
}

int autoJayHarness::cameraFrame( void *source, IMAGE_METADATA &metadata )
{
    auto &stream = MagAOX::app::dev::shmimMonitor<MagAOX::app::autoJay>::m_imageStream;
    stream.md    = &metadata;
    int result   = processImage( source, MagAOX::app::dev::shmimT{} );
    stream.md    = nullptr;
    return result;
}

bool autoJayHarness::setupValid( bool capture, std::string &error, calibration *resolved, bool settling )
{
    calibration candidate = capture ? m_captureConfig : *m_calibration;
    geometry    geom;
    bool        result = checkSetup( candidate, geom, capture, error, settling );
    if( result && capture )
        candidate.cameraGeometry = geom;
    if( resolved )
        *resolved = std::move( candidate );
    return result;
}

int autoJayHarness::configure( const std::string &contents )
{
    auto          path = m_directory / "settings.conf";
    std::ofstream output( path );
    output << contents;
    output.close();
    REQUIRE( config.readConfig( path ) == 0 );
    return loadConfigImpl( config );
}

autoJayHarness::imageT autoJayHarness::response( unsigned dm ) const
{
    imageT                      result( 44, 40 );
    const std::array<double, 4> px{ 10, 27, 17, 34 }, py{ 10, 13, 29, 31 }, amplitude{ 50, 75, 60, 90 };
    for( Eigen::Index y = 0; y < result.cols(); ++y )
        for( Eigen::Index x = 0; x < result.rows(); ++x )
        {
            double value = 0;
            for( size_t i = 0; i < px.size(); ++i )
            {
                double dx = x - px[i] - dm * 1.2, dy = y - py[i] + dm * 0.7;
                value += amplitude[i] * std::exp( -( dx * dx + dy * dy ) / ( 2 * 2.1 * 2.1 ) );
            }
            result( x, y ) = value;
        }
    return result;
}

void autoJayHarness::completeSequence()
{
    using step       = MagAOX::app::autoJayDetail::acquisition::step;
    uint64_t counter = 0;
    double   now     = monotonic();
    for( unsigned attempts = 0; attempts < 30 && m_acquisition.state() != step::complete; ++attempts )
    {
        REQUIRE( m_acquisition.state() != step::failed );
        if( m_acquisition.commandPending() )
        {
            auto        counters = triggerCounters();
            std::string error;
            REQUIRE( writeChannels( m_acquisition.command(), error ) );
            m_acquisition.commanded( counters, now );
            m_acquisition.poll( triggerCounters(), now );
            now += m_cycle.dmSleep;
        }
        imageT image = imageT::Constant( 44, 40, 500 );
        if( m_acquisition.state() == step::ncpc )
            image += response( 0 );
        if( m_acquisition.state() == step::tweeter )
            image += response( 1 );
        m_acquisition.frame( image, ++counter, now );
        now += 0.001;
    }
    REQUIRE( m_acquisition.state() == step::complete );
    m_lastFrameTime = monotonic();
}

void autoJayHarness::acknowledgment( double session, double counter, int status )
{
    numbers( "autoJay-ctrl",
             "processed",
             { { "session", session },
               { "counter", counter },
               { "status", status },
               { "applied", status == 1 ? 1 : 0 },
               { "target0", 50 },
               { "target1", 50 } } );
}

std::string autoJayHarness::calibrationRoot() const
{
    return m_calibDir + "/" + m_configName;
}
/// \endcond

/// Default requirements resolve exposure at capture and verify both operator-open loops.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay setup defaults capture stable camera settings", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::loadConfigImpl;
    MagAOX::app::autoJay::checkSetup;
    #endif
    // clang-format on
    autoJayHarness              app;
    std::string                 error;
    autoJayHarness::calibration resolved;
    REQUIRE( app.setupValid( true, error, &resolved ) );
    REQUIRE( resolved.setup.size() == 6 );
    REQUIRE( resolved.setup[0].device == "stagelosel" );
    REQUIRE( resolved.setup[0].value == "1" );
    REQUIRE( resolved.setup[1].device == "stageflowfs" );
    REQUIRE( resolved.setup[1].value == "32.0" );
    REQUIRE( resolved.setup.back().value != "*" );
    REQUIRE( std::stod( resolved.setup.back().value ) == Approx( 0.001 ) );
    REQUIRE( app.setupValid( false, error ) );
    app.numbers( "camflowfs", "exptime", { { "current", 0.002 } } );
    REQUIRE_FALSE( app.setupValid( false, error ) );
    REQUIRE( error.find( "numeric setup mismatch" ) != std::string::npos );
}

/// Adding mask requirements must retain both mandatory flowfs optical stages and the captured LOS position.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay custom mask rules retain required flowfs setup", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::loadConfigImpl;
    MagAOX::app::autoJay::checkSetup;
    #endif
    // clang-format on
    autoJayHarness app;
    REQUIRE( app.configure( "[setup]\ndevices=fwpupil\nproperties=filterName\nelements=bump\n"
                            "values=1\ntypes=2\ntolerances=0\n" ) == 0 );
    auto find = [&]( const std::string &device, const std::string &property )
    {
        return std::find_if( app.m_captureConfig.setup.begin(),
                             app.m_captureConfig.setup.end(),
                             [&]( const auto &rule ) { return rule.device == device && rule.property == property; } );
    };
    REQUIRE( find( "fwpupil", "filterName" ) != app.m_captureConfig.setup.end() );
    auto los = find( "stagelosel", "presetName" );
    REQUIRE( los != app.m_captureConfig.setup.end() );
    REQUIRE( los->element == "pupil" );
    REQUIRE( los->type == 2 );
    REQUIRE( los->value == "1" );
    auto position = find( "stagelosel", "position" );
    REQUIRE( position != app.m_captureConfig.setup.end() );
    REQUIRE( position->value == "*" );
    auto flowfs = find( "stageflowfs", "position" );
    REQUIRE( flowfs != app.m_captureConfig.setup.end() );
    REQUIRE( flowfs->element == "current" );
    REQUIRE( flowfs->type == 0 );
    REQUIRE( flowfs->value == "32.0" );
    app.toggle( "fwpupil", "filterName", "bump", true );
    app.text( "fwpupil", "fsm", "state", "READY" );
    std::string                 error;
    autoJayHarness::calibration captured;
    REQUIRE( app.setupValid( true, error, &captured ) );
    auto capturedLos =
        std::find_if( captured.setup.begin(),
                      captured.setup.end(),
                      []( const auto &rule ) { return rule.device == "stagelosel" && rule.property == "position"; } );
    REQUIRE( capturedLos != captured.setup.end() );
    REQUIRE( std::stod( capturedLos->value ) == Approx( 3.5 ) );
    app.numbers( "stageflowfs", "position", { { "current", 33 } } );
    REQUIRE_FALSE( app.setupValid( true, error ) );
}

/// Science-camera profiles must explicitly bind filter, beamsplitter, and focus requirements.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay rejects incomplete science-camera profiles", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::loadConfigImpl;
    #endif
    // clang-format on
    autoJayHarness                   app;
    int                              omitted = GENERATE( -1, 0, 1, 2 );
    const std::array<std::string, 3> devices{ "fwscind", "stagebs", "stagesci1" };
    const std::array<std::string, 3> properties{ "filterName", "position", "position" };
    const std::array<std::string, 3> elements{ "pupil", "current", "current" };
    const std::array<std::string, 3> values{ "1", "10.0", "15.0" };
    const std::array<std::string, 3> types{ "2", "0", "0" };
    std::array<std::string, 6>       fields;
    for( int i = 0; i < 3; ++i )
    {
        if( i == omitted )
            continue;
        const std::array<std::string, 6> row{ devices[i], properties[i], elements[i], values[i], types[i], "0.01" };
        for( size_t j = 0; j < fields.size(); ++j )
        {
            if( !fields[j].empty() )
                fields[j] += ',';
            fields[j] += row[j];
        }
    }
    std::string                      config = "[autoJay]\ncamera=camsci1\nprofile=science\n[setup]\n";
    const std::array<std::string, 6> keys{ "devices", "properties", "elements", "values", "types", "tolerances" };
    for( size_t i = 0; i < keys.size(); ++i )
        config += keys[i] + '=' + fields[i] + '\n';
    REQUIRE( ( app.configure( config ) == 0 ) == ( omitted < 0 ) );
}

/// Correction settling relaxes only the actuator motion check and retains required optical and loop states.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay monitors setup while an expected correction settles", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::checkSetup;
    #endif
    // clang-format on
    autoJayHarness app;
    app.numbers( "ttmperi", "axis1_voltage", { { "current", 49 }, { "target", 50 } } );
    auto voltage = app.observedProperty( "ttmperi.axis1_voltage" );
    voltage.setState( pcf::IndiProperty::Busy );
    app.deliver( voltage );
    std::string error;
    REQUIRE_FALSE( app.setupValid( false, error ) );
    error.clear();
    REQUIRE( app.setupValid( false, error, nullptr, true ) );
    app.toggle( "holoop", "loop_state", "toggle", true );
    REQUIRE_FALSE( app.setupValid( false, error, nullptr, true ) );
    REQUIRE( error.find( "loop" ) != std::string::npos );
}

/// Unknown, invalid, moving, or mismatched setup and a stale camera stream inhibit acquisition.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay rejects invalid setup before opening channels", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::begin;
    MagAOX::app::autoJay::checkSetup;
    #endif
    // clang-format on
    autoJayHarness app;
    SECTION( "AO loop closed" )
    {
        app.toggle( "holoop", "loop_state", "toggle", true );
    }
    SECTION( "alignment loop closed" )
    {
        app.toggle( "twAlign-camwfs-ctrl", "loop_state", "toggle", true );
    }
    SECTION( "LOS not pupil" )
    {
        app.toggle( "stagelosel", "presetName", "pupil", false );
    }
    SECTION( "wrong numeric stage position despite preset" )
    {
        app.numbers( "stageflowfs", "position", { { "current", 33 } } );
    }
    SECTION( "stage moving" )
    {
        auto value = app.observedProperty( "stageflowfs.position" );
        value.setState( pcf::IndiProperty::Busy );
        app.deliver( value );
    }
    SECTION( "stage not ready" )
    {
        app.text( "stageflowfs", "fsm", "state", "HOMING" );
    }
    SECTION( "missing device state" )
    {
        app.m_observed.erase( "stageflowfs.fsm" );
    }
    SECTION( "missing position element" )
    {
        app.numbers( "stageflowfs", "position", { { "target", 32 } } );
    }
    SECTION( "setup subscription awaiting its initial value" )
    {
        app.m_observed.at( "stageflowfs.position" ).value.reset();
    }
    SECTION( "setup explicitly reports an invalid state" )
    {
        auto value = app.observedProperty( "stageflowfs.position" );
        value.setState( pcf::IndiProperty::Alert );
        app.deliver( value );
    }
    SECTION( "stale camera" )
    {
        app.m_lastFrameTime -= 10;
    }
    SECTION( "missing ROI" )
    {
        app.m_observed.erase( "camflowfs.roi_region_state" );
    }
    SECTION( "metadata dimension mismatch" )
    {
        auto value = app.observedProperty( "camflowfs.roi_region_state" );
        value["w"] = 42;
        app.deliver( value );
    }
    SECTION( "binning changed" )
    {
        app.liveGeometry( 44, 40, 2 );
    }
    SECTION( "unknown units" )
    {
        auto value     = app.observedProperty( "camflowfs.roi_region_state" );
        value["units"] = 2;
        app.deliver( value );
    }
    SECTION( "actuator still settling" )
    {
        app.numbers( "ttmperi", "axis1_voltage", { { "current", 50 }, { "target", 51 } } );
    }
    SECTION( "malformed position" )
    {
        app.text( "stageflowfs", "position", "current", "32junk" );
    }
    REQUIRE_FALSE( app.begin( false ) );
    REQUIRE( app.m_openCount == 0 );
    REQUIRE_FALSE( app.m_active );
    REQUIRE( app.m_counter == 0 );
}

/// A stable INDI setup remains usable after the camera freshness interval passes.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay retains subscribed setup that has not changed", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::receive;
    MagAOX::app::autoJay::checkSetup;
    #endif
    // clang-format on
    autoJayHarness app;
    REQUIRE( app.configure( "[monitor]\nmaxAge=0.25\n" ) == 0 );
    std::string error;

    std::this_thread::sleep_for( std::chrono::milliseconds( 275 ) );
    REQUIRE_FALSE( app.setupValid( false, error ) );
    REQUIRE( error == "camera stream is stale" );

    // Only the camera resumes publishing. The setup subscriptions stay quiet.
    app.m_lastFrameTime = app.monotonic();
    REQUIRE( app.setupValid( false, error ) );
    app.numbers( "stageflowfs", "position", { { "current", 33 } } );
    REQUIRE_FALSE( app.setupValid( false, error ) );
    REQUIRE( error.find( "numeric setup mismatch" ) != std::string::npos );
}

/// The camera callback converts unsigned frames into the baseline stack and skips writes in progress.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay copies coherent camera pixels into the baseline stack", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::allocate;
    MagAOX::app::autoJay::processImage;
    #endif
    // clang-format on
    autoJayHarness app;
    REQUIRE( app.allocateCamera( IMAGESTRUCT_UINT16 ) == 0 );
    std::vector<uint16_t> pixels( 44 * 40, 40000 );
    pixels[44] = 65000;
    IMAGE_METADATA metadata{};
    metadata.cnt0 = 4;
    REQUIRE( app.cameraFrame( pixels.data(), metadata ) == 0 );
    REQUIRE( app.begin( false ) );

    const auto  before = app.triggerCounters();
    std::string error;
    REQUIRE( app.writeChannels( app.m_acquisition.command(), error ) );
    app.m_acquisition.commanded( before, app.monotonic() );
    app.m_acquisition.poll( app.triggerCounters(), app.monotonic() );

    metadata.cnt0  = 5;
    metadata.write = 1;
    REQUIRE( app.cameraFrame( pixels.data(), metadata ) == 0 );
    const auto &baseline = app.m_acquisition.stacks()[0];
    REQUIRE( baseline.count == 0 );

    metadata.write = 0;
    REQUIRE( app.cameraFrame( pixels.data(), metadata ) == 0 );
    REQUIRE( baseline.count == 1 );
    REQUIRE( baseline.firstCounter == 5 );
    REQUIRE( baseline.mean.rows() == 44 );
    REQUIRE( baseline.mean.cols() == 40 );
    REQUIRE( baseline.mean( 0, 0 ) == 40000 );
    REQUIRE( baseline.mean( 0, 1 ) == 65000 );
    REQUIRE( baseline.mean( 43, 39 ) == 40000 );
}

/// Shared live ROI extents may differ from capture while the sampling and stream metadata remain consistent.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay setup accepts a different shared live ROI", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::checkSetup;
    #endif
    // clang-format on
    autoJayHarness app;
    app.liveGeometry( 39, 35 );
    std::string error;
    REQUIRE( app.setupValid( false, error ) );
    REQUIRE( app.m_calibration->baseline.rows() == 44 );
}

/// A setup transition during channel preparation cancels the transaction and clears owned channels.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay cancels a stale begin snapshot", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::begin;
    MagAOX::app::autoJay::stop;
    #endif
    // clang-format on
    autoJayHarness app;
    app.m_changeOnOpen = true;
    REQUIRE_FALSE( app.begin( false ) );
    REQUIRE( app.m_openCount == 1 );
    REQUIRE( app.m_clearCount == 1 );
    REQUIRE_FALSE( app.m_open );
    REQUIRE_FALSE( app.m_active );
    REQUIRE( app.m_counter == 0 );
}

/// Finite source maps whose amplitude multiplication overflows must be rejected before channel ownership.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay rejects overflowing bootstrap commands before poking", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::begin;
    #endif
    // clang-format on
    autoJayHarness app;
    auto           image = app.m_captureConfig.dm[0].pattern;
    image( 1, 1 )        = 2;
    std::string error;
    REQUIRE( MagAOX::app::autoJayDetail::calibrationIO::writeImage( app.m_patternFiles[0], image, error ) );
    app.m_captureConfig.dm[0].amplitude = std::numeric_limits<float>::max();
    REQUIRE_FALSE( app.begin( true ) );
    REQUIRE( app.m_openCount == 0 );
    REQUIRE( app.m_commands.empty() );
}

/// Valid completed data advance one counter; stale generation, cancellation, and cleanup failure advance none.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay publishes only a complete current measurement", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::finish;
    MagAOX::app::autoJay::stop;
    #endif
    // clang-format on
    autoJayHarness app;
    app.toggle( "autoJay-ctrl", "loop_state", "toggle", true );
    REQUIRE( app.begin( false ) );
    app.completeSequence();
    bool accepted    = false;
    bool corrections = true;
    SECTION( "valid pair" )
    {
        accepted = true;
    }
    SECTION( "valid pair with corrections disabled" )
    {
        accepted                = true;
        corrections             = false;
        app.m_controllerEnabled = false;
        app.toggle( "autoJay-ctrl", "loop_state", "toggle", false );
    }
    SECTION( "setup changed and returned" )
    {
        app.numbers( "stageflowfs", "position", { { "current", 33 } } );
        app.numbers( "stageflowfs", "position", { { "current", 32 } } );
    }
    SECTION( "operator cancelled" )
    {
        pcf::IndiProperty request( pcf::IndiProperty::Switch, "autoJay-test", "stop" );
        request.add( pcf::IndiElement( "request" ) );
        request["request"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_stop( request ) == 0 );
    }
    SECTION( "cleanup failed" )
    {
        app.m_failClear = true;
    }
    app.finish();
    INFO( app.m_status );
    REQUIRE( app.m_counter == ( accepted ? 1 : 0 ) );
    REQUIRE( app.m_waitingControl == ( accepted && corrections ) );
    REQUIRE_FALSE( app.m_active );
    REQUIRE_FALSE( app.m_open );
    REQUIRE( app.m_clearCount >= 1 );
    if( !accepted )
        REQUIRE_FALSE( app.m_continuous );
}

/// First-reference capture saves the effective maps and references without producing a correction sample.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay captures its first reference without an existing calibration", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::begin;
    MagAOX::app::autoJay::finish;
    MagAOX::app::autoJay::loadPokeConfig;
    MagAOX::app::autoJay::resolvePattern;
    #endif
    // clang-format on
    autoJayHarness app;
    app.m_calibration.reset();
    autoJayHarness::imageT ncpc             = app.m_captureConfig.dm[0].pattern;
    autoJayHarness::imageT tweeter          = app.m_captureConfig.dm[1].pattern;
    float                  ncpcAmplitude    = 0.1F;
    float                  tweeterAmplitude = 0.1F;

    SECTION( "FITS maps" )
    {
    }

    SECTION( "Paired actuator coordinates with a relayed tweeter" )
    {
        REQUIRE( app.configure( "[ncpc]\npokeX=0 , 3\npokeY=2, 1\npokeAmp=-0.25\n"
                                "[tweeter]\ntrigger=\npokeX=2\npokeY=0\npokeAmp=0.125\n"
                                "[acquisition]\ndmSleep=0.05\n" ) == 0 );
        REQUIRE( app.m_captureConfig.dm[1].trigger.empty() );
        ncpc.setZero();
        ncpc( 0, 2 ) = 1;
        ncpc( 3, 1 ) = 1;
        tweeter.setZero();
        tweeter( 2, 0 )  = 1;
        ncpcAmplitude    = -0.25F;
        tweeterAmplitude = 0.125F;
    }

    REQUIRE( app.begin( true ) );
    app.completeSequence();
    app.finish();
    INFO( app.m_status );
    REQUIRE( app.m_calibration );
    REQUIRE( app.m_calibration->id.starts_with( "calibration__T" ) );
    REQUIRE( app.m_counter == 0 );
    REQUIRE_FALSE( app.m_waitingControl );
    REQUIRE_FALSE( app.m_active );
    autoJayHarness::calibration saved;
    std::string                 error;
    REQUIRE( MagAOX::app::autoJayDetail::latestCalibration(
        saved, app.calibrationRoot(), app.m_captureConfig.profile, error ) );
    REQUIRE( saved.id == app.m_calibration->id );
    REQUIRE( ( saved.dm[0].pattern == ncpc ).all() );
    REQUIRE( ( saved.dm[1].pattern == tweeter ).all() );
    REQUIRE( saved.dm[0].amplitude == ncpcAmplitude );
    REQUIRE( saved.dm[1].amplitude == tweeterAmplitude );
    REQUIRE( saved.dm[1].trigger == app.m_captureConfig.dm[1].trigger );
    REQUIRE( saved.dmSleep == app.m_captureConfig.dmSleep );
    REQUIRE( ( saved.dm[0].reference - app.response( 0 ) ).abs().maxCoeff() < 0.0001F );
}

/// Reject ambiguous or malformed indexed capture settings before starting acquisition.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay rejects invalid actuator list configuration", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::loadPokeConfig;
    #endif
    // clang-format on
    autoJayHarness    app;
    const std::string settings = GENERATE( "pokeX=1,2\npokeY=1\npokeAmp=0.25\n",
                                           "pokeX=1\npokeAmp=0.25\n",
                                           "pokeX=1\npokeY=1\n",
                                           "pokeX=1junk\npokeY=1\npokeAmp=0.25\n",
                                           "pokeX=1\npokeY=-1\npokeAmp=0.25\n",
                                           "pokeX=1\npokeY=1\npokeAmp=nan\n",
                                           "pokeX=1\npokeY=1\npokeAmp=0.25\npattern=other.fits\n" );
    REQUIRE( app.configure( "[ncpc]\n" + settings ) < 0 );
}

/// A capture with an out-of-range actuator must fail before any commands are sent.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay checks actuator coordinates against the DM stream", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::begin;
    MagAOX::app::autoJay::resolvePattern;
    #endif
    // clang-format on
    autoJayHarness app;
    REQUIRE( app.configure( "[ncpc]\npokeX=1\npokeY=4\npokeAmp=0.25\n" ) == 0 );
    REQUIRE_FALSE( app.begin( true ) );
    REQUIRE( app.m_status.find( "outside DM stream geometry" ) != std::string::npos );
    REQUIRE_FALSE( app.m_active );
    REQUIRE( app.m_commands.empty() );
}

/// An explicitly failed recall cannot leave an older calibration available for a subsequent measurement.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay failed recall inhibits the previously loaded calibration", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::recall;
    MagAOX::app::autoJay::begin;
    #endif
    // clang-format on
    autoJayHarness app;
    REQUIRE( app.m_calibration );
    SECTION( "no saved calibration" )
    {
    }
    SECTION( "newest bundle identifies a different camera" )
    {
        auto        previous = *app.m_calibration;
        std::string error;
        REQUIRE( MagAOX::app::autoJayDetail::saveCalibration( previous, app.calibrationRoot(), error ) );
        auto incompatible   = previous;
        incompatible.camera = "camsci1";
        REQUIRE( MagAOX::app::autoJayDetail::saveCalibration( incompatible, app.calibrationRoot(), error ) );
    }
    REQUIRE_FALSE( app.recall() );
    REQUIRE_FALSE( app.begin( false ) );
    REQUIRE( app.m_openCount == 0 );
}

/// Counter/session acknowledgment and settled voltage readback bound the interval between measurements.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay waits for its own acknowledged correction", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::controlSettled;
    #endif
    // clang-format on
    autoJayHarness app;
    app.m_counter         = 3;
    app.m_controlDeadline = app.monotonic() + 5;
    app.m_controlSettle   = 0;
    std::string error;
    REQUIRE_FALSE( app.controlSettled( error ) );
    app.acknowledgment( app.m_session + 1, 3, 1 );
    REQUIRE_FALSE( app.controlSettled( error ) );
    app.acknowledgment( app.m_session, 2, 1 );
    REQUIRE_FALSE( app.controlSettled( error ) );
    app.acknowledgment( app.m_session, 3, 1 );
    REQUIRE( app.controlSettled( error ) );
    app.numbers( "ttmperi", "axis1_voltage", { { "current", 49 }, { "target", 50 } } );
    REQUIRE_FALSE( app.controlSettled( error ) );
    app.m_controlDeadline = app.monotonic() - 1;
    REQUIRE_FALSE( app.controlSettled( error ) );
    REQUIRE( error.find( "timed out" ) != std::string::npos );
}

/// Stop clears fake owned channels, cancels continuous operation, and leaves the accepted counter unchanged.
/** \ingroup autoJay_unit_test */
TEST_CASE( "autoJay stop cancels acquisition and reports cleanup failure", "[autoJay][app]" )
{
    // clang-format off
    #ifdef AUTOJAY_TEST_DOXYGEN_REF
    MagAOX::app::autoJay::stop;
    #endif
    // clang-format on
    autoJayHarness app;
    REQUIRE( app.begin( false ) );
    app.m_continuous = true;
    app.m_failClear  = true;
    app.stop( "operator stop" );
    REQUIRE_FALSE( app.m_active );
    REQUIRE_FALSE( app.m_continuous );
    REQUIRE_FALSE( app.m_waitingControl );
    REQUIRE( app.m_clearCount == 1 );
    REQUIRE( app.m_counter == 0 );
    REQUIRE( app.m_status.find( "injected cleanup failure" ) != std::string::npos );
}

} // namespace autoJayTest
} // namespace libXWCTest
