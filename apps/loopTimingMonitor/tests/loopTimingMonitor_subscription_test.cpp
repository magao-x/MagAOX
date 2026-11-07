/** \file loopTimingMonitor_subscription_test.cpp
 * \brief Offline regression for persistent INDI subscriptions without periodic state polling.
 * \ingroup loopTimingMonitor_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../loopTimingMonitor.hpp"

#include <filesystem>

namespace libXWCTest
{
namespace loopTimingMonitorTest
{

/// \cond LOOP_TIMING_SUBSCRIPTION_TEST_HELPERS
/// Keep every FIFO, telemetry file and possible stream lookup inside a private local directory.
class SubscriptionDirectory
{
    std::string m_path; ///< Owned temporary directory, removed after the application is destroyed.

    std::string m_previous; ///< Original stream-directory environment value when present.

    bool m_hadPrevious{ false }; ///< Restore an absent variable differently from an empty value.

  public:
    /// Create and select an isolated stream directory without opening any stream.
    SubscriptionDirectory();

    /// Restore the environment and remove only this fixture's files.
    ~SubscriptionDirectory();

    /// Return the private directory used by the unactivated driver and telemetry logger.
    const std::string &path() const;
};

SubscriptionDirectory::SubscriptionDirectory()
{
    if( const char *previous = getenv( "MILK_SHM_DIR" ) )
    {
        m_previous    = previous;
        m_hadPrevious = true;
    }

    char       pattern[] = "/tmp/loop-timing-subscriptions-XXXXXX";
    const auto directory = mkdtemp( pattern );
    REQUIRE( directory != nullptr );
    m_path = directory;
    REQUIRE( setenv( "MILK_SHM_DIR", m_path.c_str(), 1 ) == 0 );
}

SubscriptionDirectory::~SubscriptionDirectory()
{
    if( m_hadPrevious )
    {
        setenv( "MILK_SHM_DIR", m_previous.c_str(), 1 );
    }
    else
    {
        unsetenv( "MILK_SHM_DIR" );
    }

    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}

const std::string &SubscriptionDirectory::path() const
{
    return m_path;
}

/// Serialize real production messages locally without activating INDI or creating an outgoing client.
class SubscriptionDriver : public MagAOX::app::indiDriver<MagAOX::app::MagAOXApp<true>>
{
    int m_input{ -1 }; ///< Private input descriptor not released by an unactivated production driver.

    mutable std::vector<std::string> m_xml; ///< Messages produced synchronously by startup and app logic.

  public:
    /// Open only the owning fixture's private FIFOs and enable response serialization.
    explicit SubscriptionDriver( MagAOX::app::MagAOXApp<true> *parent /**< [in] Owner of the private FIFO names. */ );

    /// Release the unused input descriptor before the driver closes its output.
    ~SubscriptionDriver();

    /// Capture XML instead of writing a FIFO or contacting another driver.
    void sendXml( const std::string &xml /**< [in] Production-serialized property message. */ ) const override;

    /// Count requests independently of expected ordinary read-only property publications.
    size_t requests() const;

    /// Discard messages after establishing the subscription baseline.
    void clear();
};

SubscriptionDriver::SubscriptionDriver( MagAOX::app::MagAOXApp<true> *parent )
    : indiDriver<MagAOX::app::MagAOXApp<true>>( parent, "loop-timing-subscription-test", "0", "1.7" )
{
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( stat( parent->driverInName().c_str(), &input ) == 0 );

    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        const int   descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};

        if( fstat( descriptor, &candidate ) == 0 && candidate.st_dev == input.st_dev &&
            candidate.st_ino == input.st_ino )
        {
            REQUIRE( m_input == -1 );
            m_input = descriptor;
        }
    }

    REQUIRE( m_input >= 0 );
}

SubscriptionDriver::~SubscriptionDriver()
{
    if( m_input >= 0 )
    {
        ::close( m_input );
    }

    setInputFd( -1 );
}

void SubscriptionDriver::sendXml( const std::string &xml ) const
{
    m_xml.push_back( xml );
}

size_t SubscriptionDriver::requests() const
{
    return std::count_if( m_xml.begin(),
                          m_xml.end(),
                          []( const auto &xml )
                          {
                              return xml.find( "<getProperties" ) != std::string::npos;
                          } );
}

void SubscriptionDriver::clear()
{
    m_xml.clear();
}

/// Run the real passive monitor lifecycle with isolated names and a local serialization-only driver.
class SubscriptionHarness : public MagAOX::app::loopTimingMonitor
{
  public:
    using MagAOXApp<true>::sendGetPropertySetList;

    /// Configure telemetry, nonexistent streams and FIFO names before any application thread starts.
    explicit SubscriptionHarness( const std::string &directory /**< [in] Existing private fixture directory. */ );

    /// Stop the polling and telemetry owners before member/base destruction.
    ~SubscriptionHarness() noexcept;

    /// Return the unactivated local driver's recorded messages.
    SubscriptionDriver &driver();

    /// Deliver a complete source state through the real base-class subscription dispatcher.
    void source( const std::string &device /**< [in] One configured local device name. */,
                 bool               affirmative /**< [in] OPERATING or On when true; READY or Off otherwise. */ );

    /// Check that all three registrations remain present and resolved after normal retries.
    bool subscriptionsResolved();

    /// Return the operating-state gate published by the real source callbacks.
    bool gateEnabled() const;
};

SubscriptionHarness::SubscriptionHarness( const std::string &directory )
{
    m_configName   = "loop-timing-subscription-test";
    m_cameraDevice = "local-camera";
    m_dmDevice     = "local-dm";
    m_loopDevice   = "local-loop";
    m_cameraStream = "uncreated-local-camera";
    m_sdkStream    = "uncreated-local-sdk";
    m_threadPrio   = 0;
    m_cpuset.clear();
    setupConfig();
    REQUIRE( loadConfigImpl( config ) == 0 );
    m_tel.logPath( directory );
    m_tel.logName( m_configName );
    createROIndiText( m_indiP_state, "fsm", "state" );
    m_driverInName   = directory + "/input";
    m_driverOutName  = directory + "/output";
    m_driverCtrlName = directory + "/control";

    for( const auto &name : { m_driverInName, m_driverOutName, m_driverCtrlName } )
    {
        REQUIRE( mkfifo( name.c_str(), 0600 ) == 0 );
    }

    m_indiDriver = new SubscriptionDriver( this );
    REQUIRE( m_indiDriver->good() );
}

SubscriptionHarness::~SubscriptionHarness() noexcept
{
    appShutdown();
}

SubscriptionDriver &SubscriptionHarness::driver()
{
    return *static_cast<SubscriptionDriver *>( m_indiDriver );
}

void SubscriptionHarness::source( const std::string &device, bool affirmative )
{
    const bool        loop = device == m_loopDevice;
    pcf::IndiProperty property( loop ? pcf::IndiProperty::Switch : pcf::IndiProperty::Text );
    property.setDevice( device );
    property.setName( loop ? "loop_state" : "fsm" );
    property.add( pcf::IndiElement( loop ? "toggle" : "state" ) );

    if( loop )
    {
        property["toggle"].setSwitchState( affirmative ? pcf::IndiElement::On : pcf::IndiElement::Off );
    }
    else
    {
        property["state"] = affirmative ? "OPERATING" : "READY";
    }

    handleSetProperty( property );
}

bool SubscriptionHarness::subscriptionsResolved()
{
    std::lock_guard<std::mutex> lock( m_indiCallBackMutex );

    if( m_indiSetCallBacks.size() != 3 )
    {
        return false;
    }

    for( const auto &entry : m_indiSetCallBacks )
    {
        if( !entry.second.m_defReceived || !entry.second.property ||
            entry.first != entry.second.property->createUniqueKey() )
        {
            return false;
        }
    }

    return m_allDefsReceived;
}

bool SubscriptionHarness::gateEnabled() const
{
    return m_gateEnabled.load();
}
/// \endcond

/** \brief Resolve real subscription registrations once, then retain callback updates without app-level polling.
 * \ingroup loopTimingMonitor_unit_test
 */
TEST_CASE( "loop timing keeps resolved INDI subscriptions without periodic getProperties",
           "[loopTimingMonitor][subscriptions]" )
{
    SubscriptionDirectory directory;
    SubscriptionHarness   app( directory.path() );
    REQUIRE( app.appStartup() == 0 );
    app.driver().clear();
    app.sendGetPropertySetList( false );
    REQUIRE( app.driver().requests() == 3 );
    app.source( "local-camera", true );
    app.source( "local-dm", true );
    app.source( "local-loop", true );
    app.sendGetPropertySetList( false );
    REQUIRE( app.subscriptionsResolved() );
    REQUIRE( app.gateEnabled() );
    app.driver().clear();

    for( unsigned iteration = 0; iteration < 3; ++iteration )
    {
        REQUIRE( app.appLogic() == 0 );
        app.sendGetPropertySetList( false );
        CHECK( app.driver().requests() == 0 );
        CHECK( app.subscriptionsResolved() );
        CHECK( app.gateEnabled() );
    }

    app.source( "local-loop", false );
    REQUIRE_FALSE( app.gateEnabled() );
    REQUIRE( app.appLogic() == 0 );
    app.sendGetPropertySetList( false );
    CHECK( app.driver().requests() == 0 );
    CHECK( app.subscriptionsResolved() );

    // clang-format off
    #ifdef LOOPTIMINGMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::loopTimingMonitor::appStartup;
    MagAOX::app::loopTimingMonitor::appLogic;
    MagAOX::app::loopTimingMonitor::setCallBack_source;
    MagAOX::app::MagAOXApp<true>::handleSetProperty;
    MagAOX::app::MagAOXApp<true>::sendGetPropertySetList;
    #endif
    // clang-format on
}

} // namespace loopTimingMonitorTest
} // namespace libXWCTest
