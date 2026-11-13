/** \file outletAppTest.hpp
 * \brief Shared offline transport, telemetry, and failure injection for outlet application tests.
 */
#ifndef tests_outletAppTest_hpp
#define tests_outletAppTest_hpp
#include "testXWC.hpp"
#include "../libMagAOX/libMagAOX.hpp"

#include <array>
#include <deque>
#include <filesystem>
#include <fcntl.h>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace outletHarness
{
/// Captured application diagnostic.
struct Log
{
    /// Actual call-site priority.
    flatlogs::logPrioT m_priority;
    /// Message formatted by the production logger type.
    std::string m_message;
};

/// Captured actual serialized telemetry payload.
struct Record
{
    /// Actual telemetry event code.
    flatlogs::eventCodeT m_code;
    /// FlatBuffer payload created by production recording code.
    std::vector<uint8_t> m_payload;
};

/// Resettable offline failures and captured traffic.
struct Faults
{
    /// Application diagnostics.
    std::vector<Log> m_logs;
    /// Serialized telemetry records.
    std::vector<Record> m_records;
    /// Registration call count.
    unsigned m_registrations{ 0 };
    /// Registration call to fail; zero disables the fault.
    unsigned m_failRegistration{ 0 };
    /// Telemeter setup/load/startup/logic/shutdown results.
    std::array<int, 5> m_telemResults{};
    /// Result from creating an actual telemetry payload.
    int m_recordResult{ 0 };
    /// Whether interval scheduling should force records.
    bool m_due{ false };
    /// Number of outgoing NewProperty calls.
    unsigned m_sends{ 0 };
    /// Outgoing call to fail; zero disables the fault.
    unsigned m_failSend{ 0 };
    /// Scripted production telnet return values.
    std::deque<int> m_transportResults;
    /// Commands received by the test telnet transport.
    std::vector<std::string> m_commands;
    /// Scripted devstatus text.
    std::string m_status;
    /// Last login prompt configured by the production app.
    std::string m_loginPrompt;
    /// Last CLI prompt configured by the production app.
    std::string m_prompt;
    /// I/O configuration result applied after real loading.
    int m_ioLoad{ 0 };
};
/// Current suite's offline capture and failure state.
inline Faults g_faults;

/// Fail one selected registration call.
inline bool failRegistration()
{
    return ++g_faults.m_registrations == g_faults.m_failRegistration;
}

/// Consume the next offline transport return value.
inline int transportResult()
{
    if( g_faults.m_transportResults.empty() )
        return 0;
    int result = g_faults.m_transportResults.front();
    g_faults.m_transportResults.pop_front();
    return result;
}

/// Own and clean a private temporary directory.
struct Directory
{
    /// Isolated state and FIFO root.
    std::string m_path;
    /// Allocate a directory below /tmp.
    Directory();
    /// Delete only this fixture's files.
    ~Directory();
};
inline Directory::Directory()
{
    char  path[] = "/tmp/outlet-app-XXXXXX";
    auto *result = ::mkdtemp( path );
    REQUIRE( result != nullptr );
    m_path = result;
}
inline Directory::~Directory()
{
    std::error_code error;
    std::filesystem::remove_all( m_path, error );
}
} // namespace outletHarness

namespace MagAOX
{
namespace app
{
/// Real app base with captured logging and precise registration failures.
template <bool useINDI = true>
class outletTestApp : public MagAOXApp<useINDI>
{
  public:
    /// Allow the real telemeter configuration implementation to read the app name.
    using MagAOXApp<useINDI>::m_configName;
    /// Preserve standard registration overloads outside the fault-injected signatures.
    using MagAOXApp<useINDI>::registerIndiPropertyNew;
    /// Exact INDI callback signature.
    typedef int ( *Callback )( void *, const pcf::IndiProperty & );
    /// Construct the real base after suppressing its startup logger.
    outletTestApp( const std::string &sha /**< [in] revision */, bool modified /**< [in] worktree state */ );
    /// Suppress process logging before the real base constructs.
    static const std::string &quiet( const std::string &sha /**< [in] revision */ );
    /// Capture a production diagnostic using its real message formatter.
    template <class logT, int retval = 0>
    static int log( const typename logT::messageT &message /**< [in] payload */,
                    logPrioT                       priority = logPrio::LOG_DEFAULT /**< [in] requested priority */ );
    /// Capture a default message.
    template <class logT, int retval = 0>
    static int log( logPrioT priority = logPrio::LOG_DEFAULT /**< [in] requested priority */ );
    /// Fault-inject a fully initialized New-property registration.
    int registerIndiPropertyNew( pcf::IndiProperty &property /**< [in/out] property */,
                                 Callback           callback /**< [in] callback */ );
    /// Fault-inject a New-property registration which also constructs the property.
    int registerIndiPropertyNew( pcf::IndiProperty                          &property /**< [out] property */,
                                 const std::string                          &name /**< [in] property name */,
                                 const pcf::IndiProperty::Type              &type /**< [in] property type */,
                                 const pcf::IndiProperty::PropertyPermType  &perm /**< [in] permission */,
                                 const pcf::IndiProperty::PropertyStateType &state /**< [in] initial state */,
                                 Callback                                    callback /**< [in] callback */ );
    /// Fault-inject a read-only registration.
    int registerIndiPropertyReadOnly( pcf::IndiProperty &property /**< [in/out] property */ );
    /// Fault-inject a stable source subscription.
    int registerIndiPropertySet( pcf::IndiProperty &property /**< [out] property */,
                                 const std::string &device /**< [in] device */,
                                 const std::string &name /**< [in] property */,
                                 Callback           callback /**< [in] callback */ );
};
template <bool useINDI>
outletTestApp<useINDI>::outletTestApp( const std::string &sha, bool modified )
    : MagAOXApp<useINDI>( quiet( sha ), modified )
{
}
template <bool useINDI>
const std::string &outletTestApp<useINDI>::quiet( const std::string &sha )
{
    MagAOXApp<useINDI>::m_log.m_logLevel = logPrio::LOG_EMERGENCY;
    return sha;
}
template <bool useINDI>
template <class logT, int retval>
int outletTestApp<useINDI>::log( const typename logT::messageT &message, logPrioT priority )
{
    if( priority == logPrio::LOG_DEFAULT )
        priority = logT::defaultLevel;
    outletHarness::g_faults.m_logs.push_back(
        { priority, logT::msgString( message.builder.GetBufferPointer(), message.builder.GetSize() ) } );
    return retval;
}
template <bool useINDI>
template <class logT, int retval>
int outletTestApp<useINDI>::log( logPrioT priority )
{
    return log<logT, retval>( typename logT::messageT(), priority );
}
template <bool useINDI>
int outletTestApp<useINDI>::registerIndiPropertyNew( pcf::IndiProperty &property, Callback callback )
{
    if( outletHarness::failRegistration() )
        return -1;
    return MagAOXApp<useINDI>::registerIndiPropertyNew( property, callback );
}
template <bool useINDI>
int outletTestApp<useINDI>::registerIndiPropertyNew( pcf::IndiProperty                          &property,
                                                     const std::string                          &name,
                                                     const pcf::IndiProperty::Type              &type,
                                                     const pcf::IndiProperty::PropertyPermType  &perm,
                                                     const pcf::IndiProperty::PropertyStateType &state,
                                                     Callback                                    callback )
{
    if( outletHarness::failRegistration() )
        return -1;
    return MagAOXApp<useINDI>::registerIndiPropertyNew( property, name, type, perm, state, callback );
}
template <bool useINDI>
int outletTestApp<useINDI>::registerIndiPropertyReadOnly( pcf::IndiProperty &property )
{
    if( outletHarness::failRegistration() )
        return -1;
    return MagAOXApp<useINDI>::registerIndiPropertyReadOnly( property );
}
template <bool useINDI>
int outletTestApp<useINDI>::registerIndiPropertySet( pcf::IndiProperty &property,
                                                     const std::string &device,
                                                     const std::string &name,
                                                     Callback           callback )
{
    if( outletHarness::failRegistration() )
        return -1;
    return MagAOXApp<useINDI>::registerIndiPropertySet( property, device, name, callback );
}
namespace dev
{
/// Threadless telemetry boundary retaining actual configuration and payload creation.
template <class derivedT>
class outletTestTelemeter : public telemeter<derivedT>
{
  public:
    /// Load real telemetry configuration unless this call is selected to fail.
    int setupConfig( mx::app::appConfigurator &config /**< [in/out] app configuration */ );
    /// Load real telemetry configuration unless this call is selected to fail.
    int loadConfig( mx::app::appConfigurator &config /**< [in] app configuration */ );
    /// Start a threadless sink.
    int appStartup();
    /// Execute the controller's real scheduling dispatch.
    int appLogic();
    /// Stop the threadless sink.
    int appShutdown();
    /// Force each actual recordTelem overload when a deadline is injected.
    template <class... types>
    int checkRecordTimes( const types &...type /**< [in] type selectors */ );
    /// Capture the real FlatBuffer message.
    template <class telT>
    int telem( const typename telT::messageT &message /**< [in] payload */ );
};
template <class derivedT>
int outletTestTelemeter<derivedT>::setupConfig( mx::app::appConfigurator &config )
{
    if( outletHarness::g_faults.m_telemResults[0] < 0 )
        return -1;
    return telemeter<derivedT>::setupConfig( config );
}
template <class derivedT>
int outletTestTelemeter<derivedT>::loadConfig( mx::app::appConfigurator &config )
{
    if( outletHarness::g_faults.m_telemResults[1] < 0 )
        return -1;
    return telemeter<derivedT>::loadConfig( config );
}
template <class derivedT>
int outletTestTelemeter<derivedT>::appStartup()
{
    return outletHarness::g_faults.m_telemResults[2];
}
template <class derivedT>
int outletTestTelemeter<derivedT>::appLogic()
{
    if( outletHarness::g_faults.m_telemResults[3] < 0 )
        return -1;
    return static_cast<derivedT *>( this )->checkRecordTimes();
}
template <class derivedT>
int outletTestTelemeter<derivedT>::appShutdown()
{
    return outletHarness::g_faults.m_telemResults[4];
}
template <class derivedT>
template <class... types>
int outletTestTelemeter<derivedT>::checkRecordTimes( const types &...type )
{
    if( !outletHarness::g_faults.m_due )
        return 0;
    return ( static_cast<derivedT *>( this )->recordTelem( &type ) + ... );
}
template <class derivedT>
template <class telT>
int outletTestTelemeter<derivedT>::telem( const typename telT::messageT &message )
{
    if( outletHarness::g_faults.m_recordResult < 0 )
        return -1;
    auto *begin = message.builder.GetBufferPointer();
    outletHarness::g_faults.m_records.push_back( { telT::eventCode, { begin, begin + message.builder.GetSize() } } );
    return 0;
}
/// Real I/O configuration with a selected failure result.
class outletTestIODevice : public ioDevice
{
  public:
    /// Preserve timeout parsing before returning an injected failure.
    int loadConfig( mx::app::appConfigurator &config /**< [in] app configuration */ );
};
inline int outletTestIODevice::loadConfig( mx::app::appConfigurator &config )
{
    int rv = ioDevice::loadConfig( config );
    return rv < 0 ? rv : outletHarness::g_faults.m_ioLoad;
}
} // namespace dev
} // namespace app
namespace tty
{
/// Scripted telnet boundary without sockets or device access.
struct outletTestTelnet
{
    /// Production-selected username prompt.
    std::string m_usernamePrompt;
    /// Production-selected command prompt.
    std::string m_prompt;
    /// Response returned to the production status parser.
    std::string m_strRead;
    /// Return the next connection outcome without network access.
    int connect( const std::string &address /**< [in] ignored address */,
                 const std::string &port /**< [in] ignored port */ );
    /// Capture the production prompt and consume a login outcome.
    int login( const std::string &user /**< [in] ignored username */,
               const std::string &password /**< [in] ignored password */ );
    /// Capture a production command and return scripted status/results.
    int writeRead( const std::string &command /**< [in] actual wire command */,
                   bool               echo /**< [in] ignored echo policy */,
                   int                writeTimeout /**< [in] ignored timeout */,
                   int                readTimeout /**< [in] ignored timeout */ );
    /// Consume a scripted re-read outcome.
    int read( int timeout /**< [in] ignored timeout */, bool echo /**< [in] ignored echo policy */ );
};
inline int outletTestTelnet::connect( const std::string &, const std::string & )
{
    return outletHarness::transportResult();
}
inline int outletTestTelnet::login( const std::string &, const std::string & )
{
    outletHarness::g_faults.m_loginPrompt = m_usernamePrompt;
    return outletHarness::transportResult();
}
inline int outletTestTelnet::writeRead( const std::string &command, bool, int, int )
{
    outletHarness::g_faults.m_commands.push_back( command );
    m_strRead                        = outletHarness::g_faults.m_status;
    outletHarness::g_faults.m_prompt = m_prompt;
    return outletHarness::transportResult();
}
inline int outletTestTelnet::read( int, bool )
{
    return outletHarness::transportResult();
}
} // namespace tty
} // namespace MagAOX

namespace outletHarness
{
/// Real INDI driver using private FIFOs; New messages use the production XML formatter.
class Driver : public MagAOX::app::indiDriver<MagAOX::app::MagAOXApp<true>>
{
  public:
    /// Construct without activating a receive thread or an outgoing TCP client.
    Driver( MagAOX::app::MagAOXApp<true> *parent /**< [in] fixture app */ );
    /// Explicitly release the unactivated input descriptor.
    ~Driver();
    /// Serialize outgoing commands to the private FIFO instead of a TCP client.
    int sendNewProperty( const pcf::IndiProperty &property /**< [in] actual production command */ ) override;
    /// Keep publications in memory while real command traffic uses the private FIFO.
    void sendXml( const std::string &xml /**< [in] production-formatted XML */ ) const override;
    /// Drain captured publications for the same real XML parser used for commands.
    std::string takePublished();

  private:
    /// Protect transcript capture when the app logic and callbacks publish concurrently.
    mutable std::mutex m_captureMutex;
    /// Non-command XML, preventing a bounded private FIFO from blocking state-update tests.
    mutable std::string m_published;
    /// Owned input descriptor, which the shared unactivated driver does not release.
    int m_input{ -1 };
};
inline Driver::Driver( MagAOX::app::MagAOXApp<true> *parent ) : indiDriver( parent, "outlet-test", "0", "1.7" )
{
    enableResponseMode( true );
    struct stat input{};
    REQUIRE( ::stat( parent->driverInName().c_str(), &input ) == 0 );
    for( const auto &entry : std::filesystem::directory_iterator( "/proc/self/fd" ) )
    {
        int         descriptor = std::stoi( entry.path().filename() );
        struct stat candidate{};
        if( ::fstat( descriptor, &candidate ) == 0 && candidate.st_ino == input.st_ino &&
            candidate.st_dev == input.st_dev )
            m_input = descriptor;
    }
}
inline Driver::~Driver()
{
    if( m_input >= 0 )
        ::close( m_input );
    setInputFd( -1 );
}
inline int Driver::sendNewProperty( const pcf::IndiProperty &property )
{
    if( ++g_faults.m_sends == g_faults.m_failSend )
        return -1;
    pcf::IndiXmlParser formatter( pcf::IndiMessage( pcf::IndiMessage::NewProperty, property ), "1.7" );
    sendXml( formatter.createXmlString() );
    return 0;
}

inline void Driver::sendXml( const std::string &xml ) const
{
    std::lock_guard<std::mutex> lock( m_captureMutex );
    if( xml.starts_with( "<new" ) )
        pcf::IndiConnection::sendXml( xml );
    else
        m_published += xml;
}
inline std::string Driver::takePublished()
{
    std::lock_guard<std::mutex> lock( m_captureMutex );
    std::string                 xml;
    xml.swap( m_published );
    return xml;
}

/// Generic sequential app fixture owning isolated configuration and private FIFOs.
template <class App>
class Controller : public App
{
  public:
    /// Expose the real configurator for configuration assertions.
    using App::config;
    /// Expose shutdown for configuration-error assertions.
    using App::m_shutdown;
    /// Expose INDI mutex for contention tests.
    using App::m_indiMutex;
    /// Expose registered callback maps for interface assertions.
    using App::m_indiNewCallBacks;
    using App::m_indiSetCallBacks;
    /// Own all temporary fixture artifacts.
    Directory m_directory;
    /// Read end of the private outgoing FIFO.
    int m_reader{ -1 };
    /// Set the real app's paths without running process setup.
    Controller();
    /// Stop private transport before removing fixture artifacts.
    ~Controller();
    /// Load actual INI text through the app's real configurator.
    void configText( const std::string &text /**< [in] INI contents */ );
    /// Attach a private FIFO driver with no network access.
    void driver();
    /// Parse all actual messages currently in the fixture FIFO.
    std::vector<pcf::IndiMessage> messages();
};
template <class App>
Controller<App>::Controller()
{
    this->m_configName = "test-pdu";
    this->m_basePath   = m_directory.m_path;
    this->m_configDir  = m_directory.m_path;
}
template <class App>
Controller<App>::~Controller()
{
    delete this->m_indiDriver;
    this->m_indiDriver = nullptr;
    if( m_reader >= 0 )
        ::close( m_reader );
}
template <class App>
void Controller<App>::configText( const std::string &text )
{
    this->setupConfig();
    auto          path = m_directory.m_path + "/config.conf";
    std::ofstream file( path );
    file << text;
    file.close();
    REQUIRE( this->config.readConfig( path ) == 0 );
}
template <class App>
void Controller<App>::driver()
{
    REQUIRE( this->MagAOX::app::MagAOXApp<true>::registerIndiPropertyNew( this->m_indiP_state,
                                                                          "fsm",
                                                                          pcf::IndiProperty::Text,
                                                                          pcf::IndiProperty::ReadOnly,
                                                                          pcf::IndiProperty::Idle,
                                                                          nullptr ) == 0 );
    this->m_indiP_state.add( pcf::IndiElement( "state" ) );
    this->m_driverInName   = m_directory.m_path + "/input";
    this->m_driverOutName  = m_directory.m_path + "/output";
    this->m_driverCtrlName = m_directory.m_path + "/control";
    for( const auto &path : { this->m_driverInName, this->m_driverOutName, this->m_driverCtrlName } )
        REQUIRE( ::mkfifo( path.c_str(), 0600 ) == 0 );
    m_reader = ::open( this->m_driverOutName.c_str(), O_RDONLY | O_NONBLOCK );
    REQUIRE( m_reader >= 0 );
    this->m_indiDriver = new Driver( this );
    REQUIRE( this->m_indiDriver->good() );
}
template <class App>
std::vector<pcf::IndiMessage> Controller<App>::messages()
{
    std::string xml = static_cast<Driver *>( this->m_indiDriver )->takePublished();
    char        buffer[4096];
    ssize_t     count;
    while( ( count = ::read( m_reader, buffer, sizeof( buffer ) ) ) > 0 )
        xml.append( buffer, count );
    pcf::IndiXmlParser            parser( "1.7" );
    std::string                   error;
    std::vector<pcf::IndiMessage> result;
    for( char byte : xml )
    {
        parser.parseXml( &byte, 1, error );
        REQUIRE( error.empty() );
        if( parser.getState() == pcf::IndiXmlParser::CompleteState )
        {
            result.push_back( parser.createIndiMessage() );
            parser.clear();
        }
    }
    return result;
}

/// Run a check while a separate thread owns the selected mutex.
template <class Callback>
void contended( std::mutex &mutex /**< [in] mutex to hold */, Callback check /**< [in] check run while contended */ )
{
    std::atomic<bool> ready{ false };
    std::jthread      owner(
        [&]( std::stop_token stop )
        {
            std::lock_guard<std::mutex> lock( mutex );
            ready.store( true );
            while( !stop.stop_requested() )
                std::this_thread::yield();
        } );
    while( !ready.load() )
        std::this_thread::yield();
    check();
}

/// Construct a normal source/channel property for callback tests.
inline pcf::IndiProperty property( const std::string &device /**< [in] device */,
                                   const std::string &name /**< [in] property name */,
                                   const std::string &element /**< [in] element name */,
                                   const std::string &value /**< [in] text value */ )
{
    pcf::IndiProperty result( pcf::IndiProperty::Text, device, name );
    result.add( pcf::IndiElement( element, value ) );
    return result;
}
} // namespace outletHarness
/// \endcond
#endif // tests_outletAppTest_hpp
