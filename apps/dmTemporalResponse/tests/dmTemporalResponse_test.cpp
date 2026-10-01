/** \file dmTemporalResponse_test.cpp
 * \brief Catch2 tests for the dmTemporalResponse app.
 *
 * \author Katie Twitchell (twitchell@arizona.edu)
 *
 * \ingroup dmTemporalResponse_files
 */

#include "../../../tests/testXWC.hpp"

#include <filesystem>
#include <thread>

#define protected public
#include "../dmTemporalResponse.hpp"
#undef protected

using namespace MagAOX::app;
using namespace MagAOX::app::dmTemporalResponseMath;

namespace libXWCTest
{

/** \defgroup dmTemporalResponse_unit_test dmTemporalResponse Unit Tests
 * \brief Unit tests for the dmTemporalResponse application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `dmTemporalResponse` unit tests.
/** \ingroup dmTemporalResponse_unit_test
 */
namespace dmTemporalResponseTest
{

/// \cond dmTemporalResponse_test_harness

namespace
{

/// The simulated clock, in ns.  Advances a fixed step on every read.
std::atomic<int64_t> s_fakeNs{ 0 };

/// The fake clock step per read, in ns.
std::atomic<int64_t> s_fakeStepNs{ 1000 };

/// The fake clock used for the busy-wait and command timestamps.
timespec fakeClock()
{
    return nsToTs( s_fakeNs.fetch_add( s_fakeStepNs ) + s_fakeStepNs );
}

/// Advance the fake clock to at least \p ns.
void fakeClockAtLeast( int64_t ns /**< [in] the minimum fake time [ns] */ )
{
    int64_t cur = s_fakeNs.load();
    while( cur < ns && !s_fakeNs.compare_exchange_weak( cur, ns ) )
    {
    }
}

/// A unique name for test streams and directories.
std::string uniqueName( const std::string &tag /**< [in] a tag to include in the name */ )
{
    static std::atomic<int> counter{ 0 };
    return "dmresp_test_" + std::to_string( getpid() ) + "_" + tag + "_" + std::to_string( counter++ );
}

/// Trim the trailing blanks FITS adds when padding short string values to 8 characters.
std::string fitsStr( const std::string &v /**< [in] the string read from a FITS header */ )
{
    size_t end = v.find_last_not_of( ' ' );
    return ( end == std::string::npos ) ? std::string() : v.substr( 0, end + 1 );
}

/// Point ImageStreamIO shared-memory files at a writable test sandbox.
/** As in the dm and streamWriter tests.  This also keeps test streams separate from the real system streams.
 *
 * \returns the sandbox directory
 */
std::string ensureMilkShmDir()
{
    static const std::string shmDir = []()
    {
        const std::filesystem::path path = "/tmp/dmTemporalResponse_test/shm";

        std::filesystem::create_directories( path );
        return path.string();
    }();

    setenv( "MILK_SHM_DIR", shmDir.c_str(), 1 );

    return shmDir;
}

/// Make a new-property request with a target element.
template <typename T>
pcf::IndiProperty targetProp( const std::string &device,      /**< [in] the device name */
                              const std::string &name,        /**< [in] the property name */
                              const T           &value,       /**< [in] the target value */
                              bool               text = false /**< [in] true for a text property */
)
{
    pcf::IndiProperty ip( text ? pcf::IndiProperty::Text : pcf::IndiProperty::Number );
    ip.setDevice( device );
    ip.setName( name );
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = value;
    return ip;
}

/// Make a switch request.
pcf::IndiProperty switchProp( const std::string &device, /**< [in] the device name */
                              const std::string &name,   /**< [in] the property name */
                              const std::string &element /**< [in] the element to switch on */
)
{
    pcf::IndiProperty ip( pcf::IndiProperty::Switch );
    ip.setDevice( device );
    ip.setName( name );
    ip.add( pcf::IndiElement( element ) );
    ip[element].setSwitchState( pcf::IndiElement::On );
    return ip;
}

/// Write a 2-D float FITS pattern.
void writePattern( const std::string &path,               /**< [in] the output path */
                   const mx::improc::eigenImage<float> &im /**< [in] the pattern */
)
{
    mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
    ff.write( path, im );
}

} // namespace

/// Test harness for dmTemporalResponse with a fake clock, a test DM stream, and an optional fake camera.
class dmTemporalResponse_test : public dmTemporalResponse
{
  public:
    /// A recorded DM write.
    struct dmWrite
    {
        int64_t                       m_ns; ///< Fake time of the write [ns].
        mx::improc::eigenImage<float> m_cmd; ///< The command written.
    };

    mx::improc::milkImage<float> m_dmTest; ///< The test DM stream.

    std::string m_baseTmp; ///< Temporary output root.

    std::string m_shmDir; ///< The ImageStreamIO sandbox directory.

    std::mutex m_writeMutex; ///< Guards m_writes.

    std::vector<dmWrite> m_writes; ///< All DM writes.

    std::atomic<int> m_failWrites{ 0 }; ///< If > 0, writeDM fails.

    std::mutex m_frameMutex; ///< Serializes fake-camera frames against DM writes from other threads.

    std::thread::id m_camId; ///< The fake camera thread id.

    // Fake camera model
    std::thread       m_camThread;             ///< The fake camera thread.
    std::atomic<bool> m_camStop{ false };      ///< Stop the fake camera.
    std::atomic<bool> m_camPause{ false };     ///< Pause frame production (for timeouts).
    std::atomic<int>  m_gapsToInject{ 0 };     ///< Frame-counter gaps to inject during captures.
    std::atomic<bool> m_alwaysGap{ false };    ///< Inject a gap in every capture.
    std::atomic<int>  m_stopAfterFrames{ -1 }; ///< Request stop after this many frames.
    std::atomic<int>  m_shutdownAfterFrames{ -1 }; ///< Set m_shutdown after this many frames.

    double m_T{ 1000 };      ///< Simulated frame period [us].
    double m_tau{ 3000 };    ///< Simulated DM time constant [us].
    double m_latency{ 50 };  ///< Simulated DM latency [us].
    double m_wake{ 20 };     ///< Simulated wake latency after atime [us].
    float  m_bias{ 100 };    ///< Camera bias.
    float  m_gain{ 10 };     ///< Camera gain per DM unit.

    /// Set up the harness with an nx x ny camera and DM.
    dmTemporalResponse_test( uint32_t nx = 8, /**< [in] camera and DM width */
                             uint32_t ny = 8  /**< [in] camera and DM height */
    )
    {
        // Must precede any ImageStreamIO call
        m_shmDir = ensureMilkShmDir();

        m_configName = uniqueName( "app" );

        m_nx     = nx;
        m_ny     = ny;
        m_pixget = getPixPointer<float>( IMAGESTRUCT_FLOAT );

        m_dmStreamOverride = uniqueName( "dm" );
        m_dmTest.create( m_dmStreamOverride, nx, ny );
        mx::improc::eigenImage<float> z( nx, ny );
        z.setZero();
        m_dmTest = z;

        m_baseTmp = "/tmp/" + uniqueName( "out" );
        m_baseDir = m_baseTmp;

        s_fakeNs     = tsToNs( realtimeNow() );
        s_fakeStepNs = 1000;
        m_clock      = &fakeClock;

        m_wfsFps = 1e6 / m_T;

        // Small, fast defaults
        m_pokeX        = { 3 };
        m_pokeY        = { 4 };
        m_pokeAmp      = 0.5;
        m_nDelays      = 4;
        m_nFrames      = 20;
        m_nTrials      = 4;
        m_nRef         = 2;
        m_nSettle      = 5;
        m_settle       = 0;
        m_trialTimeout = 0.5;
        m_maxRetries   = 3;
    }

    ~dmTemporalResponse_test()
    {
        stopCamera();
        std::error_code ec;
        std::filesystem::remove_all( m_baseTmp, ec );
        for( const std::string &n : { m_dmStreamOverride,
                                      m_configName + "_ref",
                                      m_configName + "_resp",
                                      m_configName + "_respavg" } )
        {
            std::filesystem::remove( m_shmDir + "/" + n + ".im.shm", ec );
        }
    }

    /// Record DM writes, then write the test stream.
    /** Writes from threads other than the fake camera (e.g. zeroing between trials) wait for any frame in progress,
     * so a frame is never generated from a DM state that changes before it is processed.
     */
    int writeDM( const mx::improc::eigenImage<float> &cmd /**< [in] the command */ ) override
    {
        if( std::this_thread::get_id() != m_camId )
        {
            std::lock_guard<std::mutex> flock( m_frameMutex );
            return recordAndWrite( cmd );
        }

        return recordAndWrite( cmd );
    }

    /// Record a DM write and write the test stream.
    int recordAndWrite( const mx::improc::eigenImage<float> &cmd /**< [in] the command */ )
    {
        if( m_failWrites > 0 )
        {
            return -1;
        }

        {
            std::lock_guard<std::mutex> lock( m_writeMutex );
            m_writes.push_back( { s_fakeNs.load(), cmd } );
        }

        return dmTemporalResponse::writeDM( cmd );
    }

    /// The most recent DM write.
    dmWrite lastWrite()
    {
        std::lock_guard<std::mutex> lock( m_writeMutex );
        return m_writes.back();
    }

    /// Simulated DM state at fake time tns for each actuator.
    /** A write of all zeros snaps the model to zero; otherwise a first-order step from zero. */
    mx::improc::eigenImage<float> dmState( int64_t tns /**< [in] fake time [ns] */ )
    {
        std::lock_guard<std::mutex> lock( m_writeMutex );

        mx::improc::eigenImage<float> st( m_nx, m_ny );
        st.setZero();

        if( m_writes.size() == 0 )
        {
            return st;
        }

        const dmWrite &w = m_writes.back();

        if( w.m_cmd.abs().maxCoeff() == 0 )
        {
            return st;
        }

        double dt = ( tns - w.m_ns ) * 1e-3 - m_latency;
        if( dt <= 0 )
        {
            return st;
        }

        return w.m_cmd * static_cast<float>( 1.0 - exp( -dt / m_tau ) );
    }

    /// Run the fake camera: frames every T of fake time, paced ~100 us of real time.
    void startCamera()
    {
        m_camStop = false;
        m_camThread = std::thread(
            [this]()
            {
                // Set before the first frame; startCamera() waits on m_lastATimeNs, which orders this write.
                m_camId = std::this_thread::get_id();

                int64_t  t0    = s_fakeNs.load();
                uint64_t cnt0  = 0;
                int      frame = 0;

                mx::improc::eigenImage<float> im( m_nx, m_ny );

                while( !m_camStop )
                {
                    if( m_camPause )
                    {
                        mx::sys::microSleep( 1000 );
                        continue;
                    }

                    int64_t atimeNs = t0 + static_cast<int64_t>( ( frame + 1 ) * m_T * 1e3 );

                    { // frame scope
                        std::lock_guard<std::mutex> flock( m_frameMutex );

                        fakeClockAtLeast( atimeNs + static_cast<int64_t>( m_wake * 1e3 ) );

                        im = m_bias + m_gain * dmState( atimeNs );

                        ++cnt0;
                        if( m_trialState == trialState::capturing && ( m_alwaysGap || m_gapsToInject > 0 ) )
                        {
                            ++cnt0; // skip a frame counter
                            if( m_gapsToInject > 0 )
                            {
                                --m_gapsToInject;
                            }
                        }

                        processFrame( im.data(), nsToTs( atimeNs ), cnt0 );
                    }

                    ++frame;

                    if( m_stopAfterFrames >= 0 && frame >= m_stopAfterFrames )
                    {
                        requestStop();
                    }

                    if( m_shutdownAfterFrames >= 0 && frame >= m_shutdownAfterFrames )
                    {
                        m_shutdown = 1;
                    }

                    mx::sys::microSleep( 100 );
                }
            } );

        // Let the first frames arrive so the clock-domain check passes
        while( m_lastATimeNs == 0 )
        {
            mx::sys::microSleep( 100 );
        }
    }

    /// Stop the fake camera.
    void stopCamera()
    {
        m_camStop = true;
        if( m_camThread.joinable() )
        {
            m_camThread.join();
        }
    }

    /// Prepare the trial state for direct processFrame tests.
    void setupStateMachine( int    N,            /**< [in] frames per trial */
                            double framePeriodUs /**< [in] frame period [us] */
    )
    {
        m_curNFrames    = N;
        m_framePeriodUs = framePeriodUs;
        m_trialBuf.resize( m_nx, m_ny, N + 1 );
        m_trialTimes.resize( N + 1 );

        m_dmStream.open( m_dmStreamOverride );
        m_cmdPos.resize( m_nx, m_ny );
        m_cmdNeg.resize( m_nx, m_ny );
        m_cmdZero.resize( m_nx, m_ny );
        m_cmdZero.setZero();
        mx::improc::eigenImage<float> empty;
        buildPokeCommand( m_cmdPos, pokeMode::actuator, 3, 4, empty, +1, 0.5 );
        buildPokeCommand( m_cmdNeg, pokeMode::actuator, 3, 4, empty, -1, 0.5 );
    }

    /// Arm a trial directly.
    void arm( double delay, /**< [in] the delay [us] */
              int    sign   /**< [in] +1 or -1 */
    )
    {
        while( sem_trywait( &m_trialSem ) == 0 )
        {
        }
        m_curDelay   = delay;
        m_curSign    = sign;
        m_trialValid = true;
        m_trialLate  = false;
        m_nCaptured  = 0;
        m_trialState = trialState::armed;
    }

    /// The run status, read under the results mutex.
    std::string status()
    {
        std::lock_guard<std::mutex> lock( m_resultsMutex );
        return m_runStatus;
    }

    /// The run phase, read under the results mutex.
    std::string phase()
    {
        std::lock_guard<std::mutex> lock( m_resultsMutex );
        return m_runPhase;
    }

    /// Whether the trial semaphore has been posted.
    bool trialPosted()
    {
        return sem_trywait( &m_trialSem ) == 0;
    }
};

/// \endcond

// ---------------------------------------------------------------------------------------------------------------------
// A. Pure helpers
// ---------------------------------------------------------------------------------------------------------------------

/// Verify timespec arithmetic helpers.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse time helpers", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::tsDiffUs( timespec(), timespec() );
    dmTemporalResponseMath::tsAddUs( timespec(), 0 );
    dmTemporalResponseMath::achievedDelay( timespec(), timespec() );
    #endif
    // clang-format on

    timespec a{ 10, 999999000 };
    timespec b = tsAddUs( a, 2.5 );
    REQUIRE( b.tv_sec == 11 );
    REQUIRE( b.tv_nsec == 1500 );
    REQUIRE( tsDiffUs( b, a ) == Approx( 2.5 ) );

    SECTION( "achieved delay same second and cross second" )
    {
        REQUIRE( achievedDelay( timespec{ 5, 0 }, timespec{ 5, 250000 } ) == Approx( 250 ) );
        REQUIRE( achievedDelay( timespec{ 5, 999900000 }, timespec{ 6, 100000 } ) == Approx( 200 ) );
    }

    SECTION( "poke before reference is negative, not clamped" )
    {
        REQUIRE( achievedDelay( timespec{ 5, 1000 }, timespec{ 5, 0 } ) == Approx( -1 ) );
    }

    SECTION( "ns packing round trips" )
    {
        REQUIRE( tsDiffUs( nsToTs( tsToNs( a ) ), a ) == Approx( 0 ) );
    }
}

/// Verify the bit-level finite checks (robust to -ffast-math) and the FITS header sentinel.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse finite checks and header sentinel", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::isFinite( 0.0 );
    dmTemporalResponseMath::isFinite( 0.0f );
    dmTemporalResponseMath::headerValue( 0.0 );
    #endif
    // clang-format on

    REQUIRE( isFinite( 1.5 ) );
    REQUIRE( isFinite( 0.0 ) );
    REQUIRE( isFinite( -1e300 ) );
    REQUIRE( !isFinite( std::numeric_limits<double>::quiet_NaN() ) );
    REQUIRE( !isFinite( std::numeric_limits<double>::infinity() ) );
    REQUIRE( !isFinite( -std::numeric_limits<double>::infinity() ) );

    REQUIRE( isFinite( 1.5f ) );
    REQUIRE( !isFinite( std::numeric_limits<float>::quiet_NaN() ) );
    REQUIRE( !isFinite( std::numeric_limits<float>::infinity() ) );

    REQUIRE( headerValue( 2.5 ) == 2.5 );
    REQUIRE( headerValue( std::numeric_limits<double>::quiet_NaN() ) == headerSentinel );
    REQUIRE( headerValue( std::numeric_limits<double>::infinity() ) == headerSentinel );
}

/// Verify `dmStreamName()` formatting and range checks.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse dmStreamName", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::dmStreamName( std::string(), 0, 7 );
    #endif
    // clang-format on

    std::string name;
    REQUIRE( dmStreamName( name, 0, 7 ) == 0 );
    REQUIRE( name == "dm00disp07" );
    REQUIRE( dmStreamName( name, 1, 7 ) == 0 );
    REQUIRE( name == "dm01disp07" );
    REQUIRE( dmStreamName( name, 2, 3 ) == 0 );
    REQUIRE( name == "dm02disp03" );
    REQUIRE( dmStreamName( name, -1, 7 ) == -1 );
    REQUIRE( dmStreamName( name, 3, 7 ) == -1 );
    REQUIRE( dmStreamName( name, 0, -1 ) == -1 );
    REQUIRE( dmStreamName( name, 0, 100 ) == -1 );
}

/// Verify poke mode parsing, actuator validation, and command validation.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse poke validation", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::parsePokeMode( pokeMode(), "" );
    dmTemporalResponseMath::validateActuator( {}, {}, 0, 0 );
    dmTemporalResponseMath::validateCommand( 0, 0 );
    #endif
    // clang-format on

    pokeMode mode;
    REQUIRE( parsePokeMode( mode, "actuator" ) == 0 );
    REQUIRE( mode == pokeMode::actuator );
    REQUIRE( parsePokeMode( mode, "pattern" ) == 0 );
    REQUIRE( mode == pokeMode::pattern );
    REQUIRE( parsePokeMode( mode, "both" ) == -1 );

    SECTION( "exactly one in-bounds actuator" )
    {
        REQUIRE( validateActuator( { 0 }, { 0 }, 11, 11 ) == 0 );
        REQUIRE( validateActuator( { 10 }, { 10 }, 11, 11 ) == 0 );
        REQUIRE( validateActuator( {}, {}, 11, 11 ) == -1 );
        REQUIRE( validateActuator( { 1, 2 }, { 1, 2 }, 11, 11 ) == -1 );
        REQUIRE( validateActuator( { 1 }, { 1, 2 }, 11, 11 ) == -1 );
        REQUIRE( validateActuator( { -1 }, { 0 }, 11, 11 ) == -1 );
        REQUIRE( validateActuator( { 11 }, { 0 }, 11, 11 ) == -1 );
        REQUIRE( validateActuator( { 0 }, { 11 }, 11, 11 ) == -1 );
    }

    SECTION( "command limit, default maxCommand = 1" )
    {
        REQUIRE( validateCommand( 1.0, 1.0 ) == 0 );
        REQUIRE( validateCommand( -1.0, 1.0 ) == 0 );
        REQUIRE( validateCommand( 1.01, 1.0 ) == -1 );
        REQUIRE( validateCommand( -1.01, 1.0 ) == -1 );
        REQUIRE( validateCommand( 0, 1.0 ) == -1 );
        REQUIRE( validateCommand( std::numeric_limits<float>::quiet_NaN(), 1.0 ) == -1 );
    }
}

/// Verify `validatePattern()`, `loadPattern()`, and `buildPokeCommand()`.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse pattern loading and command building", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::validatePattern( mx::improc::eigenImage<float>(), 0, 0 );
    dmTemporalResponseMath::loadPattern( mx::improc::eigenImage<float>(), "", 0, 0 );
    dmTemporalResponseMath::buildPokeCommand( mx::improc::eigenImage<float>(), pokeMode(), 0, 0, mx::improc::eigenImage<float>(), 0, 0 );
    #endif
    // clang-format on

    mx::improc::eigenImage<float> pat( 4, 5 );
    pat.setZero();
    pat( 1, 2 ) = 0.5;
    pat( 3, 4 ) = -0.25;

    SECTION( "validatePattern" )
    {
        REQUIRE( validatePattern( pat, 1.0, 1.0 ) == 0 );
        REQUIRE( validatePattern( pat, 2.0, 1.0 ) == 0 );
        REQUIRE( validatePattern( pat, 2.1, 1.0 ) == -1 );

        mx::improc::eigenImage<float> zero( 4, 5 );
        zero.setZero();
        REQUIRE( validatePattern( zero, 1.0, 1.0 ) == -1 );

        mx::improc::eigenImage<float> bad = pat;
        bad( 0, 0 ) = std::numeric_limits<float>::quiet_NaN();
        REQUIRE( validatePattern( bad, 1.0, 1.0 ) == -1 );
        bad( 0, 0 ) = std::numeric_limits<float>::infinity();
        REQUIRE( validatePattern( bad, 1.0, 1.0 ) == -1 );

        mx::improc::eigenImage<float> empty;
        REQUIRE( validatePattern( empty, 1.0, 1.0 ) == -1 );
    }

    SECTION( "loadPattern" )
    {
        std::string dir = "/tmp/" + uniqueName( "pat" );
        std::filesystem::create_directories( dir );

        writePattern( dir + "/good.fits", pat );

        mx::improc::eigenImage<float> loaded;
        REQUIRE( loadPattern( loaded, dir + "/good.fits", 4, 5 ) == 0 );
        REQUIRE( loaded( 1, 2 ) == Approx( 0.5 ) );
        REQUIRE( loaded( 3, 4 ) == Approx( -0.25 ) );

        REQUIRE( loadPattern( loaded, dir + "/good.fits", 5, 4 ) == -1 );
        REQUIRE( loadPattern( loaded, dir + "/missing.fits", 4, 5 ) == -1 );
        REQUIRE( loadPattern( loaded, "", 4, 5 ) == -1 );

        mx::improc::eigenCube<float> cube( 4, 5, 2 );
        cube.image( 0 ) = pat;
        cube.image( 1 ) = pat;
        mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
        ff.write( dir + "/cube.fits", cube );
        REQUIRE( loadPattern( loaded, dir + "/cube.fits", 4, 5 ) == -1 );

        std::ofstream junk( dir + "/junk.fits" );
        junk << "not a fits file";
        junk.close();
        REQUIRE( loadPattern( loaded, dir + "/junk.fits", 4, 5 ) == -1 );

        std::filesystem::remove_all( dir );
    }

    SECTION( "buildPokeCommand" )
    {
        mx::improc::eigenImage<float> cmd( 4, 5 );
        mx::improc::eigenImage<float> empty;

        REQUIRE( buildPokeCommand( cmd, pokeMode::actuator, 2, 3, empty, +1, 0.3 ) == 0 );
        REQUIRE( cmd( 2, 3 ) == Approx( 0.3 ) );
        REQUIRE( cmd.abs().sum() == Approx( 0.3 ) );

        REQUIRE( buildPokeCommand( cmd, pokeMode::actuator, 2, 3, empty, -1, 0.3 ) == 0 );
        REQUIRE( cmd( 2, 3 ) == Approx( -0.3 ) );
        REQUIRE( cmd.abs().sum() == Approx( 0.3 ) );

        REQUIRE( buildPokeCommand( cmd, pokeMode::actuator, 4, 3, empty, +1, 0.3 ) == -1 );
        REQUIRE( buildPokeCommand( cmd, pokeMode::actuator, -1, 3, empty, +1, 0.3 ) == -1 );

        REQUIRE( buildPokeCommand( cmd, pokeMode::pattern, 0, 0, pat, +1, 2.0 ) == 0 );
        REQUIRE( cmd( 1, 2 ) == Approx( 1.0 ) );
        REQUIRE( cmd( 3, 4 ) == Approx( -0.5 ) );

        REQUIRE( buildPokeCommand( cmd, pokeMode::pattern, 0, 0, pat, -1, 2.0 ) == 0 );
        REQUIRE( cmd( 1, 2 ) == Approx( -1.0 ) );
        REQUIRE( cmd( 3, 4 ) == Approx( 0.5 ) );

        mx::improc::eigenImage<float> wrong( 5, 4 );
        wrong.setOnes();
        REQUIRE( buildPokeCommand( cmd, pokeMode::pattern, 0, 0, wrong, +1, 1.0 ) == -1 );
    }
}

/// Verify `resolveSpan()`, `delayGrid()`, and `validateTrials()`.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse delay grid and trial count", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::resolveSpan( double(), 0, 0 );
    dmTemporalResponseMath::delayGrid( std::vector<double>(), 0, 0 );
    dmTemporalResponseMath::validateTrials( 0 );
    #endif
    // clang-format on

    double span = 0;
    REQUIRE( resolveSpan( span, 1500, -1 ) == 0 );
    REQUIRE( span == Approx( 1500 ) );
    REQUIRE( resolveSpan( span, 0, 1000 ) == 0 );
    REQUIRE( span == Approx( 1000 ) );
    REQUIRE( resolveSpan( span, -5, 2000 ) == 0 );
    REQUIRE( span == Approx( 500 ) );
    REQUIRE( resolveSpan( span, 0, 0 ) == -1 );
    REQUIRE( resolveSpan( span, 0, -1 ) == -1 );

    std::vector<double> d;
    REQUIRE( delayGrid( d, 1, 1000 ) == 0 );
    REQUIRE( d.size() == 1 );
    REQUIRE( d[0] == 0 );

    REQUIRE( delayGrid( d, 4, 1000 ) == 0 );
    REQUIRE( d.size() == 4 );
    REQUIRE( d[0] == Approx( 0 ) );
    REQUIRE( d[1] == Approx( 250 ) );
    REQUIRE( d[2] == Approx( 500 ) );
    REQUIRE( d[3] == Approx( 750 ) );

    REQUIRE( delayGrid( d, 3, 1000 ) == 0 );
    REQUIRE( d[1] == Approx( 1000.0 / 3 ) );
    REQUIRE( d[2] == Approx( 2000.0 / 3 ) );

    REQUIRE( delayGrid( d, 0, 1000 ) == -1 );
    REQUIRE( delayGrid( d, 4, 0 ) == -1 );

    REQUIRE( validateTrials( 2 ) == 0 );
    REQUIRE( validateTrials( 20 ) == 0 );
    REQUIRE( validateTrials( 0 ) == -1 );
    REQUIRE( validateTrials( 1 ) == -1 );
    REQUIRE( validateTrials( 21 ) == -1 );
}

/// Verify UTC directory, ISO date, and cube file name formatting.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse output names", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::runDirName( timespec() );
    dmTemporalResponseMath::isoDate( timespec() );
    dmTemporalResponseMath::cubeFileName( 0 );
    #endif
    // clang-format on

    // 2026-09-30T12:34:56 UTC
    timespec ts{ 1790771696, 0 };

    const char *oldTz = getenv( "TZ" );
    std::string saved = oldTz ? oldTz : "";
    setenv( "TZ", "America/Phoenix", 1 );
    tzset();

    REQUIRE( runDirName( ts ) == "2026-09-30T123456" );
    REQUIRE( isoDate( ts ) == "2026-09-30T12:34:56" );

    // New-year rollover
    timespec ny{ 1798761599, 0 }; // 2026-12-31T23:59:59 UTC
    REQUIRE( runDirName( ny ) == "2026-12-31T235959" );
    ny.tv_sec += 1;
    REQUIRE( runDirName( ny ) == "2027-01-01T000000" );

    if( oldTz )
    {
        setenv( "TZ", saved.c_str(), 1 );
    }
    else
    {
        unsetenv( "TZ" );
    }
    tzset();

    REQUIRE( cubeFileName( 0 ) == "dmresp_delay_00000us.fits" );
    REQUIRE( cubeFileName( 250 ) == "dmresp_delay_00250us.fits" );
    REQUIRE( cubeFileName( 333.3 ) == "dmresp_delay_00333us.fits" );
    REQUIRE( cubeFileName( 123456 ) == "dmresp_delay_123456us.fits" );
}

/// Verify `differenceCube()` cancels a constant bias and normalizes by M.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse differenceCube", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::differenceCube( mx::improc::eigenCube<float>(), mx::improc::eigenCube<double>(), mx::improc::eigenCube<double>(), 0 );
    #endif
    // clang-format on

    for( int M : { 2, 20 } )
    {
        mx::improc::eigenCube<double> sp( 3, 2, 2 ), sn( 3, 2, 2 );
        double                        bias = 1e4;

        for( int i = 0; i < 3 * 2 * 2; ++i )
        {
            double R        = 0.1 * i;
            sp.data()[i] = ( M / 2 ) * ( bias + R );
            sn.data()[i] = ( M / 2 ) * ( bias - R );
        }

        mx::improc::eigenCube<float> out;
        REQUIRE( differenceCube( out, sp, sn, M ) == 0 );
        REQUIRE( out.planes() == 2 );

        for( int i = 0; i < 3 * 2 * 2; ++i )
        {
            REQUIRE( out.data()[i] == Approx( 0.1 * i ).margin( 1e-4 ) );
        }
    }

    mx::improc::eigenCube<double> a( 3, 2, 2 ), b( 3, 2, 3 );
    mx::improc::eigenCube<float>  out;
    REQUIRE( differenceCube( out, a, b, 2 ) == -1 );
    REQUIRE( differenceCube( out, a, a, 1 ) == -1 );
}

/// Verify `buildMask()` and `projectResponse()`.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse mask and projection", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::buildMask( mx::improc::eigenImage<float>(), double(), mx::improc::eigenImage<float>(), 0 );
    dmTemporalResponseMath::projectResponse( nullptr, nullptr, nullptr, nullptr, 0, 0 );
    #endif
    // clang-format on

    mx::improc::eigenImage<float> P( 4, 4 );
    P.setConstant( 0.01 ); // noise-level pixels
    P( 1, 1 ) = 2;
    P( 2, 2 ) = -1;

    mx::improc::eigenImage<float> mask;
    double                        norm;
    REQUIRE( buildMask( mask, norm, P, 0.1 ) == 0 );
    REQUIRE( mask.sum() == Approx( 2 ) );
    REQUIRE( norm == Approx( 5 ) );

    mx::improc::eigenImage<float> B( 4, 4 ), I( 4, 4 );
    B.setConstant( 100 );

    for( double a : { 0.0, 0.3, 1.0 } )
    {
        I = B + static_cast<float>( a ) * P;
        I( 0, 0 ) += 50; // noise outside the mask is ignored
        REQUIRE( projectResponse( I.data(), B.data(), P.data(), mask.data(), 16, norm ) == Approx( a ).margin( 1e-6 ) );
    }

    mx::improc::eigenImage<float> zero( 4, 4 ), empty;
    zero.setZero();
    REQUIRE( buildMask( mask, norm, zero, 0.1 ) == -1 );
    REQUIRE( buildMask( mask, norm, empty, 0.1 ) == -1 );
}

/// Verify `crossingTime()` and `computeMetrics()` on analytic step responses.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse response metrics", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::crossingTime( double(), {}, {}, 0 );
    dmTemporalResponseMath::computeMetrics( responseMetrics(), {}, {}, {}, 0, {}, 0 );
    #endif
    // clang-format on

    double              tau = 100, lat = 20, dt = 1;
    std::vector<double> t, r, s;

    for( int i = 0; i < 2000; ++i )
    {
        double ti = i * dt;
        t.push_back( ti );
        r.push_back( ti > lat ? 1 - exp( -( ti - lat ) / tau ) : 0 );
        s.push_back( 0.01 * i );
    }

    SECTION( "first-order step" )
    {
        responseMetrics met;
        REQUIRE( computeMetrics( met, t, r, s, 5, { 1, 3 }, 1 ) == 0 );
        REQUIRE( met.m_t50 == Approx( lat + tau * log( 2 ) ).epsilon( 1e-3 ) );
        REQUIRE( met.m_rise == Approx( tau * log( 9 ) ).epsilon( 1e-3 ) );
        REQUIRE( met.m_overshoot == Approx( 0 ).margin( 1e-6 ) );
        REQUIRE( met.m_settleErr == Approx( 0 ).margin( 1e-6 ) );
        REQUIRE( met.m_delayErrMean == Approx( 2 ) );
        REQUIRE( met.m_delayErrStd == Approx( 1 ) );
        REQUIRE( met.m_lateFrac == Approx( 0.5 ) );

        // jitter is the std at the frame nearest t50
        size_t j = static_cast<size_t>( std::lround( met.m_t50 ) );
        REQUIRE( met.m_jitter == Approx( s[j] ) );
    }

    SECTION( "underdamped overshoot" )
    {
        std::vector<double> ru;
        double              zeta = 0.3, wn = 0.05;
        double              wd   = wn * sqrt( 1 - zeta * zeta );
        for( double ti : t )
        {
            ru.push_back( 1 - exp( -zeta * wn * ti ) * ( cos( wd * ti ) + zeta / sqrt( 1 - zeta * zeta ) * sin( wd * ti ) ) );
        }

        responseMetrics met;
        REQUIRE( computeMetrics( met, t, ru, s, 5, {}, 0 ) == 0 );
        REQUIRE( met.m_overshoot == Approx( exp( -zeta * M_PI / sqrt( 1 - zeta * zeta ) ) ).epsilon( 1e-2 ) );
    }

    SECTION( "no crossing is an error" )
    {
        std::vector<double> flat( t.size(), 0.2 );
        responseMetrics     met;
        REQUIRE( computeMetrics( met, t, flat, s, 5, {}, 0 ) == -1 );

        double tc;
        REQUIRE( crossingTime( tc, t, flat, 0.5 ) == -1 );
        REQUIRE( crossingTime( tc, { 0 }, { 0 }, 0.5 ) == -1 );

        // crosses 0.5 but not 0.9
        std::vector<double> part;
        for( double ti : t )
        {
            part.push_back( 0.6 * ( ti > lat ? 1 - exp( -( ti - lat ) / tau ) : 0 ) );
        }
        REQUIRE( computeMetrics( met, t, part, s, 5, {}, 0 ) == -1 );
        REQUIRE( isFinite( met.m_t50 ) );
        REQUIRE( !isFinite( met.m_rise ) );
    }

    SECTION( "inconsistent inputs" )
    {
        responseMetrics met;
        REQUIRE( computeMetrics( met, t, r, { 0 }, 5, {}, 0 ) == -1 );
    }
}

/// Verify `resampleAverage()` recovers a curve from interleaved phase-shifted samples.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse resampleAverage", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::resampleAverage( std::vector<double>(), std::vector<double>(), {}, {}, 0 );
    #endif
    // clang-format on

    // Four curves sampled every T = 100 with phase offsets 0, 25, 50, 75 of a linear ramp.
    std::vector<std::vector<double>> times, curves;
    for( int p = 0; p < 4; ++p )
    {
        std::vector<double> t, c;
        for( int i = 0; i < 10; ++i )
        {
            double ti = i * 100 + p * 25 + 1; // +1 keeps samples off bin edges
            t.push_back( ti );
            c.push_back( 2 * ti );
        }
        times.push_back( t );
        curves.push_back( c );
    }

    std::vector<double> grid, val;
    REQUIRE( resampleAverage( grid, val, times, curves, 25 ) == 0 );
    REQUIRE( grid.size() == 40 );

    for( size_t b = 0; b < grid.size(); ++b )
    {
        REQUIRE( isFinite( val[b] ) );
        REQUIRE( val[b] == Approx( 2 * ( grid[b] - 12.5 ) ) );
    }

    // Each bin holds exactly one sample, so the value equals that sample.
    REQUIRE( val[1] == Approx( 2 * 26 ) );

    SECTION( "empty bins are NaN" )
    {
        std::vector<std::vector<double>> t1 = { { 0, 100 } }, c1 = { { 1, 2 } };
        REQUIRE( resampleAverage( grid, val, t1, c1, 25 ) == 0 );
        REQUIRE( grid.size() == 5 );
        REQUIRE( val[0] == Approx( 1 ) );
        REQUIRE( !isFinite( val[1] ) );
        REQUIRE( val[4] == Approx( 2 ) );
    }

    SECTION( "non-finite samples are skipped" )
    {
        std::vector<std::vector<double>> t1 = { { 0, std::numeric_limits<double>::quiet_NaN(), 50 } },
                                         c1 = { { 1, 5, 3 } };
        REQUIRE( resampleAverage( grid, val, t1, c1, 25 ) == 0 );
        REQUIRE( val.back() == Approx( 3 ) );
    }

    SECTION( "invalid inputs" )
    {
        REQUIRE( resampleAverage( grid, val, times, curves, 0 ) == -1 );
        REQUIRE( resampleAverage( grid, val, {}, {}, 25 ) == -1 );
        REQUIRE( resampleAverage( grid, val, { { 1, 2 } }, { { 1 } }, 25 ) == -1 );
        std::vector<std::vector<double>> tn = { { std::numeric_limits<double>::quiet_NaN() } }, cn = { { 1 } };
        REQUIRE( resampleAverage( grid, val, tn, cn, 25 ) == -1 );
    }
}

/// Verify `bestDelay()` for each criterion and ties.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse bestDelay", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::bestDelay( size_t(), {}, "" );
    #endif
    // clang-format on

    std::vector<responseMetrics> m( 3 );
    m[0].m_jitter = 0.2;
    m[1].m_jitter = 0.1;
    m[2].m_jitter = 0.1;
    m[0].m_rise   = 5;
    m[1].m_rise   = 7;
    m[2].m_rise   = 3;
    m[0].m_t50    = 10;
    m[1].m_t50    = 9;
    // m[2].m_t50 is NaN

    size_t idx = 99;
    REQUIRE( bestDelay( idx, m, "jitter" ) == 0 );
    REQUIRE( idx == 1 ); // tie goes to the lower index
    REQUIRE( bestDelay( idx, m, "rise" ) == 0 );
    REQUIRE( idx == 2 );
    REQUIRE( bestDelay( idx, m, "t50" ) == 0 );
    REQUIRE( idx == 1 ); // NaN skipped
    REQUIRE( bestDelay( idx, m, "speed" ) == -1 );

    std::vector<responseMetrics> none( 2 );
    REQUIRE( bestDelay( idx, none, "jitter" ) == -1 );
}

/// Verify the SHA-256 helpers against known vectors.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse sha256", "[dmTemporalResponse][helpers]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponseMath::sha256Hex( "" );
    dmTemporalResponseMath::sha256File( std::string(), "" );
    dmTemporalResponseMath::writeOk( 0 );
    #endif
    // clang-format on

    REQUIRE( sha256Hex( "" ) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" );
    REQUIRE( sha256Hex( "abc" ) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" );
    REQUIRE( sha256Hex( "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" ) ==
             "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" );

    std::string path = "/tmp/" + uniqueName( "sha" );
    std::ofstream( path ) << "abc";
    std::string hex;
    REQUIRE( sha256File( hex, path ) == 0 );
    REQUIRE( hex == sha256Hex( "abc" ) );
    std::filesystem::remove( path );
    REQUIRE( sha256File( hex, path ) == -1 );

    REQUIRE( writeOk( 0 ) );
    REQUIRE( !writeOk( -1 ) );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. App-level: configuration and INDI
// ---------------------------------------------------------------------------------------------------------------------

/// Verify configuration defaults.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse configuration defaults", "[dmTemporalResponse][config]" )
{
    dmTemporalResponse app;

    app.setupConfig();

    mx::app::writeConfigFile( "/tmp/dmTemporalResponse_test.conf", { "none" }, { "nada" }, { "0" } );
    app.config.readConfig( "/tmp/dmTemporalResponse_test.conf" );

    app.loadConfig();
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::setupConfig();
    dmTemporalResponse::loadConfig();
    dmTemporalResponse::loadConfigImpl( app.config );
    #endif
    // clang-format on

    REQUIRE( app.m_shutdown == 0 );
    REQUIRE( app.shmimMonitorT::m_shmimName == "camwfs" );
    REQUIRE( app.wfsCamDevName() == "camwfs" );
    REQUIRE( app.dmIndex() == 0 );
    REQUIRE( app.dmChannel() == 7 );
    REQUIRE( app.pokeModeName() == "actuator" );
    REQUIRE( app.pokeX().size() == 0 );
    REQUIRE( app.pokeY().size() == 0 );
    REQUIRE( app.patternFile() == "" );
    REQUIRE( app.pokeAmp() == 0 );
    REQUIRE( app.maxCommand() == Approx( 1 ) );
    REQUIRE( app.nDelays() == 10 );
    REQUIRE( app.delaySpan() == 0 );
    REQUIRE( app.nFrames() == 20 );
    REQUIRE( app.nTrials() == 20 );
    REQUIRE( app.settle() == Approx( 0.05 ) );
    REQUIRE( app.trialTimeout() == Approx( 2 ) );
    REQUIRE( app.maxRetries() == 5 );
    REQUIRE( app.nRef() == 10 );
    REQUIRE( app.nSettle() == 5 );
    REQUIRE( app.maskThresh() == Approx( 0.1 ) );
    REQUIRE( app.resampleFactor() == 10 );
    REQUIRE( app.bestMetric() == "jitter" );
    REQUIRE( app.maxLateFrac() == Approx( 0.1 ) );
    REQUIRE( app.baseDir() == "/home/xsup/dm_response" );

    std::string sname;
    dmStreamName( sname, app.dmIndex(), app.dmChannel() );
    REQUIRE( sname == "dm00disp07" );
}

/// Verify configuration overrides and invalid poke mode rejection.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse configuration overrides", "[dmTemporalResponse][config]" )
{
    {
        dmTemporalResponse app;
        app.setupConfig();

        mx::app::writeConfigFile(
            "/tmp/dmTemporalResponse_test_override.conf",
            { "wfscam", "wfscam", "dm", "dm", "poke", "poke", "poke", "poke", "poke", "poke", "poke", "poke", "poke",
              "poke", "poke", "poke", "poke", "analysis", "analysis", "analysis", "analysis", "analysis", "analysis",
              "output" },
            { "shmimName", "camDevName", "index", "channel", "mode", "x", "y", "patternFile", "amp", "maxCommand",
              "nDelays", "delaySpan", "nFrames", "nTrials", "settle", "trialTimeout", "maxRetries", "nRef", "nSettle",
              "maskThresh", "resampleFactor", "bestMetric", "maxLateFrac", "baseDir" },
            { "camtest", "camdev", "1", "3", "pattern", "5", "6", "/tmp/p.fits", "0.25", "0.5", "7", "1500", "12", "8",
              "0.1", "3", "2", "4", "3", "0.2", "5", "rise", "0.3", "/tmp/out" } );

        app.config.readConfig( "/tmp/dmTemporalResponse_test_override.conf" );
        app.loadConfig();

        REQUIRE( app.m_shutdown == 0 );
        REQUIRE( app.shmimMonitorT::m_shmimName == "camtest" );
        REQUIRE( app.wfsCamDevName() == "camdev" );
        REQUIRE( app.dmIndex() == 1 );
        REQUIRE( app.dmChannel() == 3 );
        REQUIRE( app.pokeModeName() == "pattern" );
        REQUIRE( app.pokeX() == std::vector<int>( { 5 } ) );
        REQUIRE( app.pokeY() == std::vector<int>( { 6 } ) );
        REQUIRE( app.patternFile() == "/tmp/p.fits" );
        REQUIRE( app.pokeAmp() == Approx( 0.25 ) );
        REQUIRE( app.maxCommand() == Approx( 0.5 ) );
        REQUIRE( app.nDelays() == 7 );
        REQUIRE( app.delaySpan() == Approx( 1500 ) );
        REQUIRE( app.nFrames() == 12 );
        REQUIRE( app.nTrials() == 8 );
        REQUIRE( app.settle() == Approx( 0.1 ) );
        REQUIRE( app.trialTimeout() == Approx( 3 ) );
        REQUIRE( app.maxRetries() == 2 );
        REQUIRE( app.nRef() == 4 );
        REQUIRE( app.nSettle() == 3 );
        REQUIRE( app.maskThresh() == Approx( 0.2 ) );
        REQUIRE( app.resampleFactor() == 5 );
        REQUIRE( app.bestMetric() == "rise" );
        REQUIRE( app.maxLateFrac() == Approx( 0.3 ) );
        REQUIRE( app.baseDir() == "/tmp/out" );

        std::string sname;
        dmStreamName( sname, app.dmIndex(), app.dmChannel() );
        REQUIRE( sname == "dm01disp03" );
    }

    {
        dmTemporalResponse app;
        app.setupConfig();
        mx::app::writeConfigFile( "/tmp/dmTemporalResponse_test_badmode.conf", { "poke" }, { "mode" }, { "both" } );
        app.config.readConfig( "/tmp/dmTemporalResponse_test_badmode.conf" );
        app.loadConfig();
        REQUIRE( app.m_shutdown == 1 );
    }
}

/// Verify the INDI tunable callbacks, including rejection while running.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse INDI tunables", "[dmTemporalResponse][indi]" )
{
    dmTemporalResponse_test app;
    REQUIRE( app.createIndiProperties() == 0 );

    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::createIndiProperties();
    dmTemporalResponse::newCallBack_m_indiP_dmIndex( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_dmChannel( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_pokeMode( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_pokeX( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_pokeY( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_patternFile( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_pokeAmp( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_nDelays( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_delaySpan( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_nFrames( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_nTrials( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_settle( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_nSettle( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_maskThresh( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_bestMetric( pcf::IndiProperty() );
    dmTemporalResponse::tunableCallback( pcf::IndiProperty(), int(), pcf::IndiProperty() );
    #endif
    // clang-format on

    const std::string dev = app.m_configName;

    REQUIRE( app.newCallBack_m_indiP_dmIndex( targetProp( dev, "dm_index", 1 ) ) == 0 );
    REQUIRE( app.m_dmIndex == 1 );
    REQUIRE( app.newCallBack_m_indiP_dmChannel( targetProp( dev, "dm_channel", 5 ) ) == 0 );
    REQUIRE( app.m_dmChannel == 5 );
    REQUIRE( app.newCallBack_m_indiP_pokeX( targetProp( dev, "poke_x", 6 ) ) == 0 );
    REQUIRE( app.m_pokeX == std::vector<int>( { 6 } ) );
    REQUIRE( app.newCallBack_m_indiP_pokeY( targetProp( dev, "poke_y", 2 ) ) == 0 );
    REQUIRE( app.m_pokeY == std::vector<int>( { 2 } ) );
    REQUIRE( app.newCallBack_m_indiP_patternFile( targetProp( dev, "pattern_file", std::string( "/tmp/x.fits" ), true ) ) == 0 );
    REQUIRE( app.m_patternFile == "/tmp/x.fits" );
    REQUIRE( app.newCallBack_m_indiP_pokeAmp( targetProp( dev, "poke_amp", 0.125 ) ) == 0 );
    REQUIRE( app.m_pokeAmp == Approx( 0.125 ) );
    REQUIRE( app.newCallBack_m_indiP_nDelays( targetProp( dev, "nDelays", 6 ) ) == 0 );
    REQUIRE( app.m_nDelays == 6 );
    REQUIRE( app.newCallBack_m_indiP_delaySpan( targetProp( dev, "delaySpan", 2000.0 ) ) == 0 );
    REQUIRE( app.m_delaySpan == Approx( 2000 ) );
    REQUIRE( app.newCallBack_m_indiP_nFrames( targetProp( dev, "nFrames", 15 ) ) == 0 );
    REQUIRE( app.m_nFrames == 15 );
    REQUIRE( app.newCallBack_m_indiP_nTrials( targetProp( dev, "nTrials", 10 ) ) == 0 );
    REQUIRE( app.m_nTrials == 10 );
    REQUIRE( app.newCallBack_m_indiP_settle( targetProp( dev, "settle", 0.2 ) ) == 0 );
    REQUIRE( app.m_settle == Approx( 0.2 ) );
    REQUIRE( app.newCallBack_m_indiP_nSettle( targetProp( dev, "nSettle", 4 ) ) == 0 );
    REQUIRE( app.m_nSettle == 4 );
    REQUIRE( app.newCallBack_m_indiP_maskThresh( targetProp( dev, "maskThresh", 0.3 ) ) == 0 );
    REQUIRE( app.m_maskThresh == Approx( 0.3 ) );
    REQUIRE( app.newCallBack_m_indiP_bestMetric( targetProp( dev, "bestMetric", std::string( "t50" ), true ) ) == 0 );
    REQUIRE( app.m_bestMetric == "t50" );

    SECTION( "poke_mode selection" )
    {
        REQUIRE( app.newCallBack_m_indiP_pokeMode( switchProp( dev, "poke_mode", "pattern" ) ) == 0 );
        REQUIRE( app.m_pokeModeName == "pattern" );
        REQUIRE( app.m_indiP_pokeMode["pattern"].getSwitchState() == pcf::IndiElement::On );
        REQUIRE( app.m_indiP_pokeMode["actuator"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( app.newCallBack_m_indiP_pokeMode( switchProp( dev, "poke_mode", "actuator" ) ) == 0 );
        REQUIRE( app.m_pokeModeName == "actuator" );

        pcf::IndiProperty off = switchProp( dev, "poke_mode", "pattern" );
        off["pattern"].setSwitchState( pcf::IndiElement::Off );
        REQUIRE( app.newCallBack_m_indiP_pokeMode( off ) == -1 );
        REQUIRE( app.m_pokeModeName == "actuator" );
    }

    SECTION( "rejected while running" )
    {
        app.m_running = true;

        REQUIRE( app.newCallBack_m_indiP_nDelays( targetProp( dev, "nDelays", 3 ) ) == -1 );
        REQUIRE( app.m_nDelays == 6 );
        REQUIRE( app.newCallBack_m_indiP_pokeX( targetProp( dev, "poke_x", 1 ) ) == -1 );
        REQUIRE( app.m_pokeX == std::vector<int>( { 6 } ) );
        REQUIRE( app.newCallBack_m_indiP_pokeY( targetProp( dev, "poke_y", 1 ) ) == -1 );
        REQUIRE( app.newCallBack_m_indiP_patternFile( targetProp( dev, "pattern_file", std::string( "/y" ), true ) ) == -1 );
        REQUIRE( app.m_patternFile == "/tmp/x.fits" );
        REQUIRE( app.newCallBack_m_indiP_pokeMode( switchProp( dev, "poke_mode", "pattern" ) ) == -1 );
        REQUIRE( app.m_pokeModeName == "actuator" );

        app.m_running = false;
    }

    SECTION( "wrong property is rejected" )
    {
        REQUIRE( app.newCallBack_m_indiP_nDelays( targetProp( dev, "nFrames", 3 ) ) == -1 );
        REQUIRE( app.newCallBack_m_indiP_nDelays( targetProp( "otherdev", "nDelays", 3 ) ) == -1 );
        REQUIRE( app.m_nDelays == 6 );
    }

    SECTION( "missing target and current is rejected" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( dev );
        ip.setName( "nDelays" );
        REQUIRE( app.newCallBack_m_indiP_nDelays( ip ) == -1 );
    }
}

/// Verify the start, stop, and fps callbacks.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse INDI controls and fps", "[dmTemporalResponse][indi]" )
{
    dmTemporalResponse_test app;
    REQUIRE( app.createIndiProperties() == 0 );

    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::newCallBack_m_indiP_start( pcf::IndiProperty() );
    dmTemporalResponse::newCallBack_m_indiP_stop( pcf::IndiProperty() );
    dmTemporalResponse::setCallBack_m_indiP_wfsFps( pcf::IndiProperty() );
    dmTemporalResponse::requestStart();
    dmTemporalResponse::requestStop();
    #endif
    // clang-format on

    const std::string dev = app.m_configName;

    SECTION( "stop while idle is harmless" )
    {
        REQUIRE( app.newCallBack_m_indiP_stop( switchProp( dev, "stop", "request" ) ) == 0 );
        REQUIRE( app.m_stopRequested == true );
        REQUIRE( app.m_running == false );

        pcf::IndiProperty noreq = switchProp( dev, "stop", "other" );
        REQUIRE( app.newCallBack_m_indiP_stop( noreq ) == 0 );
    }

    SECTION( "start posts the start semaphore once" )
    {
        REQUIRE( app.newCallBack_m_indiP_start( switchProp( dev, "start", "request" ) ) == 0 );
        REQUIRE( app.m_running == true );
        REQUIRE( sem_trywait( &app.m_startSem ) == 0 );

        // A second start while running is rejected
        REQUIRE( app.newCallBack_m_indiP_start( switchProp( dev, "start", "request" ) ) == -1 );
        REQUIRE( sem_trywait( &app.m_startSem ) == -1 );

        pcf::IndiProperty off = switchProp( dev, "start", "request" );
        off["request"].setSwitchState( pcf::IndiElement::Off );
        REQUIRE( app.newCallBack_m_indiP_start( off ) == 0 );

        REQUIRE( app.newCallBack_m_indiP_start( switchProp( dev, "start", "other" ) ) == 0 );
        app.m_running = false;
    }

    SECTION( "fps set-callback" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( app.m_wfsCamDevName );
        ip.setName( "fps" );
        ip.add( pcf::IndiElement( "current" ) );
        ip["current"] = 2000.0;

        REQUIRE( app.setCallBack_m_indiP_wfsFps( ip ) == 0 );
        REQUIRE( app.m_wfsFps == Approx( 2000 ) );

        pcf::IndiProperty nocur( pcf::IndiProperty::Number );
        nocur.setDevice( app.m_wfsCamDevName );
        nocur.setName( "fps" );
        REQUIRE( app.setCallBack_m_indiP_wfsFps( nocur ) == 0 );
        REQUIRE( app.m_wfsFps == Approx( 2000 ) );

        // A change during a run flags an abort
        app.m_running    = true;
        app.m_run.m_fps  = 2000;
        ip["current"]    = 1000.0;
        REQUIRE( app.setCallBack_m_indiP_wfsFps( ip ) == 0 );
        REQUIRE( app.m_fpsChanged == true );
        app.m_running = false;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// B. App-level: per-frame state machine
// ---------------------------------------------------------------------------------------------------------------------

/// Verify the same-frame poke path for both signs.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse state machine same-frame poke", "[dmTemporalResponse][statemachine]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::processFrame( nullptr, timespec(), 0 );
    dmTemporalResponse::writeDM( mx::improc::eigenImage<float>() );
    #endif
    // clang-format on

    for( int sign : { +1, -1 } )
    {
        dmTemporalResponse_test app;
        app.setupStateMachine( 3, 1000 );

        mx::improc::eigenImage<float> im( 8, 8 );
        int64_t                       t0 = s_fakeNs.load() + 1000000;

        // Idle frames change nothing
        im.setConstant( 7 );
        REQUIRE( app.processFrame( im.data(), nsToTs( t0 ), 1 ) == 0 );
        REQUIRE( app.m_writes.size() == 0 );
        REQUIRE( app.m_lastATimeNs == t0 );

        app.arm( 250, sign );

        // Trigger frame; wake 20 us after atime
        int64_t tTrig = t0 + 1000000;
        s_fakeNs      = tTrig + 20000;
        im.setConstant( 1 );
        REQUIRE( app.processFrame( im.data(), nsToTs( tTrig ), 2 ) == 0 );

        REQUIRE( app.m_trialState == dmTemporalResponse::trialState::capturing );
        REQUIRE( app.m_writes.size() == 1 );
        REQUIRE( app.lastWrite().m_cmd( 3, 4 ) == Approx( sign * 0.5 ) );
        REQUIRE( app.lastWrite().m_cmd.abs().sum() == Approx( 0.5 ) );
        REQUIRE( app.m_dmTest()( 3, 4 ) == Approx( sign * 0.5 ) );
        REQUIRE( app.m_trialLate == false );

        double achieved = achievedDelay( app.m_trigATime, app.m_tCmd );
        REQUIRE( achieved >= 250 );
        REQUIRE( achieved < 255 );

        // Baseline plane holds the trigger frame
        REQUIRE( app.m_trialBuf.image( 0 )( 0, 0 ) == Approx( 1 ) );

        // N = 3 captured frames
        for( int k = 1; k <= 3; ++k )
        {
            REQUIRE( app.trialPosted() == false );
            im.setConstant( 10 * k );
            REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + k * 1000000 ), 2 + k ) == 0 );
            REQUIRE( app.m_trialBuf.image( k )( 5, 5 ) == Approx( 10 * k ) );
        }

        REQUIRE( app.trialPosted() == true );
        REQUIRE( app.m_trialState == dmTemporalResponse::trialState::idle );
        REQUIRE( app.m_trialValid == true );
        REQUIRE( tsDiffUs( app.m_trialTimes[3], nsToTs( tTrig ) ) == Approx( 3000 ) );
    }
}

/// Verify zero delay pokes immediately and is flagged late.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse state machine late poke", "[dmTemporalResponse][statemachine]" )
{
    dmTemporalResponse_test app;
    app.setupStateMachine( 2, 1000 );

    mx::improc::eigenImage<float> im( 8, 8 );
    im.setZero();

    int64_t tTrig = s_fakeNs.load() + 1000000;
    s_fakeNs      = tTrig + 20000;

    app.arm( 0, +1 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig ), 10 ) == 0 );
    REQUIRE( app.m_writes.size() == 1 );
    REQUIRE( app.m_trialLate == true );
    REQUIRE( achievedDelay( app.m_trigATime, app.m_tCmd ) >= 0 );
}

/// Verify a poke deferred past one frame (delay > T).
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse state machine deferred poke", "[dmTemporalResponse][statemachine]" )
{
    dmTemporalResponse_test app;
    app.setupStateMachine( 2, 1000 );

    mx::improc::eigenImage<float> im( 8, 8 );

    int64_t tTrig = s_fakeNs.load() + 1000000;
    s_fakeNs      = tTrig + 20000;

    app.arm( 1500, -1 );

    im.setConstant( 1 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig ), 20 ) == 0 );
    REQUIRE( app.m_writes.size() == 0 );
    REQUIRE( app.m_trialState == dmTemporalResponse::trialState::waitPoke );

    // Next frame: poke at trigger + 1500 us, baseline updated to this frame
    s_fakeNs = tTrig + 1000000 + 20000;
    im.setConstant( 2 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 1000000 ), 21 ) == 0 );
    REQUIRE( app.m_writes.size() == 1 );
    REQUIRE( app.lastWrite().m_cmd( 3, 4 ) == Approx( -0.5 ) );
    REQUIRE( achievedDelay( app.m_trigATime, app.m_tCmd ) >= 1500 );
    REQUIRE( app.m_trialBuf.image( 0 )( 0, 0 ) == Approx( 2 ) );
    REQUIRE( app.m_trialState == dmTemporalResponse::trialState::capturing );

    // Capture index 1 is the first frame after the poke
    im.setConstant( 3 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 2000000 ), 22 ) == 0 );
    REQUIRE( app.m_trialBuf.image( 1 )( 0, 0 ) == Approx( 3 ) );
    im.setConstant( 4 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 3000000 ), 23 ) == 0 );
    REQUIRE( app.trialPosted() );
    REQUIRE( app.m_trialValid );

    SECTION( "gap while waiting for a deferred poke invalidates the trial" )
    {
        app.arm( 1500, +1 );
        REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 4000000 ), 30 ) == 0 );
        REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 5000000 ), 32 ) == 0 );
        REQUIRE( app.m_trialValid == false );
    }
}

/// Verify a frame-counter gap during capture invalidates the trial.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse state machine frame gap", "[dmTemporalResponse][statemachine]" )
{
    dmTemporalResponse_test app;
    app.setupStateMachine( 3, 1000 );

    mx::improc::eigenImage<float> im( 8, 8 );
    im.setZero();

    int64_t tTrig = s_fakeNs.load() + 1000000;
    s_fakeNs      = tTrig + 20000;

    app.arm( 100, +1 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig ), 100 ) == 0 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 1000000 ), 101 ) == 0 );
    REQUIRE( app.processFrame( im.data(), nsToTs( tTrig + 3000000 ), 103 ) == 0 );

    REQUIRE( app.m_trialValid == false );
    REQUIRE( app.trialPosted() );
    REQUIRE( app.m_trialState == dmTemporalResponse::trialState::idle );
}

/// Verify stop during the busy-wait and a failed DM write end the trial.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse state machine stop and write failure", "[dmTemporalResponse][statemachine]" )
{
    dmTemporalResponse_test app;
    app.setupStateMachine( 3, 1000 );

    mx::improc::eigenImage<float> im( 8, 8 );
    im.setZero();

    SECTION( "stop during busy-wait" )
    {
        int64_t tTrig    = s_fakeNs.load() + 1000000;
        s_fakeNs         = tTrig;
        app.m_stopRequested = true;

        app.arm( 900, +1 );
        REQUIRE( app.processFrame( im.data(), nsToTs( tTrig ), 1 ) == 0 );
        REQUIRE( app.m_writes.size() == 0 );
        REQUIRE( app.m_trialValid == false );
        REQUIRE( app.trialPosted() );
        REQUIRE( app.m_trialState == dmTemporalResponse::trialState::idle );
    }

    SECTION( "DM write failure" )
    {
        int64_t tTrig    = s_fakeNs.load() + 1000000;
        s_fakeNs         = tTrig + 20000;
        app.m_failWrites = 1;

        app.arm( 0, +1 );
        REQUIRE( app.processFrame( im.data(), nsToTs( tTrig ), 1 ) == -1 );
        REQUIRE( app.m_trialValid == false );
        REQUIRE( app.trialPosted() );
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// B. App-level: full synthetic runs
// ---------------------------------------------------------------------------------------------------------------------

/// Verify `prepareRun()` rejects each invalid start condition without writing the DM.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse start validation", "[dmTemporalResponse][run]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::prepareRun( timespec() );
    dmTemporalResponse::snapshotParams( dmTemporalResponse::runParams() );
    #endif
    // clang-format on

    dmTemporalResponse_test app;
    timespec                now = realtimeNow();

    SECTION( "no camera frames" )
    {
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    app.m_lastATimeNs = s_fakeNs.load();

    SECTION( "camera not connected" )
    {
        app.m_nx = 0;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "clock domain mismatch" )
    {
        app.m_lastATimeNs = s_fakeNs.load() - 5000000000LL;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "odd nTrials" )
    {
        app.m_nTrials = 3;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "nSettle greater than nFrames" )
    {
        app.m_nSettle = 30;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "bad bestMetric" )
    {
        app.m_bestMetric = "speed";
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "fps unknown with default span" )
    {
        app.m_wfsFps = -1;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "bad delay grid" )
    {
        app.m_nDelays = 0;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "amp = 0 and amp > maxCommand" )
    {
        app.m_pokeAmp = 0;
        REQUIRE( app.prepareRun( now ) == -1 );
        app.m_pokeAmp = 1.5;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "bad DM index" )
    {
        app.m_dmStreamOverride = "";
        app.m_dmIndex          = 5;
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "DM stream missing" )
    {
        app.m_dmStreamOverride = uniqueName( "nodm" );
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "actuator: two entries, and out of bounds" )
    {
        app.m_pokeX = { 1, 2 };
        app.m_pokeY = { 1, 2 };
        REQUIRE( app.prepareRun( now ) == -1 );
        app.m_pokeX = { 8 };
        app.m_pokeY = { 0 };
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "pattern: missing, invalid, and valid" )
    {
        app.m_pokeModeName = "pattern";
        app.m_patternFile  = "/tmp/" + uniqueName( "missing" ) + ".fits";
        REQUIRE( app.prepareRun( now ) == -1 );
        REQUIRE( app.m_patternInfo.find( "error" ) == 0 );

        std::filesystem::create_directories( app.m_baseTmp );
        mx::improc::eigenImage<float> pat( 8, 8 );
        pat.setConstant( 3 ); // 0.5 * 3 > maxCommand
        app.m_patternFile = app.m_baseTmp + "/pat.fits";
        writePattern( app.m_patternFile, pat );
        REQUIRE( app.prepareRun( now ) == -1 );
        REQUIRE( app.m_patternInfo.find( "error" ) == 0 );

        pat.setConstant( 1 );
        writePattern( app.m_patternFile, pat );
        REQUIRE( app.prepareRun( now ) == 0 );
        REQUIRE( app.m_patternInfo.find( "sha256=" ) != std::string::npos );
        REQUIRE( std::filesystem::exists( app.m_runDir + "/pattern.fits" ) );
        REQUIRE( app.m_cmdPos( 0, 0 ) == Approx( 0.5 ) );
        REQUIRE( app.m_cmdNeg( 7, 7 ) == Approx( -0.5 ) );
    }

    SECTION( "unwritable output directory" )
    {
        app.m_baseDir = "/proc/dmresp_cannot_write_here";
        REQUIRE( app.prepareRun( now ) == -1 );
    }

    SECTION( "valid actuator start" )
    {
        REQUIRE( app.prepareRun( now ) == 0 );
        REQUIRE( app.m_delays.size() == 4 );
        REQUIRE( app.m_span == Approx( 1000 ) );
        REQUIRE( app.m_delaysText != "" );
        REQUIRE( std::filesystem::is_directory( app.m_runDir ) );
        REQUIRE( app.m_runDir == app.m_baseTmp + "/" + runDirName( now ) );
    }

    // Nothing is ever written to the DM by prepareRun
    REQUIRE( app.m_writes.size() == 0 );
}

/// Verify a nominal full run against the fake camera: cubes, bias cancellation, metrics, and summary files.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse full synthetic run", "[dmTemporalResponse][run]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::runMeasurement();
    dmTemporalResponse::runReference();
    dmTemporalResponse::runTrial( 0, 0 );
    dmTemporalResponse::writeCube( 0, mx::improc::eigenCube<float>(), {}, 0 );
    dmTemporalResponse::writeReference();
    dmTemporalResponse::writeSummary( {}, {} );
    dmTemporalResponse::appendRunHeader( mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY>() );
    dmTemporalResponse::zeroDM();
    #endif
    // clang-format on

    dmTemporalResponse_test app;
    app.startCamera();

    REQUIRE( app.runMeasurement() == 0 );

    app.stopCamera();

    REQUIRE( app.m_runStatus == "done" );
    REQUIRE( app.m_delays.size() == 4 );

    // The DM ends at zero
    REQUIRE( app.lastWrite().m_cmd.abs().maxCoeff() == 0 );
    REQUIRE( app.m_dmTest().abs().maxCoeff() == 0 );

    // Files
    for( double d : app.m_delays )
    {
        REQUIRE( std::filesystem::exists( app.m_runDir + "/" + cubeFileName( d ) ) );
    }
    REQUIRE( std::filesystem::exists( app.m_runDir + "/reference.fits" ) );
    REQUIRE( std::filesystem::exists( app.m_runDir + "/summary_curves.fits" ) );
    REQUIRE( std::filesystem::exists( app.m_runDir + "/summary_metrics.fits" ) );
    REQUIRE( std::filesystem::exists( app.m_runDir + "/summary_superres.fits" ) );
    REQUIRE( !std::filesystem::exists( app.m_runDir + "/pattern.fits" ) );

    // Cube contents: bias cancels, settled value is gain*amp at the poked pixel
    mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
    mx::improc::eigenCube<float>                     cube;
    mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY>      fh;
    ff.read( cube, fh, app.m_runDir + "/" + cubeFileName( app.m_delays[1] ) );

    REQUIRE( cube.rows() == 8 );
    REQUIRE( cube.cols() == 8 );
    REQUIRE( cube.planes() == 20 );
    REQUIRE( cube.image( 19 )( 3, 4 ) == Approx( app.m_gain * app.m_pokeAmp ).epsilon( 0.01 ) );
    REQUIRE( cube.image( 19 )( 0, 0 ) == Approx( 0 ).margin( 1e-6 ) );
    REQUIRE( cube.image( 0 )( 5, 5 ) == Approx( 0 ).margin( 1e-6 ) );

    REQUIRE( fh["DELAYUS"].value<double>() == Approx( 250 ) );
    REQUIRE( fh["NTRIALS"].value<int>() == 4 );
    REQUIRE( fh["NINVALID"].value<int>() == 0 );
    REQUIRE( fitsStr( fh["POKEMODE"].value<std::string>() ) == "actuator" );
    REQUIRE( fh["POKEX"].value<int>() == 3 );
    REQUIRE( fh["POKEY"].value<int>() == 4 );
    REQUIRE( fitsStr( fh["DMSTREAM"].value<std::string>() ) == app.m_dmStreamOverride );

    // Metrics match the first-order model: t50 = latency + tau ln2, rise = tau ln9
    for( size_t k = 0; k < app.m_metrics.size(); ++k )
    {
        const responseMetrics &m = app.m_metrics[k];
        REQUIRE( m.m_t50 == Approx( app.m_latency + app.m_tau * log( 2 ) ).epsilon( 0.05 ) );
        REQUIRE( m.m_rise == Approx( app.m_tau * log( 9 ) ).epsilon( 0.05 ) );
        REQUIRE( std::fabs( m.m_delayErrMean ) < 30 );
        REQUIRE( m.m_jitter == Approx( 0 ).margin( 1e-3 ) );
    }

    // Zero delay is always late (20 us wake latency); the others are not
    REQUIRE( app.m_metrics[0].m_lateFrac == Approx( 1 ) );
    REQUIRE( app.m_metrics[1].m_lateFrac == Approx( 0 ) );
    REQUIRE( app.m_metrics[3].m_lateFrac == Approx( 0 ) );

    REQUIRE( app.m_haveBest );

    // Super-sampled response follows the model
    mx::improc::eigenImage<float> sr;
    ff.read( sr, app.m_runDir + "/summary_superres.fits" );
    int nChecked = 0;
    for( int i = 0; i < sr.rows(); ++i )
    {
        double t = sr( i, 0 );
        double v = sr( i, 1 );
        if( isFinite( v ) && t > app.m_latency + 200 )
        {
            REQUIRE( v == Approx( 1 - exp( -( t - app.m_latency ) / app.m_tau ) ).margin( 0.05 ) );
            ++nChecked;
        }
    }
    REQUIRE( nChecked > 20 );
}

/// Verify the M/2 positive then M/2 negative ordering and zeroing between trials.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse trial ordering", "[dmTemporalResponse][run]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::runTrialSet( 0, 0, false, mx::improc::eigenCube<double>(), mx::improc::eigenCube<double>(), std::vector<std::vector<double>>(), std::vector<std::vector<double>>(), std::vector<double>(), int() );
    #endif
    // clang-format on

    dmTemporalResponse_test app;
    app.m_nDelays = 1;
    app.startCamera();
    REQUIRE( app.runMeasurement() == 0 );
    app.stopCamera();

    // Sequence of non-zero pokes: reference (nRef +, nRef -), then delay 0 (M/2 +, M/2 -)
    std::vector<int> signs;
    bool             prevZero = true;
    for( const auto &w : app.m_writes )
    {
        float v = w.m_cmd( 3, 4 );
        if( v != 0 )
        {
            REQUIRE( prevZero ); // the DM is zeroed before every poke
            signs.push_back( v > 0 ? +1 : -1 );
        }
        prevZero = ( v == 0 );
    }

    REQUIRE( signs == std::vector<int>( { +1, +1, -1, -1, +1, +1, -1, -1 } ) );
}

/// Verify an injected frame gap is retried and counted.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse retry on frame gap", "[dmTemporalResponse][run]" )
{
    dmTemporalResponse_test app;
    app.m_nDelays = 1;
    app.startCamera();

    // Let the reference pass finish, then inject one gap
    std::thread injector(
        [&app]()
        {
            while( app.phase() != "measuring" && app.status() == "running" )
            {
                mx::sys::microSleep( 100 );
            }
            app.m_gapsToInject = 1;
        } );

    int rv = app.runMeasurement();
    injector.join();
    app.stopCamera();
    REQUIRE( rv == 0 );

    mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
    mx::improc::eigenCube<float>                     cube;
    mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY>      fh;
    ff.read( cube, fh, app.m_runDir + "/" + cubeFileName( 0 ) );
    REQUIRE( fh["NINVALID"].value<int>() == 1 );
    REQUIRE( cube.image( 19 )( 3, 4 ) == Approx( app.m_gain * app.m_pokeAmp ).epsilon( 0.01 ) );
}

/// Verify too many frame gaps abort the run, keeping only finished cubes.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse abort on retries exceeded", "[dmTemporalResponse][run]" )
{
    dmTemporalResponse_test app;
    app.m_nDelays = 2;
    app.startCamera();

    std::thread injector(
        [&app]()
        {
            while( app.m_progDelayIdx < 1 && app.status() == "running" )
            {
                mx::sys::microSleep( 100 );
            }
            app.m_alwaysGap = true;
        } );

    int rv = app.runMeasurement();
    injector.join();
    app.stopCamera();
    REQUIRE( rv == -1 );

    REQUIRE( app.m_runStatus == "error" );
    REQUIRE( std::filesystem::exists( app.m_runDir + "/" + cubeFileName( app.m_delays[0] ) ) );
    REQUIRE( !std::filesystem::exists( app.m_runDir + "/" + cubeFileName( app.m_delays[1] ) ) );
    REQUIRE( app.m_dmTest().abs().maxCoeff() == 0 );
}

/// Verify a camera that stops producing frames times out and aborts cleanly.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse trial timeout", "[dmTemporalResponse][run]" )
{
    dmTemporalResponse_test app;
    app.m_trialTimeout = 0.1;
    app.startCamera();
    app.m_camPause = true;

    REQUIRE( app.runMeasurement() == -1 );
    app.stopCamera();

    REQUIRE( app.m_runStatus == "error" );
    REQUIRE( app.m_dmTest().abs().maxCoeff() == 0 );
}

/// Verify stop and shutdown mid-run return promptly with the DM zeroed.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse stop and shutdown mid-run", "[dmTemporalResponse][run]" )
{
    SECTION( "stop" )
    {
        dmTemporalResponse_test app;
        app.m_stopAfterFrames = 150;
        app.startCamera();

        REQUIRE( app.runMeasurement() == -1 );
        app.stopCamera();

        REQUIRE( app.m_runStatus == "stopped" );
        REQUIRE( app.m_dmTest().abs().maxCoeff() == 0 );
        REQUIRE( !std::filesystem::exists( app.m_runDir + "/summary_metrics.fits" ) );
    }

    SECTION( "shutdown" )
    {
        dmTemporalResponse_test app;
        app.m_shutdownAfterFrames = 150;
        app.startCamera();

        REQUIRE( app.runMeasurement() == -1 );
        app.stopCamera();

        REQUIRE( app.m_runStatus != "done" );
        REQUIRE( app.m_dmTest().abs().maxCoeff() == 0 );
    }

    SECTION( "fps change" )
    {
        dmTemporalResponse_test app;
        app.startCamera();

        std::thread changer(
            [&app]()
            {
                while( app.phase() != "measuring" && app.status() == "running" )
                {
                    mx::sys::microSleep( 100 );
                }
                app.m_fpsChanged = true;
            } );

        int rv = app.runMeasurement();
        changer.join();
        app.stopCamera();
        REQUIRE( rv == -1 );

        REQUIRE( app.m_runStatus == "error" );
    }
}

/// Verify a full run in pattern mode with a multi-actuator FITS pattern.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse pattern-mode run", "[dmTemporalResponse][run]" )
{
    dmTemporalResponse_test app;
    app.m_pokeModeName = "pattern";
    app.m_nDelays      = 2;

    std::filesystem::create_directories( app.m_baseTmp );
    mx::improc::eigenImage<float> pat( 8, 8 );
    pat.setZero();
    pat( 1, 1 ) = 1.0;
    pat( 6, 2 ) = -0.5;
    app.m_patternFile = app.m_baseTmp + "/pattern_in.fits";
    writePattern( app.m_patternFile, pat );

    std::string sha;
    sha256File( sha, app.m_patternFile );

    app.startCamera();
    REQUIRE( app.runMeasurement() == 0 );
    app.stopCamera();

    REQUIRE( std::filesystem::exists( app.m_runDir + "/pattern.fits" ) );

    mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
    mx::improc::eigenCube<float>                     cube;
    mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY>      fh;
    ff.read( cube, fh, app.m_runDir + "/" + cubeFileName( app.m_delays[0] ) );

    REQUIRE( fitsStr( fh["POKEMODE"].value<std::string>() ) == "pattern" );
    REQUIRE( fitsStr( fh["PATSHA"].value<std::string>() ) == sha );
    REQUIRE( cube.image( 19 )( 1, 1 ) == Approx( app.m_gain * app.m_pokeAmp * 1.0 ).epsilon( 0.01 ) );
    REQUIRE( cube.image( 19 )( 6, 2 ) == Approx( app.m_gain * app.m_pokeAmp * -0.5 ).epsilon( 0.01 ) );
    REQUIRE( cube.image( 19 )( 3, 4 ) == Approx( 0 ).margin( 1e-6 ) );

    for( const responseMetrics &m : app.m_metrics )
    {
        REQUIRE( m.m_t50 == Approx( app.m_latency + app.m_tau * log( 2 ) ).epsilon( 0.05 ) );
    }
}

/// Verify a reference pass with no WFS response aborts the run.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse zero reference aborts", "[dmTemporalResponse][run]" )
{
    dmTemporalResponse_test app;
    app.m_gain = 0;
    app.startCamera();

    REQUIRE( app.runMeasurement() == -1 );
    app.stopCamera();

    REQUIRE( app.m_runStatus == "error" );
    REQUIRE( !std::filesystem::exists( app.m_runDir + "/reference.fits" ) );
}

/// Verify the measurement thread runs a requested start and clears the running flag.
/**
 * \ingroup dmTemporalResponse_unit_test
 */
TEST_CASE( "dmTemporalResponse measurement thread", "[dmTemporalResponse][run]" )
{
    // clang-format off
    #ifdef DMTEMPORALRESPONSE_TEST_DOXYGEN_REF
    dmTemporalResponse::measThreadStart( nullptr );
    dmTemporalResponse::measThreadExec();
    dmTemporalResponse::setRunState( "", "" );
    dmTemporalResponse::setPatternInfo( "" );
    #endif
    // clang-format on

    dmTemporalResponse_test app;
    app.m_nDelays = 1;
    app.startCamera();

    app.m_measThreadInit = false;
    std::thread th( dmTemporalResponse::measThreadStart, &app );

    int startRv = app.requestStart();

    // Wait for the run to finish
    for( int i = 0; i < 2000 && ( app.m_running || app.status() != "done" ); ++i )
    {
        mx::sys::milliSleep( 5 );
    }

    std::string status  = app.status();
    bool        running = app.m_running;

    // Join before any REQUIRE so a failure can not destroy a joinable thread
    app.m_shutdown = 1;
    sem_post( &app.m_startSem );
    th.join();
    app.stopCamera();

    REQUIRE( startRv == 0 );
    REQUIRE( status == "done" );
    REQUIRE( running == false );
}

} // namespace dmTemporalResponseTest

} // namespace libXWCTest
