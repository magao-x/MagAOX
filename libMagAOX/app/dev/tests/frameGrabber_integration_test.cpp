/** \file frameGrabber_integration_test.cpp
 * \brief Offline execution of the real framegrabber loop with deterministic camera and clock endpoints.
 */
#include "../../../../tests/testXWC.hpp"
#include "../../MagAOXApp.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <ctime>
#include <deque>
#include <memory>
#include <type_traits>
#include <vector>

#include <mx/improc/imageUtils.hpp>
#include <ImageStreamIO/ImageStreamIO.h>
#include "../../../logger/types/telem_fgstage.hpp"

/// \cond FRAMEGRABBER_INTEGRATION_HELPERS
namespace frameGrabberIntegration
{
int64_t           monoNs    = 1000000000;            ///< Deterministic local monotonic clock.
size_t            monoReads = 0;                     ///< Number of optional diagnostic monotonic reads.
size_t            realReads = 0;                     ///< Number of existing publication realtime reads.
constexpr int64_t realBase  = 1790000000000000000LL; ///< Independent external-clock epoch.

/// Supply deterministic clocks only to the included inline collector and acquisition loop.
int clockRead( clockid_t clock /**< [in] Requested POSIX clock. */,
               timespec *stamp /**< [out] Deterministic timestamp. */ );

int clockRead( clockid_t clock, timespec *stamp )
{
    int64_t value;

    if( clock == CLOCK_MONOTONIC )
    {
        ++monoReads;
        value = monoNs;
        monoNs += 100;
    }
    else if( clock == CLOCK_REALTIME )
    {
        ++realReads;
        value = realBase + monoNs;
    }
    else
    {
        errno = EINVAL;
        return -1;
    }

    *stamp = { static_cast<time_t>( value / 1000000000 ), static_cast<long>( value % 1000000000 ) };
    return 0;
}
} // namespace frameGrabberIntegration
/// \endcond

#define clock_gettime frameGrabberIntegration::clockRead
#include "../frameGrabber.hpp"
#undef clock_gettime

namespace libXWCTest
{
namespace frameGrabberIntegrationTest
{
using namespace MagAOX::fgTiming;

/// \cond FRAMEGRABBER_INTEGRATION_HELPERS
/// Exercise the actual producer loop without starting an SDK, logger or scheduling helper.
struct Camera : MagAOX::app::MagAOXApp<false>, MagAOX::app::dev::frameGrabber<Camera>
{
    friend class MagAOX::app::dev::frameGrabber<Camera>;
    using Grabber = MagAOX::app::dev::frameGrabber<Camera>;
    using Grabber::m_fgTiming;
    using Grabber::m_traceLogFailed;

    size_t                attempts{ 0 };            ///< Scripted acquisition attempts.
    size_t                loads{ 0 };               ///< Pixel conversion calls.
    size_t                posts{ 0 };               ///< Completed mock sync notifications.
    size_t                frames{ 3 };              ///< Successful publications before ending the loop.
    int                   acquisitionResult{ 0 };   ///< Scripted SDK result; nonzero ends the test after one attempt.
    bool                  failLoad{ false };        ///< Reject conversion before any output publication.
    int64_t               postDelayNs{ 0 };         ///< Additional first-frame sync work, without sleeping.
    int                   reader{ -1 };             ///< Dedicated local primary-reader semaphore.
    bool                  mainBeforeSync{ true };   ///< Every sync callback found the main semaphore already posted.
    bool                  metadataComplete{ true }; ///< Every sync callback saw completed main metadata.
    std::vector<timespec> acquisition;              ///< Original SDK timestamps retained at successful publication.
    std::vector<timespec> publication;              ///< Actual existing realtime writetime samples.
    std::vector<uint64_t> counters;                 ///< Shared-memory publication counters observed by the sync hook.

    int64_t conversionDelayNs{ 0 }; ///< Extra elapsed time during the eighth conversion, after baseline warmup.

    size_t traceCalls{ 0 }; ///< Individual records handed to the ordinary-thread telemetry stub.

    bool failTraceLog{ false }; ///< Inject an ordinary telemetry wrapper failure.

    bool traceVerified{ true }; ///< Production collector records satisfy the typed logger validator.

    std::vector<uint64_t> traceIds; ///< Serialized record IDs used to check one-time emission.

    /// Configure one isolated producer and preallocated timing storage.
    explicit Camera( bool   stages /**< [in] Enable optional local clocks. */,
                     size_t traceMax = 0 /**< [in] Optional bounded individual capture capacity. */ );

    /// Remove a partially configured fixture stream if a test exits before the real loop's cleanup.
    ~Camera() noexcept;

    /// Satisfy the app interface without hardware startup.
    int appStartup() override;

    /// Satisfy the app interface without ordinary application I/O.
    int appLogic() override;

    /// Satisfy the app interface without altering fixture-owned state.
    int appShutdown() override;

    /// Keep the simulated camera in READY for the actual acquisition loop.
    int state() const;

    /// Keep the simulated camera powered throughout the scripted loop.
    int powerState() const;

    /// Configure a two-by-two unsigned-short frame.
    int configureAcquisition();

    /// Report an unchanged diagnostic cadence.
    float fps() const;

    /// Claim one isolated reader so the mock sync hook can check publication order.
    int startAcquisition();

    /// Simulate independent wait and rearm endpoints while preserving the SDK timestamp.
    int acquireAndCheckValid();

    /// Copy deterministic pixels or inject a conversion failure.
    int loadImageIntoStream( void *destination /**< [out] Output pixel slot. */ );

    /// Observe the main notification, then perform scripted synchronization work.
    int frameGrabberPostPublish( IMAGE *image /**< [in] Main stream just published by the real loop. */ );

    /// Satisfy the reconfiguration interface; this fixture never requests a retry.
    int reconfig();

    /// Run the production loop synchronously and drain the resulting timing events afterward.
    void run();

    /// Capture diagnostic serialization without creating a telemetry file or logger thread.
    template <typename telT>
    int telem( const typename telT::messageT &message /**< [in] Consumer-built telemetry message. */ );
};

Camera::Camera( bool stages, size_t traceMax ) : MagAOXApp( "test", false )
{
    static size_t identity = 0;
    m_configName           = "fg-integration-test";
    m_shmimName            = "fg_timing_" + std::to_string( getpid() ) + "_" + std::to_string( ++identity );
    m_fgThreadInit         = false;
    m_fgTiming.configure( 2, 32, stages, 20e-6, 1e-6, traceMax );
}

Camera::~Camera() noexcept
{
    if( m_imageStream )
    {
        ImageStreamIO_destroyIm( m_imageStream );
        free( m_imageStream );
        m_imageStream = nullptr;
    }
}

int Camera::appStartup()
{
    return 0;
}

int Camera::appLogic()
{
    return 0;
}

int Camera::appShutdown()
{
    return 0;
}

int Camera::state() const
{
    return MagAOX::app::stateCodes::READY;
}

int Camera::powerState() const
{
    return 1;
}

int Camera::configureAcquisition()
{
    m_width = m_height = 2;
    m_dataType         = _DATATYPE_UINT16;
    return 0;
}

float Camera::fps() const
{
    return 2000;
}

int Camera::startAcquisition()
{
    reader = ImageStreamIO_getsemwaitindex( m_imageStream, 0 );

    if( reader < 0 )
    {
        m_shutdown = true;
        return -1;
    }

    ImageStreamIO_semflush( m_imageStream, reader );
    return 0;
}

int Camera::acquireAndCheckValid()
{
    ++attempts;
    frameGrabberTimingMark( Marker::waitEnter );
    frameGrabberIntegration::monoNs += 100000;
    frameGrabberTimingMark( Marker::waitReturn );
    frameGrabberIntegration::monoNs += 1000;
    frameGrabberTimingMark( Marker::rearmReturn );
    m_currImageTimestamp = { 1234, static_cast<long>( attempts * 500000 ) };

    if( acquisitionResult )
    {
        m_shutdown = true;
    }

    return acquisitionResult;
}

int Camera::loadImageIntoStream( void *destination )
{
    ++loads;

    if( failLoad )
    {
        m_shutdown = true;
        return -1;
    }

    std::fill_n( static_cast<uint16_t *>( destination ), 4, uint16_t( attempts ) );
    frameGrabberIntegration::monoNs += 10000;

    if( attempts == 8 )
    {
        frameGrabberIntegration::monoNs += conversionDelayNs;
    }

    return 0;
}

int Camera::frameGrabberPostPublish( IMAGE *image )
{
    mainBeforeSync   = mainBeforeSync && sem_trywait( image->semptr[reader] ) == 0;
    metadataComplete = metadataComplete && image->md->write == 0 && image->md->cnt0 == posts + 1;
    acquisition.push_back( image->md->atime );
    publication.push_back( image->md->writetime );
    counters.push_back( image->md->cnt0 );
    ++posts;

    if( posts == 1 )
    {
        frameGrabberIntegration::monoNs += postDelayNs;
    }

    if( posts == frames )
    {
        m_shutdown = true;
    }

    return 0;
}

int Camera::reconfig()
{
    return 0;
}

void Camera::run()
{
    fgThreadExec();
    m_fgTiming.update( frameGrabberIntegration::monoNs,
                       frameGrabberIntegration::realBase + frameGrabberIntegration::monoNs );
}

template <typename telT>
int Camera::telem( const typename telT::messageT &message )
{
    if constexpr( std::is_same_v<telT, MagAOX::logger::telem_fgtrace> )
    {
        ++traceCalls;
        const auto *record = MagAOX::logger::GetTelem_fgtrace_fb( message.builder.GetBufferPointer() );
        traceIds.push_back( record->id() );
        flatlogs::bufferPtrT buffer;
        const int            created =
            flatlogs::logHeader::createLog<telT>( buffer, { 1, 2 }, message, flatlogs::logPrio::LOG_TELEM );
        traceVerified = traceVerified && created == 0 && telT::verify( buffer, telT::length( message ) );
        return failTraceLog ? -1 : 0;
    }

    return 0;
}

/// Reset the deterministic clocks before each isolated execution.
void resetClock();

void resetClock()
{
    frameGrabberIntegration::monoNs    = 1000000000;
    frameGrabberIntegration::monoReads = frameGrabberIntegration::realReads = 0;
}

/// Difference two external realtime samples without epoch-to-double rounding.
int64_t difference( const timespec &a /**< [in] Earlier endpoint. */, const timespec &b /**< [in] Later endpoint. */ );

int64_t difference( const timespec &a, const timespec &b )
{
    return int64_t( b.tv_sec - a.tv_sec ) * 1000000000 + b.tv_nsec - a.tv_nsec;
}
/// \endcond

/** \brief The production loop preserves acquisition stamps and publishes main metadata before the sync hook.
 * \ingroup frameGrabber_unit_test
 */
TEST_CASE( "Framegrabber loop preserves timestamps and main-to-sync order", "[dev::frameGrabber][integration]" )
{
    resetClock();
    Camera camera( true );
    camera.run();
    REQUIRE( camera.posts == 3 );
    CHECK( camera.mainBeforeSync );
    CHECK( camera.metadataComplete );
    CHECK( camera.counters == std::vector<uint64_t>{ 1, 2, 3 } );

    for( size_t i = 0; i < camera.acquisition.size(); ++i )
    {
        CHECK( camera.acquisition[i].tv_sec == 1234 );
        CHECK( camera.acquisition[i].tv_nsec == long( ( i + 1 ) * 500000 ) );
    }

    const auto &summary = camera.m_fgTiming.summary();
    CHECK( summary.total.published == 3 );
    CHECK( summary.total.failed == 0 );
    CHECK( summary.total.dropped == 0 );
    CHECK( summary.metrics[metricIndex( Metric::acquisitionInterval )].mean == Approx( 500e-6 ) );
    CHECK( frameGrabberIntegration::realReads == 3 );

    // clang-format off
#ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber<Camera>::fgThreadExec;
    MagAOX::app::dev::frameGrabber<Camera>::postPublishHook;
#endif
    // clang-format on
}

/** \brief Extra sync work leaves the current write unchanged and enters the next write interval.
 * \ingroup frameGrabber_unit_test
 */
TEST_CASE( "Framegrabber post-publication work belongs to the following frame", "[dev::frameGrabber][integration]" )
{
    resetClock();
    std::vector<timespec> baselinePublication;

    {
        Camera baseline( true );
        baseline.run();
        baselinePublication = baseline.publication;
    }

    resetClock();
    Camera delayed( true );
    delayed.postDelayNs = 2000000;
    delayed.run();
    REQUIRE( baselinePublication.size() == 3 );
    REQUIRE( delayed.publication.size() == 3 );
    CHECK( difference( baselinePublication[0], delayed.publication[0] ) == 0 );
    CHECK( difference( delayed.publication[0], delayed.publication[1] ) ==
           difference( baselinePublication[0], baselinePublication[1] ) + delayed.postDelayNs );
    CHECK( difference( delayed.publication[1], delayed.publication[2] ) ==
           difference( baselinePublication[1], baselinePublication[2] ) );
    const auto &stats = delayed.m_fgTiming.summary().metrics[metricIndex( Metric::syncPost )];
    REQUIRE( stats.count == 3 );
    CHECK( stats.max >= 0.002 );
    CHECK( delayed.mainBeforeSync );

    // clang-format off
#ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber<Camera>::fgThreadExec;
#endif
    // clang-format on
}

/** \brief Disabling optional stages retains paired timing and avoids all diagnostic monotonic clock reads.
 * \ingroup frameGrabber_unit_test
 */
TEST_CASE( "Framegrabber stage disablement leaves only existing write clocks", "[dev::frameGrabber][integration]" )
{
    resetClock();
    Camera camera( false );
    camera.run();
    CHECK( camera.posts == 3 );
    CHECK( frameGrabberIntegration::monoReads == 0 );
    CHECK( frameGrabberIntegration::realReads == 3 );
    const auto &summary = camera.m_fgTiming.summary();
    CHECK_FALSE( summary.stagesEnabled );
    CHECK( summary.total.published == 3 );
    CHECK( summary.metrics[metricIndex( Metric::acquisitionInterval )].count == 2 );
    CHECK( summary.metrics[metricIndex( Metric::conversion )].count == 0 );

    // clang-format off
#ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber<Camera>::frameGrabberTimingMark;
#endif
    // clang-format on
}

/** \brief Failed acquisition and conversion attempts cannot publish shared-memory notifications.
 * \ingroup frameGrabber_unit_test
 */
TEST_CASE( "Framegrabber failed attempts do not publish output", "[dev::frameGrabber][integration]" )
{
    resetClock();
    Camera camera( true );

    SECTION( "negative acquisition result" )
    {
        camera.acquisitionResult = -1;
    }

    SECTION( "rejected acquisition result" )
    {
        camera.acquisitionResult = 1;
    }

    SECTION( "conversion fails before the publication stamp" )
    {
        camera.failLoad = true;
    }

    camera.run();
    CHECK( camera.attempts == 1 );
    CHECK( camera.posts == 0 );
    CHECK( camera.loads == ( camera.failLoad ? 1 : 0 ) );
    CHECK( frameGrabberIntegration::realReads == 0 );
    const auto &summary = camera.m_fgTiming.summary();
    CHECK( summary.total.published == 0 );
    CHECK( summary.total.events == 1 );
    CHECK( summary.total.failed == ( camera.acquisitionResult == 1 ? 0 : 1 ) );
    CHECK( summary.total.rejected == ( camera.acquisitionResult == 1 ? 1 : 0 ) );

    // clang-format off
#ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber<Camera>::fgThreadExec;
#endif
    // clang-format on
}

/** \brief Individual capture reuses producer clocks and emits bounded contexts once on the caller thread.
 * \ingroup frameGrabber_unit_test
 */
TEST_CASE( "Framegrabber excursion telemetry stays outside the acquisition loop",
           "[dev::frameGrabber][integration][trace]" )
{
    resetClock();
    size_t producerReads = 0;

    {
        Camera baseline( true );
        baseline.frames            = 12;
        baseline.conversionDelayNs = 30000;
        baseline.run();
        producerReads = frameGrabberIntegration::monoReads;
    }

    resetClock();
    Camera camera( true, 8 );
    camera.frames            = 12;
    camera.conversionDelayNs = 30000;
    camera.run();
    CHECK( frameGrabberIntegration::monoReads == producerReads );
    CHECK( frameGrabberIntegration::realReads == 12 );
    CHECK( camera.traceCalls == 0 );
    // This fixture advances its next SDK return relative to the preceding loop,
    // so one delayed conversion makes one late interval without a catch-up early interval.
    REQUIRE( camera.m_fgTiming.traceRecords().size() == 1 );

    SECTION( "each context is emitted only once" )
    {
        CHECK( camera.recordFGTraceTimings() == 0 );
    }

    SECTION( "reported telemetry wrapper failures are counted without retry duplication" )
    {
        camera.failTraceLog = true;
        CHECK( camera.recordFGTraceTimings() == -1 );
    }

    CHECK( camera.traceCalls == 1 );
    CHECK( camera.traceVerified );
    CHECK( camera.m_traceLogFailed == ( camera.failTraceLog ? 1 : 0 ) );
    REQUIRE( camera.traceIds.size() == 1 );
    CHECK( camera.traceIds[0] > 0 );
    CHECK( camera.m_fgTiming.traceRecords().empty() );
    CHECK( camera.recordFGTraceTimings() == 0 );
    CHECK( camera.traceCalls == 1 );

    // clang-format off
#ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber<Camera>::recordFGTraceTimings;
    MagAOX::app::dev::frameGrabber<Camera>::fgThreadExec;
#endif
    // clang-format on
}

/** \brief A stopped producer's incomplete neighborhood remains a valid typed record across reset or expiry.
 * \ingroup frameGrabber_unit_test
 */
TEST_CASE( "Framegrabber partial excursion contexts pass the production telemetry validator",
           "[dev::frameGrabber][integration][trace]" )
{
    resetClock();
    Camera camera( true, 8 );
    camera.frames            = 9;
    camera.conversionDelayNs = 30000;
    camera.run();
    REQUIRE( camera.m_fgTiming.traceRecords().empty() );
    REQUIRE( camera.m_fgTiming.traceStatus().pending == 1 );
    auto expected = TraceEnd::boundary;

    SECTION( "configuration boundary" )
    {
        camera.m_fgTiming.newSession();
    }

    SECTION( "stopped acquisition expiry" )
    {
        frameGrabberIntegration::monoNs += 3000000000LL;
        expected = TraceEnd::expired;
    }

    camera.m_fgTiming.update( frameGrabberIntegration::monoNs,
                              frameGrabberIntegration::realBase + frameGrabberIntegration::monoNs );
    REQUIRE( camera.m_fgTiming.traceRecords().size() == 1 );
    CHECK( camera.m_fgTiming.traceRecords()[0].present == 0x0f );
    CHECK( camera.m_fgTiming.traceRecords()[0].end == expected );
    CHECK( camera.recordFGTraceTimings() == 0 );
    CHECK( camera.traceCalls == 1 );
    CHECK( camera.traceVerified );
    CHECK( camera.m_fgTiming.traceRecords().empty() );

    // clang-format off
#ifdef FRAMEGRABBER_TEST_DOXYGEN_REF
    MagAOX::app::dev::frameGrabber<Camera>::recordFGTraceTimings;
#endif
    // clang-format on
}

} // namespace frameGrabberIntegrationTest
} // namespace libXWCTest
