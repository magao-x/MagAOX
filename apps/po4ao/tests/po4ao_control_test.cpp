/** \file po4ao_control_test.cpp
 * \brief Exercise production frame and replay methods with CPU-only SDK boundaries.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../po4ao_utils.hpp"
#include "../po4aoStatistics.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

/** \cond PO4AO_CONTROL_TEST_SUPPORT */
namespace po4aoControlHarness
{
namespace dev
{
/// Monitor tag; no monitor thread or instrument connection exists in these tests.
struct shmimT
{
};
} // namespace dev
/// Shared-memory metadata fields used by the production publication functions.
struct Metadata
{
    uint64_t cnt0{ 0 }, cnt1{ 0 }, cnt2{ 0 };
    int      write{ 0 };
};
/// In-memory image boundary with enough space for the small replay fixture.
struct IMAGE
{
    Metadata              md[1];
    std::array<float, 64> storage{};
    struct
    {
        void *raw{ nullptr };
    } array;
    std::function<void()> posted; ///< Optional boundary fault injected after publication.

    /// Bind the image to its own CPU storage.
    IMAGE();
};
inline IMAGE::IMAGE()
{
    array.raw = storage.data();
}
/// Replace the semaphore post with an optional deterministic boundary callback.
inline void ImageStreamIO_sempost( IMAGE *image /**< [in] Published in-memory image. */,
                                   int    index /**< [in] Ignored semaphore index. */ )
{
    static_cast<void>( index );
    if( image->posted )
        image->posted();
}
/// Model identity needed by the production admission logic; inference itself is an SDK double.
struct EngineBundle
{
    uint64_t           generation{ 1 }, modelId{ 42 };
    int                historyLength{ 2 };
    std::vector<float> linearWeights{ 1 }; ///< Test policy uses the CPU boundary.
};
/// Boundary fixture declaring the state used by the included production method definitions.
/** Model loading and device output are doubles; collection, frame processing, queue
 * ownership, history and shared-memory snapshot publication are the actual methods.
 */
class po4ao
{
  public:
    struct shmimMonitorT
    {
        static inline IMAGE m_imageStream;
    };
    static constexpr size_t c_episodeSlots = 3, c_latencySamples = 16;
    int                     m_nModes{ 3 }, m_nModesControlled{ 2 },
        m_replayBufferSize{ 4 }; ///< Fixture stream width and retained replay capacity.
    uint32_t m_warmupEpisodes{ 2 }, m_iterationsPerEpisode{ 2 },
        m_latencyDeadlineUs{ 1000000 };                     ///< Frame counts and a nonrestrictive CPU test deadline.
    float m_integratorGain{ 0.5F }, m_maxCommandAbs{ 4.F }; ///< Fallback gain and controlled-mode limit.
    float m_explorationSigmaStart{ 1.F }, m_explorationSigmaEnd{ 0.25F }; ///< Configured exploration endpoints.
    std::unique_ptr<float[]> m_inputState{ new float[9] },
        m_output{ new float[3] }; ///< Preallocated CPU tensor/output storage.
    std::unique_ptr<CircularBuffer> m_observationHistory{
        new CircularBuffer( 3, 2 ) }; ///< Retained observation history.
    std::unique_ptr<CircularBuffer> m_commandHistory{
        new CircularBuffer( 3, 1 ) }; ///< Retained emitted-command history.
    std::unique_ptr<CircularBuffer> m_publishObservations{
        new CircularBuffer( 3, 4 ) }; ///< Publisher-owned observation window.
    std::unique_ptr<CircularBuffer> m_publishCommands{
        new CircularBuffer( 3, 4 ) };                   ///< Publisher-owned emitted-command window.
    std::vector<float> m_explorationNoise{ -1.F, 2.F }; ///< Fixed exploration row for exact command assertions.
    size_t             m_explorationNoiseRows{ 1 }, m_explorationNoiseCols{ 2 }; ///< Exploration table dimensions.
    EngineBundle       m_model;                                                  ///< Retained model identity fixture.
    EngineBundle      *m_activeEngine{ &m_model }; ///< Policy selected by the frame callback.
    std::atomic<bool>  m_useNeuralNetwork{ true },
        m_replayShutdown{ false }; ///< Policy eligibility and publisher shutdown flags.
    std::atomic<uint64_t> m_controlRequest{ 3 }, m_appliedControl{ 0 },
        m_modelGeneration{ 1 }; ///< Control request/acknowledgement and model generation.
    std::atomic<uint64_t> m_modelResetRequest{ 0 }, m_appliedReset{ 1 },
        m_effectiveModelId{ 0 };                      ///< Reset request/acknowledgement and emitted model identity.
    std::atomic<int>      m_effectiveController{ 0 }; ///< Last successfully emitted controller.
    std::atomic<uint64_t> m_collectionGeneration{ 1 }, m_warmupRecollectRequest{ 0 },
        m_episodeCounter{ 0 }; ///< Collection request and episode counters.
    uint64_t m_activeCollectionGeneration{ 0 },
        m_publicationGeneration{ 0 }; ///< Producer and publisher collection identities.
    WarmupCollection m_collection;    ///< Actual generation-aware collection lifecycle.
    bool             m_initialWarmupPending{ false },
        m_haveInputCounter{ false }; ///< Initial collection admission and source-counter validity.
    uint64_t                                  m_lastInputCounter{ 0 }; ///< Last processed source publication counter.
    std::array<EpisodeBuffer, c_episodeSlots> m_episodeBuffers;        ///< Preallocated episode handoff storage.
    std::array<std::atomic<int>, c_episodeSlots> m_episodeStates;      ///< Production queue-slot ownership states.
    std::array<int, c_episodeSlots>              m_readyQueue{};       ///< Producer-consumer episode indices.
    std::atomic<uint64_t>                        m_readyRead{ 0 }, m_readyWrite{ 0 }; ///< Atomic queue positions.
    int                     m_activeEpisodeSlot{ 0 };          ///< Slot currently owned by the frame producer.
    std::mutex              m_replayMutex, m_publicationMutex; ///< Publisher wait and stream-publication locks.
    std::condition_variable m_replayCondition;                 ///< Publisher wakeup condition.
    IMAGE                   m_replayStream, m_replayMetadataStream, m_warmupStream,
        m_warmupMetadataStream; ///< In-memory rolling/frozen replay and metadata images.
    bool m_outputOpened{ true }, m_replayOpened{ true },
        m_replayMetadataOpened{ true }; ///< Output and rolling replay availability.
    bool                      m_warmupOpened{ true }, m_warmupMetadataOpened{ true }; ///< Frozen replay availability.
    std::atomic<bool>         m_runAdmission{ true };                                 ///< Whole-app callback admission.
    std::atomic<uint32_t>     m_effectiveBackend{ 0 };                      ///< Backend of the emitted command.
    MagAOX::app::Po4aoSamples m_runtimeSamples;                             ///< Production telemetry sample ring.
    std::atomic<uint64_t>     m_deadlineMisses{ 0 }, m_missedFrames{ 0 };   ///< Latency and source-gap statistics.
    std::atomic<uint64_t>     m_inferenceFailures{ 0 }, m_replayDrops{ 0 }; ///< Inference and queue failure statistics.
    bool                      m_failInference{ false }, m_failOutput{ false }; ///< Injected SDK/output failures.
    std::vector<std::array<float, 3>> m_emitted; ///< Successfully emitted commands captured at the output boundary.
    std::function<void()> m_duringOutput;        ///< Request injection during actual output acknowledgement boundary.

    /// Configure preallocated histories and episode slots.
    po4ao();

    /// Leave model loading outside this fixture's boundary.
    void activatePendingEngine();

    /// Simulate the model-reset boundary, preserving the production generation contract.
    void resetActiveModel( uint64_t generation /**< [in] Reset identity. */ );

    /// Supply a known pre-limit base policy result, or inject an inference failure.
    bool runInference( const float *observation /**< [in] Current observation. */ );

    /// Record output and inject a concurrent request before production acknowledgement.
    int sendOutput();

    /// Emit a new source frame using the actual production callback.
    int frame( uint64_t counter /**< [in] Source publication counter. */ );

    /// Drain the production publication queue synchronously for deterministic tests.
    void drain();

    /// Fill the fallback command before noise and limits.
    void makeIntegratorCommand( const float *observation /**< [in] Observation. */ );

    /// Invalidate the actual replay metadata contract.
    int invalidateReplay( IMAGE &stream /**< [in,out] Replay. */,
                          bool   opened /**< [in] Stream availability. */,
                          IMAGE &metadataStream /**< [in,out] Metadata. */,
                          bool   metadataOpened /**< [in] Availability. */ );

    /// Publish the actual replay layout and paired counters.
    int publishReplay( IMAGE   &stream /**< [in,out] Replay. */,
                       bool     opened /**< [in] Availability. */,
                       IMAGE   &metadataStream /**< [in,out] Metadata. */,
                       bool     metadataOpened /**< [in] Availability. */,
                       uint64_t episodeIdentifier /**< [in] Episode ID. */,
                       uint64_t modelId /**< [in] Effective policy. */ );

    /// Queue a completed production episode.
    void finishEpisode();

    /// Run the production snapshot publisher without sockets or instrument streams.
    void replayPublisherLoop();

    /// Start a new collection using the production method.
    void beginWarmupCollection( uint64_t generation /**< [in] Collection ID. */,
                                uint64_t control /**< [in] Request. */,
                                uint64_t modelId /**< [in] Effective policy. */,
                                uint64_t reset /**< [in] Reset ID. */ );

    /// Process one production frame with fake SDK boundaries.
    int processImage( void *curr_src /**< [in] Observation. */, const dev::shmimT &dummy /**< [in] Monitor tag. */ );
};

#include "../po4ao_control.hpp"

inline po4ao::po4ao()
{
    for( size_t slot = 0; slot < c_episodeSlots; ++slot )
    {
        m_episodeBuffers[slot].configure( m_iterationsPerEpisode, m_nModes );
        m_episodeStates[slot].store( slot == 0 ? 1 : 0 );
    }
    const float previous[]{ 2.F, -2.F, 99.F };
    m_observationHistory->add( previous );
    m_commandHistory->add( previous );
}
inline void po4ao::activatePendingEngine()
{
}
inline void po4ao::resetActiveModel( uint64_t generation )
{
    static_cast<void>( generation );
    m_useNeuralNetwork.store( false );
    m_observationHistory->reset();
    m_commandHistory->reset();
    m_episodeBuffers[static_cast<size_t>( m_activeEpisodeSlot )].reset();
}
inline bool po4ao::runInference( const float *observation )
{
    static_cast<void>( observation );
    m_output[0] = 5.F;
    m_output[1] = -1.F;
    return !m_failInference;
}
inline int po4ao::sendOutput()
{
    if( m_duringOutput )
        m_duringOutput();
    if( m_failOutput )
        return -1;
    m_emitted.push_back( { m_output[0], m_output[1], m_output[2] } );
    return 0;
}
inline int po4ao::frame( uint64_t counter )
{
    shmimMonitorT::m_imageStream.md[0].cnt0 = counter;
    float observation[]{ 2.F, -2.F, 99.F };
    return processImage( observation, dev::shmimT{} );
}
inline void po4ao::drain()
{
    m_replayShutdown.store( true );
    replayPublisherLoop();
    m_replayShutdown.store( false );
}
} // namespace po4aoControlHarness
/** \endcond */

namespace libXWCTest
{
namespace po4aoTest
{
/// Verify the production callback retains history, explores the learned output and publishes complete replay.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production recollection preserves learned control and emitted history", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupRecollectRequest.store( 1 );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage( nullptr, dev::shmimT{} );
    MagAOX::app::po4ao::beginWarmupCollection( 1, 3, 42, 1 );
    MagAOX::app::po4ao::replayPublisherLoop();
#endif
    // clang-format on
    REQUIRE( app.frame( 10 ) == 0 );
    REQUIRE( app.m_effectiveController.load() == 2 );
    REQUIRE( app.m_useNeuralNetwork.load() );
    REQUIRE( app.m_observationHistory->size() == 2 );
    REQUIRE( app.m_emitted.back()[0] == Approx( 4.F ) ); // Noise is added before clipping: 5 - 1, not 4 - 1.
    REQUIRE( app.m_emitted.back()[1] == Approx( 1.F ) );
    REQUIRE( app.m_emitted.back()[2] == 99.F );
    REQUIRE( app.m_commandHistory->getItem( 0 )[1] == Approx( 1.F ) );
    REQUIRE( app.frame( 10 ) == 0 ); // Duplicate notifications do not consume noise or acknowledge anything.
    REQUIRE( app.m_collection.frames() == 1 );
    REQUIRE( app.m_emitted.size() == 1 );
    REQUIRE( app.frame( 11 ) == 0 );
    REQUIRE( app.frame( 12 ) == 0 );
    REQUIRE( app.frame( 13 ) == 0 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::publishing );
    REQUIRE( app.m_warmupMetadataStream.md[0].cnt0 == 0 );
    app.drain();
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
    auto *metadata = static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw );
    REQUIRE( metadata[0] == 4 );
    REQUIRE( metadata[2] == 42 );
    REQUIRE( app.m_warmupStream.md[0].cnt0 == app.m_warmupMetadataStream.md[0].cnt0 );
    const auto *replay = static_cast<float *>( app.m_warmupStream.array.raw );
    for( size_t index = 0; index < 4; ++index )
        for( size_t mode = 0; mode < 3; ++mode )
            REQUIRE( replay[12 + index * 3 + mode] == app.m_emitted[3 - index][mode] );
    REQUIRE( app.frame( 14 ) == 0 );
    REQUIRE( app.m_emitted.back()[1] == Approx( -1.F ) ); // Exploration ended, policy remains active.
}

/// Verify control-Off collection keeps the integrator selected even when a policy is retained.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production warmup works with control disabled", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_controlRequest.store( 4 );
    app.m_warmupRecollectRequest.store( 1 );
    for( uint64_t frame = 1; frame <= 4; ++frame )
    {
        REQUIRE( app.frame( frame ) == 0 );
        REQUIRE( app.m_effectiveController.load() == 1 );
        REQUIRE( app.m_appliedControl.load() == 4 );
    }
    app.drain();
    REQUIRE( app.m_emitted.front()[0] == Approx( 0.F ) );
    REQUIRE( app.m_emitted.front()[1] == Approx( 1.F ) );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
    REQUIRE( static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw )[2] == 0 );
    REQUIRE( app.m_useNeuralNetwork.load() );
}

/// Verify interruption removes noise, rejects incomplete snapshots and preserves ordinary fault handling.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production rejects interrupted warmup", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupRecollectRequest.store( 1 );
    REQUIRE( app.frame( 1 ) == 0 );
    uint64_t next = 2;
    SECTION( "source gap" )
    {
        next = 3;
    }
    SECTION( "policy reload" )
    {
        app.m_model.modelId = 43;
    }
    SECTION( "disable" )
    {
        app.m_controlRequest.store( 4 );
    }
    SECTION( "inference failure" )
    {
        app.m_failInference = true;
    }
    SECTION( "model reset" )
    {
        app.m_modelGeneration.store( 2 );
        app.m_modelResetRequest.store( 2 );
    }
    REQUIRE( app.frame( next ) == 0 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::interrupted );
    REQUIRE( app.m_collection.frames() == 1 );
    app.drain();
    REQUIRE( static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw )[0] == 0 );
    REQUIRE_FALSE( app.m_collection.published( 1 ) );
}

/// Verify acknowledgement describes the request actually used for an emitted output.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production acknowledges only emitted control and reset", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_duringOutput = [&app]() { app.m_controlRequest.store( 4 ); };
    REQUIRE( app.frame( 1 ) == 0 );
    REQUIRE( app.m_appliedControl.load() == 3 );
    REQUIRE( app.m_controlRequest.load() == 4 );
    REQUIRE( app.frame( 1 ) == 0 );
    REQUIRE( app.m_appliedControl.load() == 3 );
    app.m_duringOutput = {};
    REQUIRE( app.frame( 2 ) == 0 );
    REQUIRE( app.m_appliedControl.load() == 4 );
    app.m_modelGeneration.store( 2 );
    app.m_modelResetRequest.store( 2 );
    app.m_failOutput = true;
    REQUIRE( app.frame( 3 ) == -1 );
    REQUIRE( app.m_appliedReset.load() == 1 );
}

/// Verify a saturated handoff queue cannot produce a complete warmup snapshot.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production publication backlog interrupts collection", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupEpisodes = 3;
    app.m_warmupRecollectRequest.store( 1 );
    for( uint64_t frame = 1; frame <= 6; ++frame )
        REQUIRE( app.frame( frame ) == 0 );
    REQUIRE( app.m_replayDrops.load() == 1 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::interrupted );
    app.drain();
    REQUIRE( static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw )[0] == 0 );
}

/// Verify an interruption during the final stream publication invalidates that snapshot.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production invalidates a raced frozen snapshot", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupRecollectRequest.store( 1 );
    app.m_warmupStream.posted = [&app]() { app.m_collection.interrupt( 1 ); };
    for( uint64_t frame = 1; frame <= 4; ++frame )
    {
        REQUIRE( app.frame( frame ) == 0 );
    }
    app.drain();
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::interrupted );
    REQUIRE( static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw )[0] == 0 );
}
/// Verify recollection preserves the active history and restarts exploration only on a new request.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production starts another fixed-policy collection", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupRecollectRequest.store( 1 );
    for( uint64_t frame = 1; frame <= 4; ++frame )
        REQUIRE( app.frame( frame ) == 0 );
    app.drain();
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
    REQUIRE( app.frame( 5 ) == 0 );
    REQUIRE( app.m_emitted.back()[1] == Approx( -1.F ) );
    app.m_collectionGeneration.store( 2 );
    app.m_warmupRecollectRequest.store( 2 );
    REQUIRE( app.frame( 6 ) == 0 );
    REQUIRE( app.m_collection.frames() == 1 );
    REQUIRE( app.m_effectiveModelId.load() == 42 );
    REQUIRE( app.m_observationHistory->size() == 2 );
    REQUIRE( app.m_emitted.back()[1] == Approx( 1.F ) );
    for( uint64_t frame = 7; frame <= 9; ++frame )
        REQUIRE( app.frame( frame ) == 0 );
    app.drain();
    REQUIRE( WarmupCollection::generation( app.m_collection.token() ) == 2 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
}

/// Verify a partial rolling snapshot never makes the frozen warmup valid.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production leaves partial warmup unpublished", "[po4ao][control]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupRecollectRequest.store( 1 );
    REQUIRE( app.frame( 1 ) == 0 );
    REQUIRE( app.frame( 2 ) == 0 );
    app.drain();
    REQUIRE( static_cast<uint32_t *>( app.m_replayMetadataStream.array.raw )[0] == 2 );
    REQUIRE( static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw )[0] == 0 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::collecting );
}
} // namespace po4aoTest
} // namespace libXWCTest
