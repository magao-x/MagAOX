/** \file po4ao_test.cpp
 * \brief Core PO4AO model, callback, routing worker and telemetry sampling regressions.
 *
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../po4ao_utils.hpp"
#include <cmath>
#include <limits>
#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include "../po4aoRouting.hpp"
#include "../../nnReconstructor/tests/nnTestSupport.hpp"
#include <future>
#include <iomanip>
#include <sstream>

namespace libXWCTest
{
/// CPU-only regression tests for the PO4AO application helpers.
/** \ingroup po4ao_unit_test
 */
namespace po4aoTest
{

/** \cond PO4AO_TEST_SUPPORT */
/// CPU-only stand-in for the ownership and generation fields used during model handoff.
struct HandoffModel
{
    uint64_t generation{ 0 }; ///< Reload request generation represented by this candidate.
};
/** \endcond */

/// Verify a candidate published after its reset cannot activate in the new generation.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO rejects a stale model published after reset", "[po4ao]" )
{
    HandoffModel                candidate{ 1 };
    std::atomic<HandoffModel *> pending{ nullptr };
    std::atomic<HandoffModel *> retired{ nullptr };
    std::atomic<int>            status{ 1 };
    std::atomic<uint64_t>       generation{ 1 };
    const uint64_t              checkedGeneration = generation.load();

    // The loader checked generation 1; reset completed before its publication.
    generation.fetch_add( 1 );
    status.store( 2 );
    pending.store( &candidate );
    REQUIRE( candidate.generation == checkedGeneration );

    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    ::takePendingModel( pending, retired, status, generation.load() );
#endif
    // clang-format on

    REQUIRE( takePendingModel( pending, retired, status, generation.load() ) == nullptr );
    REQUIRE( pending.load() == nullptr );
    REQUIRE( retired.load() == &candidate );
    REQUIRE( status.load() == 0 );
    // Background ownership remains explicit; handoff neither deletes nor modifies the model.
    REQUIRE( retired.exchange( nullptr ) == &candidate );
    REQUIRE( candidate.generation == 1 );
}

/// Verify a post-reset candidate can be activated exactly once.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO takes a current generation model once", "[po4ao]" )
{
    HandoffModel                candidate{ 2 };
    std::atomic<HandoffModel *> pending{ &candidate };
    std::atomic<HandoffModel *> retired{ nullptr };
    std::atomic<int>            status{ 2 };
    REQUIRE( takePendingModel( pending, retired, status, 2 ) == &candidate );
    REQUIRE( takePendingModel( pending, retired, status, 2 ) == nullptr );
    REQUIRE( pending.load() == nullptr );
    REQUIRE( retired.load() == nullptr );
}

/// Verify handoff waits for background retirement instead of losing an owned model.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO preserves a busy model retirement slot", "[po4ao]" )
{
    HandoffModel                previous{ 1 };
    HandoffModel                candidate{ 2 };
    std::atomic<HandoffModel *> pending{ &candidate };
    std::atomic<HandoffModel *> retired{ &previous };
    std::atomic<int>            status{ 2 };
    REQUIRE( takePendingModel( pending, retired, status, 2 ) == nullptr );
    REQUIRE( pending.load() == &candidate );
    REQUIRE( retired.load() == &previous );
    REQUIRE( status.load() == 2 );
    REQUIRE( retired.exchange( nullptr ) == &previous );
    REQUIRE( takePendingModel( pending, retired, status, 2 ) == &candidate );
}

/// Verify source counter jumps, repeats, and resets are discontinuities.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO classifies source counter continuity", "[po4ao]" )
{
    REQUIRE( sourceCounterIsConsecutive( 100, 0, false ) );
    REQUIRE( sourceCounterIsConsecutive( 101, 100, true ) );
    REQUIRE_FALSE( sourceCounterIsConsecutive( 104, 100, true ) );
    REQUIRE_FALSE( sourceCounterIsConsecutive( 100, 100, true ) );
    REQUIRE_FALSE( sourceCounterIsConsecutive( 0, 100, true ) );
}

/// Verify only repeat counters with an established previous frame are ignored.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO identifies duplicate source notifications", "[po4ao]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    sourceCounterIsDuplicate( 0, 0, false );
#endif
    // clang-format on

    REQUIRE_FALSE( sourceCounterIsDuplicate( 0, 0, false ) );
    REQUIRE_FALSE( sourceCounterIsDuplicate( 100, 100, false ) );
    REQUIRE( sourceCounterIsDuplicate( 0, 0, true ) );
    REQUIRE( sourceCounterIsDuplicate( 100, 100, true ) );
    REQUIRE_FALSE( sourceCounterIsDuplicate( 101, 100, true ) );
    REQUIRE_FALSE( sourceCounterIsDuplicate( 104, 100, true ) );
    REQUIRE_FALSE( sourceCounterIsDuplicate( 0, 100, true ) );
}

/// Verify CPU linear inference applies independent temporal weights per mode.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO evaluates a linear local policy on the CPU", "[po4ao]" )
{
    const float input[]   = { 1.F, 2.F, 3.F, 4.F, 5.F, 6.F };
    const float weights[] = { 1.F, 0.F, 0.F, 1.F, 2.F, 3.F };
    float       output[]  = { 0.F, 0.F };

    REQUIRE( applyLinearLocalPolicy( output, input, weights, 3, 2, 2 ) );
    REQUIRE( output[0] == Approx( 11.F ) );
    REQUIRE( output[1] == Approx( 22.F ) );
}

/// Verify policy blending has exact fallback and learned-policy endpoints.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO blends policy and fallback commands", "[po4ao]" )
{
    const float observation[] = { 2.F, -4.F };
    float       fallback[]    = { 10.F, 6.F };
    REQUIRE( blendPolicyCommand( fallback, observation, 2, 0.5F, 0.F ) );
    REQUIRE( fallback[0] == Approx( 1.F ) );
    REQUIRE( fallback[1] == Approx( -2.F ) );

    float command[] = { 10.F, 6.F };

    REQUIRE( blendPolicyCommand( command, observation, 2, 0.5F, 0.25F ) );
    REQUIRE( command[0] == Approx( 3.25F ) );
    REQUIRE( command[1] == Approx( 0.F ) );

    float learned[] = { 3.F, -5.F };
    REQUIRE( blendPolicyCommand( learned, observation, 2, 0.5F, 1.F ) );
    REQUIRE( learned[0] == Approx( 3.F ) );
    REQUIRE( learned[1] == Approx( -5.F ) );
}

/// Verify the exported policy tensor matches the Python modal-policy feature order.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO assembles modal policy history", "[po4ao]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    assemblePolicyInput( nullptr, 0, 0, *(CircularBuffer *)nullptr, *(CircularBuffer *)nullptr );
#endif
    // clang-format on

    CircularBuffer observations( 2, 3 );
    CircularBuffer commands( 2, 2 );
    const float    observation0[] = { 10.F, 11.F };
    const float    observation1[] = { 20.F, 21.F };
    const float    observation2[] = { 30.F, 31.F };
    const float    command0[]     = { 100.F, 101.F };
    const float    command1[]     = { 200.F, 201.F };
    observations.add( observation0 );
    observations.add( observation1 );
    observations.add( observation2 );
    commands.add( command0 );
    commands.add( command1 );

    float input[10] = {};
    REQUIRE( assemblePolicyInput( input, 2, 3, observations, commands ) );
    const float expected[] = { 30.F, 31.F, 10.F, 11.F, 20.F, 21.F, 100.F, 101.F, 200.F, 201.F };
    for( size_t index = 0; index < 10; ++index )
    {
        REQUIRE( input[index] == expected[index] );
    }
}

/// Verify a short policy uses the newest samples from a larger retained history.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO assembles a shorter active policy history", "[po4ao]" )
{
    CircularBuffer observations( 1, 4 );
    CircularBuffer commands( 1, 3 );
    for( float value : { 1.F, 2.F, 3.F, 4.F } )
    {
        observations.add( &value );
    }
    for( float value : { 10.F, 20.F, 30.F } )
    {
        commands.add( &value );
    }

    float input[3] = {};
    REQUIRE( assemblePolicyInput( input, 1, 2, observations, commands ) );
    const float expected[] = { 4.F, 3.F, 30.F };
    for( size_t index = 0; index < 3; ++index )
    {
        REQUIRE( input[index] == expected[index] );
    }
}

/// Verify an engine trained on a modal prefix can use full-sized retained histories.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO assembles a controlled modal prefix", "[po4ao]" )
{
    CircularBuffer observations( 4, 2 );
    CircularBuffer commands( 4, 1 );
    const float    observation0[] = { 10.F, 11.F, 12.F, 13.F };
    const float    observation1[] = { 20.F, 21.F, 22.F, 23.F };
    const float    command0[]     = { 30.F, 31.F, 32.F, 33.F };
    observations.add( observation0 );
    observations.add( observation1 );
    commands.add( command0 );

    float input[6] = {};
    REQUIRE( assemblePolicyInput( input, 2, 2, observations, commands ) );
    const float expected[] = { 20.F, 21.F, 10.F, 11.F, 30.F, 31.F };
    for( size_t index = 0; index < 6; ++index )
    {
        REQUIRE( input[index] == expected[index] );
    }
}

/// Verify incomplete history is rejected instead of silently changing the tensor contract.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO rejects incomplete policy history", "[po4ao]" )
{
    CircularBuffer observations( 1, 3 );
    CircularBuffer commands( 1, 2 );
    const float    value = 1.F;
    observations.add( &value );
    commands.add( &value );
    float input[5] = {};
    REQUIRE_FALSE( assemblePolicyInput( input, 1, 3, observations, commands ) );
}

/// Verify command limiting rejects invalid values and enforces the per-mode limit.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO limits differential modal commands", "[po4ao]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    limitCommand( nullptr, 0, 0 );
#endif
    // clang-format on

    float command[] = { 2.F, -0.5F };
    REQUIRE( limitCommand( command, 2, 1.F ) );
    REQUIRE( command[0] == Approx( 1.F ) );
    REQUIRE( command[1] == Approx( -0.5F ) );

    float invalid[] = { std::numeric_limits<float>::quiet_NaN() };
    REQUIRE_FALSE( limitCommand( invalid, 1, 1.F ) );

    const uint32_t infinityBits = 0x7F800000U;
    float          infinite[1];
    memcpy( infinite, &infinityBits, sizeof( infinityBits ) );
    REQUIRE_FALSE( limitCommand( infinite, 1, 1.F ) );
}

/// Verify only the actively controlled prefix is limited while higher modes pass through exactly.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO passes uncontrolled modes through", "[po4ao]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    finalizeControlledCommand( nullptr, nullptr, 0, 0, 0 );
#endif
    // clang-format on

    float       command[]     = { 2.F, -0.5F, 99.F, 99.F };
    const float observation[] = { 0.F, 0.F, 3.25F, -4.5F };
    REQUIRE( finalizeControlledCommand( command, observation, 4, 2, 1.F ) );
    REQUIRE( command[0] == Approx( 1.F ) );
    REQUIRE( command[1] == Approx( -0.5F ) );
    REQUIRE( command[2] == observation[2] );
    REQUIRE( command[3] == observation[3] );

    const float invalidObservation[] = { 0.F, 0.F, std::numeric_limits<float>::quiet_NaN(), 0.F };
    REQUIRE_FALSE( finalizeControlledCommand( command, invalidObservation, 4, 2, 1.F ) );
}

/// Verify resetting an episode removes samples and its collection identity.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO resets replay episode state", "[po4ao]" )
{
    EpisodeBuffer episode;
    episode.configure( 2, 1 );
    const float value = 1.F;
    REQUIRE( episode.add( &value, &value ) );
    episode.episode    = 3;
    episode.generation = 2;

    episode.reset();

    REQUIRE( episode.count == 0 );
    REQUIRE( episode.episode == 0 );
    REQUIRE( episode.generation == 0 );
}

/// Verify noise follows the current learned or fallback command and precedes limiting.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO explores either base controller once before command limits", "[po4ao]" )
{
    const float observation[]{ 4.F, -2.F, 99.F };
    const float noise[]{ -1.F, 2.F };
    float       learned[]{ 3.F, -0.5F, 0.F };
    REQUIRE( blendPolicyCommand( learned, observation, 2, 0.5F, 0.5F ) );
    addExploration( learned, 2, noise, 0, 3, 1.F, 0.25F );
    REQUIRE( finalizeControlledCommand( learned, observation, 3, 2, 2.F ) );
    REQUIRE( learned[0] == Approx( 1.5F ) ); // Blend 2.5, then noise -1, then limit.
    REQUIRE( learned[1] == Approx( 1.25F ) );
    REQUIRE( learned[2] == 99.F );

    float integrator[]{ 2.F, -1.F, 0.F };
    addExploration( integrator, 2, noise, 2, 3, 1.F, 0.25F );
    REQUIRE( finalizeControlledCommand( integrator, observation, 3, 2, 2.F ) );
    REQUIRE( integrator[0] == Approx( 1.75F ) );
    REQUIRE( integrator[1] == Approx( -0.5F ) );
    REQUIRE( integrator[2] == 99.F );
    CircularBuffer commands( 3, 2 );
    commands.add( learned );
    commands.add( integrator );
    EpisodeBuffer episode;
    episode.configure( 2, 3 );
    REQUIRE( episode.add( observation, learned ) );
    REQUIRE( episode.add( observation, integrator ) );
    REQUIRE( commands.getItem( 0 )[0] == episode.commands[3] );
    REQUIRE( commands.getItem( 1 )[1] == episode.commands[1] );
}

/// Verify emission completion and replay publication are distinct, generation-bound events.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO collection completes only after snapshot publication", "[po4ao]" )
{
    WarmupCollection collection;
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::idle );
    collection.begin( 10, 2, 3, 42, 1 );
    collection.validate( 3, 42, 1 );
    REQUIRE( collection.frames() == 0 );
    REQUIRE_FALSE( collection.published( 10 ) );
    collection.emitted();
    REQUIRE( collection.frames() == 1 );
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::collecting );
    collection.emitted();
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::publishing );
    collection.emitted();
    REQUIRE( collection.frames() == 2 );
    REQUIRE_FALSE( collection.published( 9 ) );
    REQUIRE( collection.published( 10 ) );
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::ready );
    REQUIRE_FALSE( collection.published( 10 ) );
    collection.validate( 4, 0, 2 );
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::ready );
}

/// Verify control-Off integrator collection neither activates nor requires a retained policy.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO can collect under the disabled integrator", "[po4ao]" )
{
    WarmupCollection collection;
    collection.begin( 2, 1, 4, 0, 1 );
    collection.validate( 4, 0, 1 );
    collection.emitted();
    REQUIRE( collection.published( 2 ) );
    REQUIRE( collection.frames() == 1 );
}

/// Verify incompatible samples, failed publication and reset cannot complete a fixed-policy dataset.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO interrupts collection on controller changes and publication failures", "[po4ao]" )
{
    WarmupCollection collection;
    collection.begin( 5, 2, 3, 42, 1 );
    collection.emitted();
    SECTION( "control changed" )
    {
        collection.validate( 4, 42, 1 );
    }
    SECTION( "policy changed" )
    {
        collection.validate( 3, 43, 1 );
    }
    SECTION( "inference fell back" )
    {
        collection.validate( 3, 0, 1 );
    }
    SECTION( "reset" )
    {
        collection.validate( 3, 42, 2 );
    }
    SECTION( "publisher backlog or failure" )
    {
        collection.interrupt( 5 );
    }
    SECTION( "failure after final emission" )
    {
        collection.emitted();
        collection.interrupt( 5 );
    }
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::interrupted );
    const auto frames = collection.frames();
    collection.emitted();
    REQUIRE( collection.frames() == frames );
    REQUIRE_FALSE( collection.published( 5 ) );
}

/// Verify recollection starts a fresh schedule while old publisher callbacks cannot change it.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO ignores stale collection publisher results", "[po4ao]" )
{
    WarmupCollection collection;
    collection.begin( 1, 1, 3, 42, 1 );
    collection.emitted();
    collection.begin( 2, 1, 3, 42, 1 );
    REQUIRE( collection.frames() == 0 );
    collection.interrupt( 1 );
    REQUIRE_FALSE( collection.published( 1 ) );
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::collecting );
    collection.emitted();
    REQUIRE_FALSE( collection.published( 1 ) );
    REQUIRE( collection.published( 2 ) );
}

} // namespace po4aoTest
} // namespace libXWCTest

/** \cond PO4AO_CONTROL_TEST_SUPPORT */
namespace po4aoControlHarness
{
using namespace MagAOX::app;
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
    uint64_t cnt0{ 0 }, cnt1{ 0 }, cnt2{ 0 }; ///< Publication, episode and effective-model counters.
    int      write{ 0 };                      ///< In-progress publication marker.
};
/// In-memory image boundary with enough space for the small replay fixture.
struct IMAGE
{
    Metadata              md[1];     ///< Metadata owned by this synthetic image.
    std::array<float, 64> storage{}; ///< Small replay storage used by the production callback.
    struct
    {
        void *raw{ nullptr };     ///< Address of the owning image storage.
    } array;                      ///< ImageStreamIO-compatible data view.
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
    {
        image->posted();
    }
}
/// Model identity needed by the production admission logic; inference itself is an SDK double.
struct EngineBundle
{
    uint64_t           generation{ 1 }, modelId{ 42 }; ///< Accepted reload and effective-policy identities.
    int                historyLength{ 2 };             ///< Temporal input width of the fixture policy.
    std::vector<float> linearWeights{ 1 };             ///< Test policy uses the CPU boundary.
};
/// Boundary fixture declaring the state used by the included production method definitions.
/** Model loading and device output are doubles; collection, frame processing, queue
 * ownership, history and shared-memory snapshot publication are the actual methods.
 */
class po4ao : public MagAOXApp<true>
{
  public:
    /// Monitor storage supplied without a live stream or monitor thread.
    struct shmimMonitorT
    {
        static inline IMAGE m_imageStream; ///< In-memory source image shared by callback tests.
    };
    static constexpr size_t c_episodeSlots = 3, c_latencySamples = 16; ///< Bounded fixture capacities.
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
    std::mutex         m_callbackMutex;            ///< App callback and allocation ownership boundary.
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

    /// Offline model command properties and coherent status snapshots.
    pcf::IndiProperty m_indiP_reloadToggle, m_indiP_resetModel, m_indiP_controlToggle, m_indiP_recollectWarmup,
        m_indiP_modelStatus, m_indiP_collectionStatus;

    /// Routing conflicts and retained per-command rejection state.
    bool m_runBusy{ false }, m_runRunning{ true }, m_reloadRejected{ false }, m_resetRejected{ false },
        m_controlRejected{ false }, m_recollectRejected{ false };

    /// Model-loader boundary and acknowledged model identity.
    std::atomic<int>        m_reloadStatus{ 0 };
    std::atomic<uint64_t>   m_activeModelId{ 42 }, m_reloadGeneration{ 1 };
    std::atomic<bool>       m_activeCpuInference{ true }, m_reloadRequested{ false };
    std::mutex              m_reloadErrorMutex;
    std::condition_variable m_loaderCondition;
    std::string             m_reloadError, m_runPhase{ "running" }, m_runtimeInstance{ "test" };
    uint64_t                m_statusSequence{ 0 }; ///< Monotonic status publication sequence.

    /// Accept source-location logs without starting the production logging service.
    struct Message
    {
        /// Discard the boundary payload.
        template <class... T>
        Message( T &&.../**< [in] Unused logging payload. */ )
        {
        }
    };

    /// Preserve production return values while discarding fixture log messages.
    template <class T, int result = 0>
    int log( Message /**< [in] Unused payload. */, int /**< [in] Unused priority. */ = 0 )
    {
        return result;
    }

    using MagAOXApp<true>::updateIfChanged;
    /// Record the production multi-element snapshot without transport.
    template <class T>
    void updateIfChanged( pcf::IndiProperty              &property /**< [in,out] Snapshot. */,
                          const std::vector<std::string> &names /**< [in] Element names. */,
                          const std::vector<T>           &values /**< [in] Element values. */ );

    /// Execute the production app loop, including stopped-app reset acknowledgement.
    int appLogic();

    /// Routing execution is outside this command/callback fixture.
    void updateRunControl()
    {
    }

    /// Telemetry serialization is covered by the independent telemetry executable.
    void updateRuntimeStatistics()
    {
    }

    /// Publish production status, including retained command rejection acknowledgements.
    void updateControlStatus();

    /// Execute the production rejection acknowledgement.
    int rejectModelCommand( pcf::IndiProperty &property /**< [in,out] Command. */,
                            bool              &rejected /**< [out] Rejection state. */ );

    /// Production reload command boundary.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_reloadToggle );
    /// Production reset command boundary.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_resetModel );
    /// Production control-selection command boundary.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_controlToggle );
    /// Production recollection command boundary.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_recollectWarmup );

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
                       uint64_t modelId /**< [in] Selected collection policy. */ );

    /// Queue a completed production episode.
    void finishEpisode();

    /// Run the production snapshot publisher without sockets or instrument streams.
    void replayPublisherLoop();

    /// Start a new collection using the production method.
    void beginWarmupCollection( uint64_t generation /**< [in] Collection ID. */,
                                uint64_t control /**< [in] Request. */,
                                uint64_t modelId /**< [in] Selected collection policy. */,
                                uint64_t reset /**< [in] Reset ID. */ );

    /// Process one production frame with fake SDK boundaries.
    int processImage( void *curr_src /**< [in] Observation. */, const dev::shmimT &dummy /**< [in] Monitor tag. */ );
};

template <class T>
void po4ao::updateIfChanged( pcf::IndiProperty              &property,
                             const std::vector<std::string> &names,
                             const std::vector<T>           &values )
{
    for( size_t i = 0; i < names.size(); ++i )
    {
        if( !property.find( names[i] ) )
        {
            property.add( pcf::IndiElement( names[i] ) );
        }
        property[names[i]] = values[i];
    }
}

#include "../po4ao_control.hpp"

inline po4ao::po4ao() : MagAOXApp<true>( 0, 0 )
{
    createStandardIndiToggleSw( m_indiP_reloadToggle, "reload_engine" );
    createStandardIndiToggleSw( m_indiP_resetModel, "reset_model" );
    createStandardIndiToggleSw( m_indiP_controlToggle, "control_enable" );
    createStandardIndiToggleSw( m_indiP_recollectWarmup, "recollect_warmup" );
    for( const char *name : { "state", "requested_generation", "generation", "frames", "target_frames" } )
    {
        m_indiP_collectionStatus.add( pcf::IndiElement( name ) );
    }
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
    {
        m_duringOutput();
    }
    if( m_failOutput )
    {
        return -1;
    }
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
    {
        for( size_t mode = 0; mode < 3; ++mode )
        {
            REQUIRE( replay[12 + index * 3 + mode] == app.m_emitted[3 - index][mode] );
        }
    }
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

/// Verify gaps retain complete episodes and reuse the discarded partial episode's exploration indices.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production resumes warmup after input gaps", "[po4ao][control][gap]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::finishEpisode;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    bool                       learned = true;
    SECTION( "retained learned policy" )
    {
    }
    SECTION( "initial integrator warmup" )
    {
        app.m_useNeuralNetwork.store( false );
        learned = false;
    }
    SECTION( "disabled control with retained policy" )
    {
        app.m_controlRequest.store( 4 );
        learned = false;
    }
    app.m_warmupRecollectRequest.store( 1 );
    REQUIRE( app.frame( 1 ) == 0 );
    REQUIRE( app.frame( 2 ) == 0 );
    REQUIRE( app.frame( 3 ) == 0 );
    const auto discarded = app.m_emitted.back();
    REQUIRE( app.m_collection.frames() == 3 );
    REQUIRE( app.frame( 5 ) == 0 );
    REQUIRE( app.m_collection.frames() == 2 );
    REQUIRE( app.m_episodeCounter.load() == 1 );
    REQUIRE( app.m_episodeBuffers[app.m_activeEpisodeSlot].count == 0 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::collecting );
    uint64_t next = 6;
    if( learned )
    {
        REQUIRE( app.frame( 6 ) == 0 );
        REQUIRE( app.m_effectiveModelId.load() == 0 );
        REQUIRE( app.m_emitted.back()[1] == Approx( -1.F ) ); // No exploration during history refill.
        REQUIRE( app.frame( 8 ) == 0 );                       // Another gap before the history has refilled.
        REQUIRE( app.frame( 9 ) == 0 );
        REQUIRE( app.m_collection.frames() == 2 );
        next = 10;
    }
    const auto resumed = app.m_emitted.size();
    REQUIRE( app.frame( next ) == 0 );
    REQUIRE( app.m_emitted.size() == resumed + 1 );
    REQUIRE( app.m_emitted.back() == discarded );
    REQUIRE( app.m_effectiveModelId.load() == ( learned ? 42 : 0 ) );
    REQUIRE( app.m_collection.frames() == 3 );
    REQUIRE( app.frame( next ) == 0 ); // A duplicate never consumes an exploration sample.
    REQUIRE( app.m_collection.frames() == 3 );
    REQUIRE( app.m_emitted.size() == resumed + 1 );
    REQUIRE( app.frame( next + 1 ) == 0 );
    app.drain();
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
    REQUIRE( WarmupCollection::generation( app.m_collection.token() ) == 1 );
    REQUIRE( app.m_collection.frames() == 4 );
    const auto *metadata = static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw );
    REQUIRE( metadata[0] == 4 );
    REQUIRE( metadata[2] == ( learned ? 42 : 0 ) );
    const auto                 *replay = static_cast<float *>( app.m_warmupStream.array.raw );
    const std::array<size_t, 4> accepted{ resumed + 1, resumed, 1, 0 };
    for( size_t sample = 0; sample < accepted.size(); ++sample )
    {
        for( size_t mode = 0; mode < 3; ++mode )
        {
            REQUIRE( replay[12 + sample * 3 + mode] == app.m_emitted[accepted[sample]][mode] );
        }
    }
}

/// Verify a retained policy's initial history refill is excluded from its collected episodes.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production waits for policy history before accepting warmup", "[po4ao][control][gap]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::beginWarmupCollection;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_observationHistory->reset();
    app.m_commandHistory->reset();
    app.m_warmupRecollectRequest.store( 1 );
    REQUIRE( app.frame( 1 ) == 0 );
    REQUIRE( app.m_effectiveModelId.load() == 0 );
    REQUIRE( app.m_collection.frames() == 0 );
    REQUIRE( app.m_episodeBuffers[app.m_activeEpisodeSlot].count == 0 );
    for( uint64_t frame = 2; frame <= 5; ++frame )
    {
        REQUIRE( app.frame( frame ) == 0 );
        REQUIRE( app.m_effectiveModelId.load() == 42 );
    }
    app.drain();
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
    REQUIRE( static_cast<uint32_t *>( app.m_warmupMetadataStream.array.raw )[2] == 42 );
}

/// Verify a later gap cannot invalidate complete episodes queued for the matching frozen publication.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production publishes completed warmup despite a later gap", "[po4ao][control][gap]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
    MagAOX::app::po4ao::replayPublisherLoop;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    app.m_warmupRecollectRequest.store( 1 );
    for( uint64_t frame = 1; frame <= 4; ++frame )
    {
        REQUIRE( app.frame( frame ) == 0 );
    }
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::publishing );
    REQUIRE( app.frame( 6 ) == 0 );
    REQUIRE( app.frame( 7 ) == 0 );
    REQUIRE( app.m_collection.frames() == 4 );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::publishing );
    app.drain();
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::ready );
    REQUIRE( WarmupCollection::generation( app.m_collection.token() ) == 1 );
    const auto *replay = static_cast<float *>( app.m_warmupStream.array.raw );
    for( size_t sample = 0; sample < 4; ++sample )
    {
        REQUIRE( replay[12 + sample * 3 + 1] == app.m_emitted[3 - sample][1] );
    }
    REQUIRE( app.frame( 8 ) == 0 );
    REQUIRE( app.m_emitted.size() == 7 );
    REQUIRE( app.m_effectiveModelId.load() == 42 );
}

/// Verify disabling and re-enabling prediction after a gap needs history, not fresh collection or reload.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO production resumes the same policy without warmup", "[po4ao][control][gap]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    REQUIRE( app.frame( 1 ) == 0 );
    REQUIRE( app.m_effectiveModelId.load() == 42 );
    app.m_controlRequest.store( 4 );
    REQUIRE( app.frame( 3 ) == 0 );
    app.m_controlRequest.store( 5 );
    REQUIRE( app.frame( 4 ) == 0 );
    REQUIRE( app.m_effectiveModelId.load() == 0 );
    REQUIRE( app.frame( 5 ) == 0 );
    REQUIRE( app.m_emitted.size() == 4 );
    REQUIRE( app.m_effectiveModelId.load() == 42 );
    REQUIRE( app.m_effectiveController.load() == 2 );
    REQUIRE( app.m_emitted.back()[1] == Approx( -1.F ) );
    REQUIRE( WarmupCollection::phase( app.m_collection.token() ) == CollectionPhase::idle );
    REQUIRE_FALSE( app.m_reloadRequested.load() );
    REQUIRE( app.m_modelResetRequest.load() == 0 );
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
    REQUIRE( app.frame( 2 ) == 0 );
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
    app.m_duringOutput = [&app]()
    {
        app.m_controlRequest.store( 4 );
    };
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
    {
        REQUIRE( app.frame( frame ) == 0 );
    }
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
    app.m_warmupStream.posted = [&app]()
    {
        app.m_collection.interrupt( 1 );
    };
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
    {
        REQUIRE( app.frame( frame ) == 0 );
    }
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
    {
        REQUIRE( app.frame( frame ) == 0 );
    }
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

namespace libXWCTest::po4aoTest
{
using namespace MagAOX::app;
/** \brief Each command/readback failure retains routing identity and prevents premature detachment. */
TEST_CASE( "PO4AO routing preserves the request identity and never parks on failure", "[po4ao][routing]" )
{
    for( bool start : { false, true } )
    {
        for( int failure = -1; failure < ( start ? 7 : 6 ); ++failure )
        {
            Po4aoRouteBackend        b;
            Po4aoRouteSnapshot       retained;
            const Po4aoRouteSnapshot before{ "po4ao_output", "previous", "nn" };
            int                      step   = 0;
            bool                     parked = false;
            std::vector<std::string> phases;
            auto                     boundary = [&]
            {
                if( step++ == failure )
                {
                    throw std::runtime_error( "injected failure" );
                }
            };
            b.capture = [&]( auto )
            {
                boundary();
                return before;
            };
            b.reserveWriter = [&]( const auto &s, auto )
            {
                REQUIRE( s.writer == "nn" );
                boundary();
            };
            b.freshOutput = [&]( auto )
            {
                boundary();
            };
            b.switchInput = [&]( const auto &target, auto )
            {
                REQUIRE( target == ( start ? "po4ao_output" : "aol1_modevalWFS" ) );
                boundary();
            };
            b.applyDefault = [&]( auto )
            {
                boundary();
            };
            b.restoreWriter = [&]( const auto &s, auto )
            {
                REQUIRE( s.writer == before.writer );
                boundary();
            };
            b.verifyConsumption = [&]( const auto &, auto )
            {
                boundary();
            };
            b.park = [&]( auto )
            {
                parked = true;
            };
            auto run = [&]
            {
                routePo4ao(
                    start,
                    b,
                    retained,
                    [&]( const auto &p )
                    {
                        phases.push_back( p );
                    },
                    std::chrono::steady_clock::now() );
            };
            if( failure < 0 )
            {
                REQUIRE_NOTHROW( run() );
                REQUIRE( parked == !start );
                REQUIRE( phases.back() == ( start ? "running" : "stopped" ) );
            }
            else
            {
                REQUIRE_THROWS( run() );
                REQUIRE_FALSE( parked );
            }
            if( failure != 0 )
            {
                REQUIRE( retained.route == before.route );
            }
        }
    }
}

/** \brief Empty, sub-microsecond, idle and mixed-model windows remain distinguishable. */
TEST_CASE( "PO4AO telemetry describes the actual rolling window", "[po4ao][telemetry]" )
{
    Po4aoSamples samples;
    auto         s = samples.snapshot( 0 );
    REQUIRE( s.count == 0 );
    REQUIRE_FALSE( s.fresh );
    samples.record( 0, 0, 0, 100 );
    s = samples.snapshot( 0 );
    REQUIRE( s.count == 1 );
    REQUIRE( s.p50 == 0 );
    REQUIRE( s.fresh );
    REQUIRE_FALSE( s.mixed );
    REQUIRE_FALSE( samples.snapshot( s.sequence ).fresh );
    constexpr uint64_t model = UINT64_MAX - 10;
    for( unsigned i = 1; i <= 1024; ++i )
    {
        samples.record( i, model, 2, 100 + i );
    }
    s = samples.snapshot( 1 );
    REQUIRE( s.count == 1024 );
    REQUIRE( s.sequence == 1025 );
    REQUIRE( s.model == model );
    REQUIRE( s.backend == 2 );
    REQUIRE( s.p50 == 512 );
    REQUIRE( s.p95 == 973 );
    REQUIRE( s.p99 == 1014 );
    REQUIRE( s.maximum == 1024 );
    REQUIRE_FALSE( s.mixed );
    samples.record( 0, 0, 0, 2000 );
    s = samples.snapshot( 1025 );
    REQUIRE( s.fresh );
    REQUIRE( s.mixed );
    REQUIRE( s.lastTimeNs == 2000 );
    REQUIRE( s.model == 0 );
    REQUIRE( s.backend == 0 );
}
} // namespace libXWCTest::po4aoTest

/// \cond TEST_HARNESS
namespace po4aoRunHarness
{
using namespace MagAOX::app;
/// Record routing boundary calls without starting a child or contacting a device.
struct RouteIO
{
    std::vector<std::string> events; ///< Ordered boundary calls.

    std::string current{ "aol1_modevalWFS" }; ///< Simulated downstream route.
    bool        failCalibration{ false };     ///< Withhold acknowledgement of queued writer restart.
    bool        failVerify{ false };          ///< Inject a consumption failure.
    /// Capture the current route and a test calibration identity.
    Po4aoRouteSnapshot capture();
    /// Return the observed route.
    std::string route() const
    {
        return current;
    }
    /// Change the test route.
    void switchInput( const std::string &target /**< [in] Requested input. */,
                      Po4aoRouteBackend::Deadline /**< [in] Deadline. */ );
    /// Record applying the default calibration.
    void applyDefault( Po4aoRouteBackend::Deadline /**< [in] Deadline. */ );
    /// Provide or withhold actual-consumption acknowledgement.
    void verifyConsumption( const std::string & /**< [in] Input. */,
                            Po4aoRouteBackend::Deadline /**< [in] Deadline. */ );
};
/// State used by the included production worker; transport and monitor scheduling are doubles.
struct po4ao : MagAOXApp<true>, dev::shmimMonitor<po4ao>
{
    using shmimMonitorT = dev::shmimMonitor<po4ao>;
    RouteIO               m_routeIO; ///< Fake utility boundary observed by the production worker.
    std::string           m_outputChannel{ "po4ao_output" }; ///< Fixed instrument route under test.
    Po4aoRouteSnapshot    m_runSnapshot;                     ///< Identity retained across routing failure and recovery.
    std::thread           m_runThread;     ///< Owned production worker, joined before fixture destruction.
    std::mutex            m_callbackMutex; ///< App callback and allocation ownership boundary.
    std::atomic<bool>     m_runCancel{ false }, m_runAdmission{ true }; ///< Shutdown and callback admission.
    std::atomic<uint64_t> m_outputSequence{ 0 };                        ///< Output progress observed while routing.
    bool m_runPending{ false }, m_runBusy{ false }, m_runTarget{ true }, m_runRunning{ true }, m_runVerified{ false },
        m_runCheckOnly{ false }; ///< Requested, executing and verified routing state.
    uint64_t    m_runSequence{ 0 }, m_activeCollectionGeneration{ 1 }; ///< Routing and replay generations.
    std::string m_runPhase{ "unverified" }, m_runRoute{ "unknown" }, m_runError, m_runRequestError,
        m_runOwner{ "test-1" }; ///< Published route status and stable reservation token.
    std::string m_nnDevice{ "nn" }, m_nnRoutingOwner, m_nnReleasedOwner, m_nnRoutingSource, m_nnRoutingPhase,
        m_nnInstance{ "nn-1" }, m_nnSequence{ "1" }, m_reservedInstance; ///< Observed NN reservation identity.
    std::string                           m_runtimeInstance{ "test" };   ///< Synthetic runtime process identity.
    std::chrono::steady_clock::time_point m_nnObserved{ std::chrono::steady_clock::now() };  ///< Readback freshness.
    pcf::IndiProperty                     m_indiP_run, m_indiP_runStatus, m_indiP_nnRouting; ///< Offline properties.
    std::atomic<int>                      m_reloadStatus{ 0 }; ///< Model-loading admission boundary.
    WarmupCollection                      m_collection;        ///< Production warmup invalidated on detachment.
    std::condition_variable               m_replayCondition;   ///< Wakeup used when routing interrupts collection.
    /// Initialize a running fixture with no instrument connection.
    po4ao();
    /// Join any production worker launched by a request.
    ~po4ao();
    /// Acknowledge the reservation through its transport boundary.
    int                   sendNewProperty( const pcf::IndiProperty &p /**< [in] Native reservation request. */ );
    std::atomic<uint64_t> m_modelResetRequest{ 0 }; ///< Pending reset admission boundary.
    /// Production method declarations.
    void runPhase( const std::string &phase /**< [in] Phase. */ );
    /// Await a boundary condition.
    void waitRun( const std::function<bool()> &ready /**< [in] Predicate. */,
                  Po4aoRouteBackend::Deadline  deadline /**< [in] Deadline. */ );
    /// Send reservation intent.
    void requestWriterHold( const std::string &action /**< [in] Action. */ );
    /// Read reservation completion.
    bool writerHeld( bool released /**< [in] Restore acknowledgement. */ );
    /// Check completed restoration under the INDI lock.
    bool reservationReleased() const;
    /// Publish a request rejection.
    int rejectRun( const std::string &reason /**< [in] Rejection reason. */ );
    /// Execute the production worker.
    void runWorker( bool start /**< [in] Target. */ );
    /// Launch and report queued work.
    void updateRunControl();
    /// Native command boundary.
    INDI_NEWCALLBACK_DECL( po4ao, m_indiP_run );
    /// Native observation boundary.
    INDI_SETCALLBACK_DECL( po4ao, m_indiP_nnRouting );
};
Po4aoRouteSnapshot RouteIO::capture()
{
    events.push_back( "capture" );
    return { current, "default", "" };
}
void RouteIO::switchInput( const std::string &target, Po4aoRouteBackend::Deadline )
{
    events.push_back( "switch" );
    current = target;
}
void RouteIO::applyDefault( Po4aoRouteBackend::Deadline )
{
    events.push_back( "default" );
    if( failCalibration )
    {
        throw std::runtime_error( "CACAO calibration restart not acknowledged" );
    }
}
void RouteIO::verifyConsumption( const std::string &, Po4aoRouteBackend::Deadline )
{
    events.push_back( "verify" );
    if( failVerify )
    {
        throw std::runtime_error( "stalled consumer" );
    }
}
po4ao::po4ao() : MagAOXApp<true>( 0, 0 )
{
    state( stateCodes::OPERATING );
    m_indiP_run.add( pcf::IndiElement( "toggle", pcf::IndiElement::On ) );
    for( const auto *element : { "phase", "route", "error", "owner", "verified" } )
    {
        m_indiP_runStatus.add( pcf::IndiElement( element ) );
    }
}
po4ao::~po4ao()
{
    m_runCancel = true;
    if( m_runThread.joinable() )
    {
        m_runThread.join();
    }
}
int po4ao::sendNewProperty( const pcf::IndiProperty &p )
{
    m_routeIO.events.push_back( p["action"].get<std::string>() );
    m_nnRoutingOwner = p["action"].get<std::string>() == "hold" ? m_runOwner : "";
    m_nnRoutingPhase = m_nnRoutingOwner.empty() ? "released" : "held";
    if( m_nnRoutingOwner.empty() )
    {
        m_nnReleasedOwner = m_runOwner;
    }
    m_nnRoutingSource = "nn";
    m_nnObserved      = std::chrono::steady_clock::now();
    return 0;
}
#include "../po4ao_run.hpp"
} // namespace po4aoRunHarness
/// \endcond

namespace libXWCTest::po4aoTest
{
/** \brief Failed consumption never parks the running monitor or acknowledges Stop. */
TEST_CASE( "Production PO4AO worker parks only after verified consumption", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao app;
    app.m_routeIO.failVerify = true;
    app.runWorker( false );
    REQUIRE( app.state() == MagAOX::app::stateCodes::OPERATING );
    REQUIRE( app.m_runAdmission.load() );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE_FALSE( app.m_runVerified );
    REQUIRE( app.m_runError.find( "stalled consumer" ) != std::string::npos );
    app.m_routeIO.failVerify = false;
    app.runWorker( false );
    app.updateRunControl();
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
    REQUIRE_FALSE( app.m_runAdmission.load() );
    REQUIRE( app.m_runPhase == "stopped" );
    REQUIRE( app.m_runVerified );
    REQUIRE( app.m_runSnapshot.writer == "nn" );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
#endif
    // clang-format on
}

/** \brief Stop drains app work without requiring a monitor mapping-closed acknowledgement. */
TEST_CASE( "PO4AO Stop waits for the active callback", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao       app;
    std::unique_lock<std::mutex> callbackLock( app.m_callbackMutex );
    auto                         worker   = std::async( std::launch::async,
                                                        [&]
                                                        {
                                  app.runWorker( false );
                                                        } );
    const auto                   deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 2 );
    while( app.m_runAdmission.load() && std::chrono::steady_clock::now() < deadline )
    {
        std::this_thread::yield();
    }
    const bool admissionClosed = !app.m_runAdmission.load();
    const bool waiting         = worker.wait_for( std::chrono::milliseconds( 20 ) ) == std::future_status::timeout;
    callbackLock.unlock();
    worker.get();
    REQUIRE( admissionClosed );
    REQUIRE( waiting );
    REQUIRE( app.m_runVerified );
    app.updateRunControl();
    REQUIRE( app.state() == MagAOX::app::stateCodes::READY );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
    MagAOX::app::po4ao::updateRunControl;
#endif
    // clang-format on
}

/** \brief Duplicate verified targets recheck consumption without repeating either utility. */
TEST_CASE( "Production PO4AO duplicate Stop does not recalibrate", "[po4ao][runWorker]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
#endif
    // clang-format on
    po4aoRunHarness::po4ao app;
    app.runWorker( false );
    app.m_routeIO.events.clear();
    app.m_runCheckOnly = true;
    app.runWorker( false );
    REQUIRE( app.m_routeIO.events == std::vector<std::string>{ "verify" } );
    REQUIRE( app.m_runPhase == "stopped" );
    app.m_routeIO.current = "po4ao_output";
    app.runWorker( false );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE_FALSE( app.m_runVerified );
}
/** \brief A rejected native request must terminate the GUI acknowledgement wait. */
TEST_CASE( "Run conflict rejection publishes Alert", "[po4ao][reviewRegression]" )
{
    po4aoRunHarness::po4ao               app;
    MagAOX::app::MagAOXApp<true>::Driver driver;
    app.m_indiDriver = &driver;
    SECTION( "reset" )
    {
        app.m_modelResetRequest = 2;
    }
    SECTION( "loading" )
    {
        app.m_reloadStatus = 1;
    }
    SECTION( "pending activation" )
    {
        app.m_reloadStatus = 2;
    }
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( app.m_indiP_run.getState() == INDI_ALERT );
    REQUIRE_FALSE( app.m_runPending );
    REQUIRE( app.m_runPhase == "unverified" );
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( driver.m_published.size() == 2 );
    REQUIRE( driver.m_published[0].getState() == INDI_ALERT );
    REQUIRE( driver.m_published[1].getState() == INDI_ALERT );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}

/** \brief Retrying a partial operation must preserve the token, instance and recovery snapshot. */
TEST_CASE( "Run retry retains unresolved reservation", "[po4ao][reviewRegression]" )
{
    po4aoRunHarness::po4ao app;
    app.m_runPhase = "failed";
    app.m_runError = "calibration incomplete";
    app.m_runOwner = app.m_nnRoutingOwner = "unresolved-owner";
    app.m_reservedInstance                = app.m_nnInstance;
    app.m_nnRoutingPhase                  = "held";
    app.m_runSnapshot                     = { "po4ao_output", "previous", "nn" };
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( app.m_runOwner == "unresolved-owner" );
    REQUIRE( app.m_reservedInstance == app.m_nnInstance );
    REQUIRE( app.m_runSnapshot.route == "po4ao_output" );
    REQUIRE( app.m_runSnapshot.calibration == "previous" );
    REQUIRE( app.m_runSnapshot.writer == "nn" );
    REQUIRE( app.m_runError == "calibration incomplete" );
    REQUIRE( app.m_indiP_runStatus["error"].get<std::string>().find( "calibration incomplete" ) != std::string::npos );
    REQUIRE_FALSE( app.m_runPending );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}
/** \brief Only a fresh exact-token release from the same NN instance permits a new routing operation. */
TEST_CASE( "Run retry requires matching completed recovery", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao app;
    app.m_runPhase         = "failed";
    app.m_runOwner         = "retained-token";
    app.m_reservedInstance = app.m_nnInstance;
    app.m_nnRoutingPhase   = "released";
    app.m_nnReleasedOwner  = app.m_runOwner;
    bool recovered         = false;
    SECTION( "matching recovery" )
    {
        recovered = true;
    }
    SECTION( "wrong token" )
    {
        app.m_nnReleasedOwner = "earlier-token";
    }
    SECTION( "stale recovery" )
    {
        app.m_nnObserved -= std::chrono::seconds( 6 );
    }
    SECTION( "NN restarted" )
    {
        app.m_nnInstance = "new-instance";
    }
    SECTION( "restoration still running" )
    {
        app.m_nnRoutingPhase = "restoring";
    }
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    MagAOX::app::MagAOXApp<true>::Driver driver;
    app.m_indiDriver = &driver;
    if( app.m_nnInstance != app.m_reservedInstance )
    {
        REQUIRE_THROWS( app.writerHeld( true ) );
    }
    else
    {
        REQUIRE( app.writerHeld( true ) == recovered );
    }
    REQUIRE( ( app.newCallBack_m_indiP_run( request ) == 0 ) == recovered );
    REQUIRE( app.m_runPending == recovered );
    if( recovered )
    {
        REQUIRE( app.m_runOwner != "retained-token" );
        REQUIRE( driver.m_published.back().getState() == INDI_BUSY );
    }
    else
    {
        REQUIRE( app.m_runOwner == "retained-token" );
    }
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::writerHeld;
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}

/** \brief Missing calibration restart acknowledgement retains the reservation and available PO4AO output. */
TEST_CASE( "Run worker does not release or park after incomplete calibration", "[po4ao][runWorker]" )
{
    po4aoRunHarness::po4ao app;
    app.m_routeIO.failCalibration = true;
    app.runWorker( false );
    REQUIRE( app.m_runPhase == "failed" );
    REQUIRE( app.m_nnRoutingPhase == "held" );
    REQUIRE( app.m_nnRoutingOwner == app.m_runOwner );
    REQUIRE( app.state() == MagAOX::app::stateCodes::OPERATING );
    REQUIRE_FALSE( app.m_runVerified );
    REQUIRE( app.m_routeIO.events == std::vector<std::string>{ "capture", "hold", "switch", "default" } );
    pcf::IndiProperty request( pcf::IndiProperty::Switch );
    request.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
    const auto owner = app.m_runOwner;
    REQUIRE( app.newCallBack_m_indiP_run( request ) < 0 );
    REQUIRE( app.m_runOwner == owner );
    REQUIRE( app.m_runSnapshot.writer == "nn" );
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::runWorker;
    MagAOX::app::po4ao::newCallBack_m_indiP_run;
#endif
    // clang-format on
}
} // namespace libXWCTest::po4aoTest

namespace libXWCTest::po4aoTest
{
/** \brief Source gaps retain finite pass-through values while enforcing controlled limits and all-mode finiteness. */
TEST_CASE( "PO4AO source gaps use common output validation", "[po4ao][control][regression]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
#endif
    // clang-format on
    for( const auto observation :
         { std::array<float, 3>{ 2.F, -2.F, 99.F }, std::array<float, 3>{ 100.F, -100.F, 99.F } } )
    {
        po4aoControlHarness::po4ao app;
        REQUIRE( app.frame( 1 ) == 0 );
        auto input                                                          = observation;
        po4aoControlHarness::po4ao::shmimMonitorT::m_imageStream.md[0].cnt0 = 3;
        REQUIRE( app.processImage( input.data(), po4aoControlHarness::dev::shmimT{} ) == 0 );
        REQUIRE( app.m_emitted.back()[0] == std::min( observation[0], 4.F ) );
        REQUIRE( app.m_emitted.back()[1] == std::max( observation[1], -4.F ) );
        REQUIRE( app.m_emitted.back()[2] == 99.F );
        REQUIRE( app.m_effectiveController.load() == 3 );
    }
    // Inject stream bytes, avoiding fast-math folding of floating-point infinity constants.
    for( uint32_t invalidBits : { 0x7FC00000U, 0x7F800000U } )
    {
        for( size_t mode = 0; mode < 3; ++mode )
        {
            po4aoControlHarness::po4ao app;
            REQUIRE( app.frame( 1 ) == 0 );
            std::array<float, 3> input{ 2.F, -2.F, 99.F };
            memcpy( &input[mode], &invalidBits, sizeof( invalidBits ) );
            uint32_t observedBits = 0;
            memcpy( &observedBits, &input[mode], sizeof( observedBits ) );
            REQUIRE( observedBits == invalidBits );
            po4aoControlHarness::po4ao::shmimMonitorT::m_imageStream.md[0].cnt0 = 3;
            REQUIRE( app.processImage( input.data(), po4aoControlHarness::dev::shmimT{} ) == 0 );
            REQUIRE( app.m_emitted.back() == std::array<float, 3>{ 0.F, 0.F, 0.F } );
            REQUIRE( app.m_effectiveController.load() == 0 );
        }
    }
}

/** \brief Failed output retains the reset request, and an acknowledgement never consumes a newer reset. */
TEST_CASE( "PO4AO reset acknowledgement survives output failure and supersession", "[po4ao][control][regression]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::processImage;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    REQUIRE( app.frame( 1 ) == 0 );
    app.m_modelGeneration   = 2;
    app.m_modelResetRequest = 2;
    SECTION( "failed publication retries the same reset" )
    {
        app.m_failOutput = true;
        REQUIRE( app.frame( 2 ) == -1 );
        REQUIRE( app.m_appliedReset.load() == 1 );
        REQUIRE( app.m_modelResetRequest.load() == 2 );
        app.m_failOutput = false;
        REQUIRE( app.frame( 3 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 2 );
        REQUIRE( app.m_modelResetRequest.load() == 0 );
        REQUIRE( app.frame( 4 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 2 );
    }
    SECTION( "later reset supersedes a failed publication" )
    {
        app.m_failOutput = true;
        REQUIRE( app.frame( 2 ) == -1 );
        app.m_modelGeneration   = 3;
        app.m_modelResetRequest = 3;
        app.m_failOutput        = false;
        REQUIRE( app.frame( 3 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 3 );
        REQUIRE( app.m_modelResetRequest.load() == 0 );
    }
    SECTION( "new reset during successful output remains pending" )
    {
        app.m_duringOutput = [&]
        {
            app.m_modelGeneration   = 3;
            app.m_modelResetRequest = 3;
        };
        REQUIRE( app.frame( 2 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 2 );
        REQUIRE( app.m_modelResetRequest.load() == 3 );
        app.m_duringOutput = {};
        REQUIRE( app.frame( 3 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 3 );
        REQUIRE( app.m_modelResetRequest.load() == 0 );
    }
    SECTION( "disabled callback admission cannot acknowledge output" )
    {
        app.m_runAdmission = false;
        REQUIRE( app.frame( 2 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 1 );
        REQUIRE( app.m_modelResetRequest.load() == 2 );
        REQUIRE( app.m_emitted.size() == 1 );
        app.m_runAdmission = true;
        REQUIRE( app.frame( 3 ) == 0 );
        REQUIRE( app.m_appliedReset.load() == 2 );
        REQUIRE( app.m_modelResetRequest.load() == 0 );
    }
}
} // namespace libXWCTest::po4aoTest

namespace libXWCTest::po4aoTest
{
/** \brief Rejected model commands publish Alert even for repeated requests and retain it through status polling. */
TEST_CASE( "PO4AO model commands acknowledge routing conflicts", "[po4ao][control][regression]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::newCallBack_m_indiP_reloadToggle;
    MagAOX::app::po4ao::newCallBack_m_indiP_resetModel;
    MagAOX::app::po4ao::newCallBack_m_indiP_controlToggle;
    MagAOX::app::po4ao::newCallBack_m_indiP_recollectWarmup;
    MagAOX::app::po4ao::updateControlStatus;
#endif
    // clang-format on
    using App      = po4aoControlHarness::po4ao;
    using Callback = int ( App::* )( const pcf::IndiProperty & );
    const std::array<std::pair<pcf::IndiProperty App::*, Callback>, 4> commands{
        { { &App::m_indiP_reloadToggle, &App::newCallBack_m_indiP_reloadToggle },
          { &App::m_indiP_resetModel, &App::newCallBack_m_indiP_resetModel },
          { &App::m_indiP_controlToggle, &App::newCallBack_m_indiP_controlToggle },
          { &App::m_indiP_recollectWarmup, &App::newCallBack_m_indiP_recollectWarmup } } };
    for( const auto &[member, callback] : commands )
    {
        App                                  app;
        MagAOX::app::MagAOXApp<true>::Driver driver;
        app.m_indiDriver = &driver;
        REQUIRE( app.frame( 1 ) == 0 );
        app.updateControlStatus();
        auto      &property = app.*member;
        const auto applied  = property["toggle"].getSwitchState();
        auto       request  = property;
        request["toggle"].setSwitchState( member == &App::m_indiP_controlToggle ? pcf::IndiElement::Off
                                                                                : pcf::IndiElement::On );
        app.m_runBusy = true;
        for( int retry = 0; retry < 2; ++retry )
        {
            REQUIRE( ( app.*callback )( request ) == -1 );
            REQUIRE( driver.m_published.size() == static_cast<size_t>( retry + 1 ) );
            REQUIRE( driver.m_published.back().getName() == property.getName() );
            REQUIRE( driver.m_published.back().getState() == INDI_ALERT );
            REQUIRE( driver.m_published.back()["toggle"].getSwitchState() == applied );
            app.updateControlStatus();
            REQUIRE( property.getState() == INDI_ALERT );
        }
        REQUIRE( app.m_modelResetRequest.load() == 0 );
        REQUIRE( app.m_modelGeneration.load() == 1 );
        REQUIRE( app.m_controlRequest.load() == 3 );
        REQUIRE_FALSE( app.m_reloadRequested.load() );
        REQUIRE( app.m_warmupRecollectRequest.load() == 0 );
        app.m_runBusy = false;
        app.updateControlStatus();
        REQUIRE( property.getState() == INDI_ALERT );
        REQUIRE( ( app.*callback )( request ) == 0 );
        REQUIRE( property.getState() == INDI_BUSY );
        app.updateControlStatus();
        REQUIRE( property.getState() == INDI_BUSY );
    }
    SECTION( "accepted retry of an applied target still acknowledges the request" )
    {
        App app;
        REQUIRE( app.frame( 1 ) == 0 );
        app.updateControlStatus();
        auto request  = app.m_indiP_controlToggle;
        app.m_runBusy = true;
        REQUIRE( app.newCallBack_m_indiP_controlToggle( request ) == -1 );
        app.m_runBusy = false;
        REQUIRE( app.newCallBack_m_indiP_controlToggle( request ) == 0 );
        REQUIRE( app.m_indiP_controlToggle.getState() == INDI_BUSY );
        REQUIRE( app.m_controlRequest.load() == 3 );
        app.updateControlStatus();
        REQUIRE( app.m_indiP_controlToggle.getState() == INDI_OK );
    }
    SECTION( "recollection while stopped is explicitly rejected" )
    {
        App                                  app;
        MagAOX::app::MagAOXApp<true>::Driver driver;
        app.m_indiDriver = &driver;
        app.m_runRunning = false;
        auto request     = app.m_indiP_recollectWarmup;
        request["toggle"].setSwitchState( pcf::IndiElement::On );
        REQUIRE( app.newCallBack_m_indiP_recollectWarmup( request ) == -1 );
        REQUIRE( driver.m_published.size() == 1 );
        app.updateControlStatus();
        REQUIRE( app.m_indiP_recollectWarmup.getState() == INDI_ALERT );
    }
}
} // namespace libXWCTest::po4aoTest

namespace libXWCTest::po4aoTest
{
/** \brief A parked app acknowledges reset without a source frame and resumes without reviving the old request. */
TEST_CASE( "PO4AO reset while stopped remains independent of frame publication", "[po4ao][control][regression]" )
{
    // clang-format off
#ifdef PO4AO_TEST_DOXYGEN_REF
    MagAOX::app::po4ao::appLogic;
    MagAOX::app::po4ao::processImage;
#endif
    // clang-format on
    po4aoControlHarness::po4ao app;
    REQUIRE( app.frame( 1 ) == 0 );
    app.m_runRunning        = false;
    app.m_runAdmission      = false;
    app.m_modelGeneration   = 2;
    app.m_modelResetRequest = 2;
    REQUIRE( app.appLogic() == 0 );
    REQUIRE( app.m_appliedReset.load() == 2 );
    REQUIRE( app.m_modelResetRequest.load() == 0 );
    REQUIRE( app.m_emitted.size() == 1 );
    app.m_runRunning   = true;
    app.m_runAdmission = true;
    REQUIRE( app.frame( 2 ) == 0 );
    REQUIRE( app.m_appliedReset.load() == 2 );
    REQUIRE( app.m_modelResetRequest.load() == 0 );
}
} // namespace libXWCTest::po4aoTest
