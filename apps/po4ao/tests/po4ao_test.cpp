/** \file po4ao_test.cpp
 * \brief Catch2 tests for PO4AO model handoff, modal history, and command safety.
 *
 * \ingroup po4ao_files
 */

#include "../../../tests/testXWC.hpp"

#include "../po4ao_utils.hpp"

#include <cmath>
#include <limits>

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
        REQUIRE( input[index] == expected[index] );
}

/// Verify a short policy uses the newest samples from a larger retained history.
/** \ingroup po4ao_unit_test
 */
TEST_CASE( "PO4AO assembles a shorter active policy history", "[po4ao]" )
{
    CircularBuffer observations( 1, 4 );
    CircularBuffer commands( 1, 3 );
    for( float value : { 1.F, 2.F, 3.F, 4.F } )
        observations.add( &value );
    for( float value : { 10.F, 20.F, 30.F } )
        commands.add( &value );

    float input[3] = {};
    REQUIRE( assemblePolicyInput( input, 1, 2, observations, commands ) );
    const float expected[] = { 4.F, 3.F, 30.F };
    for( size_t index = 0; index < 3; ++index )
        REQUIRE( input[index] == expected[index] );
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
        REQUIRE( input[index] == expected[index] );
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
    collection.validate( 3, 42, 1, true );
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
    collection.validate( 4, 0, 2, false );
    REQUIRE( WarmupCollection::phase( collection.token() ) == CollectionPhase::ready );
}

/// Verify control-Off integrator collection neither activates nor requires a retained policy.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO can collect under the disabled integrator", "[po4ao]" )
{
    WarmupCollection collection;
    collection.begin( 2, 1, 4, 0, 1 );
    collection.validate( 4, 0, 1, true );
    collection.emitted();
    REQUIRE( collection.published( 2 ) );
    REQUIRE( collection.frames() == 1 );
}

/// Verify incompatible samples, failed publication and reset cannot complete a fixed-policy dataset.
/** \ingroup po4ao_unit_test */
TEST_CASE( "PO4AO interrupts collection on controller changes and source gaps", "[po4ao]" )
{
    WarmupCollection collection;
    collection.begin( 5, 2, 3, 42, 1 );
    collection.emitted();
    SECTION( "control changed" )
    {
        collection.validate( 4, 42, 1, true );
    }
    SECTION( "policy changed" )
    {
        collection.validate( 3, 43, 1, true );
    }
    SECTION( "inference fell back" )
    {
        collection.validate( 3, 0, 1, true );
    }
    SECTION( "reset" )
    {
        collection.validate( 3, 42, 2, true );
    }
    SECTION( "source gap" )
    {
        collection.validate( 3, 42, 1, false );
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
