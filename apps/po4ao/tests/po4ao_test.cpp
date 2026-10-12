/** \file po4ao_test.cpp
 * \brief Catch2 tests for PO4AO history ordering and command safety.
 *
 * \ingroup po4ao_files
 */

/** \defgroup po4ao_unit_test PO4AO Unit Tests
 * \brief Unit tests for the PO4AO application.
 *
 * \ingroup application_unit_test
 */

#include "../../../tests/testXWC.hpp"

#include "../po4ao_utils.hpp"

#include <cmath>
#include <limits>

namespace libXWCTest
{
namespace po4aoTest
{

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

    float       command[]     = { 10.F, 6.F };

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

} // namespace po4aoTest
} // namespace libXWCTest
