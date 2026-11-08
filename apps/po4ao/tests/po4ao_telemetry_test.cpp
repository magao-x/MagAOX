/** \file po4ao_telemetry_test.cpp
 * \brief Round-trip the generated production PO4AO schema without GPU dependencies.
 * \ingroup po4ao_unit_test
 */
#include "../../../tests/catch2/catch.hpp"
#include "../../../libMagAOX/logger/types/generated/telem_po4ao_generated.h"

namespace libXWCTest::po4aoTest
{
/** \brief Exact integer identity/counters survive serialization, including empty and idle windows. */
TEST_CASE( "PO4AO telemetry round trip preserves exact counters and identity", "[po4ao][telemetry]" )
{
    using namespace MagAOX::logger;
    for( bool fresh : { false, true } )
        for( bool mixed : { false, true } )
        {
            constexpr uint64_t             exact = UINT64_MAX - 31;
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish( CreateTelem_po4ao_fb( builder,
                                                  0,
                                                  4,
                                                  5,
                                                  6,
                                                  1024,
                                                  exact,
                                                  exact - 1,
                                                  fresh,
                                                  mixed,
                                                  exact - 2,
                                                  exact - 3,
                                                  exact - 4,
                                                  exact - 5,
                                                  100,
                                                  exact - 6,
                                                  2 ) );
            flatbuffers::Verifier verifier( builder.GetBufferPointer(), builder.GetSize() );
            REQUIRE( VerifyTelem_po4ao_fbBuffer( verifier ) );
            const auto *s = GetTelem_po4ao_fb( builder.GetBufferPointer() );
            REQUIRE( s->p50_us() == 0 );
            REQUIRE( s->p95_us() == 4 );
            REQUIRE( s->p99_us() == 5 );
            REQUIRE( s->max_us() == 6 );
            REQUIRE( s->samples() == 1024 );
            REQUIRE( s->sequence() == exact );
            REQUIRE( s->sample_time_ns() == exact - 1 );
            REQUIRE( s->fresh() == fresh );
            REQUIRE( s->mixed_window() == mixed );
            REQUIRE( s->missed_frames() == exact - 2 );
            REQUIRE( s->deadline_misses() == exact - 3 );
            REQUIRE( s->inference_failures() == exact - 4 );
            REQUIRE( s->replay_drops() == exact - 5 );
            REQUIRE( s->deadline_us() == 100 );
            REQUIRE( s->model_id() == exact - 6 );
            REQUIRE( s->backend() == 2 );
            builder.GetBufferPointer()[0] = 0xff;
            flatbuffers::Verifier invalid( builder.GetBufferPointer(), builder.GetSize() );
            REQUIRE_FALSE( VerifyTelem_po4ao_fbBuffer( invalid ) );
        }
    flatbuffers::FlatBufferBuilder empty;
    empty.Finish( CreateTelem_po4ao_fb( empty ) );
    const auto *s = GetTelem_po4ao_fb( empty.GetBufferPointer() );
    REQUIRE( s->samples() == 0 );
    REQUIRE( s->sequence() == 0 );
    REQUIRE_FALSE( s->fresh() );
    REQUIRE_FALSE( s->mixed_window() );
}
} // namespace libXWCTest::po4aoTest
