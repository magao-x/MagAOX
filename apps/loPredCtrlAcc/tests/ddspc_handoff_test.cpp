/** \file ddspc_handoff_test.cpp
 * \brief Verify explicit copy-gain correction without changing the default Legacy compatibility path.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "legacy_replay.hpp"

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief An explicit opt-out fixes copy_gain loading while missing keys and default Legacy output stay compatible. */
TEST_CASE( "Handoff copy gain correction is explicit and backward compatible", "[loPredCtrlAcc][app][handoff]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    MagAOX::app::loPredCtrlAcc::loadConfigImpl;
    MagAOX::app::loPredCtrlAcc::processImage;
#endif
    // clang-format on
    for( const auto compat : { true, false } )
        for( const auto supplied : { true, false } )
        {
            auto                               app = current_legacy_app();
            std::map<std::string, std::string> config{ { "parameters.gain", "0.3" },
                                                       { "parameters.num_modes", "2" },
                                                       { "parameters.legacy_copy_gain_compat", compat ? "1" : "0" } };
            if( supplied )
                config["parameters.copy_gain"] = "1.0";
            REQUIRE( app->initialize( config, 2 ) == 0 );
            const float expected = !compat && supplied ? 1.f : .3f;
            REQUIRE( app->state_snapshot().m_copyGain == expected );
            ReplayMatrix residual( 2, 1 );
            residual << .1f, -.2f;
            const auto frame = app->frame( residual, true ); // Offline stream-free lifecycle double only.
            REQUIRE( ( frame.m_output.array() == ( expected * residual ).array() ).all() );
        }
}
} //namespace libXWCTest::loPredCtrlAccTest
