/** \file alpaoCtrl_test.cpp
 * \brief Offline ALPAO SDK outcome and timing endpoint tests with isolated shared memory.
 * \ingroup alpaoCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include "sdk/asdkWrapper.h"

/// \cond TEST_HARNESS
namespace alpaoTimingTest
{
int      result{ 0 };     ///< Scripted SDK return value.
unsigned calls{ 0 };      ///< SDK submissions made by commandDM.
int64_t  firstEntry{ 0 }; ///< SDK entry after the start marker.
int64_t  lastReturn{ 0 }; ///< SDK completion before the end marker.
asdkDM   mirror;          ///< Software-only mirror identity.
} // namespace alpaoTimingTest

/// Return a software handle without initializing hardware.
asdkDM *asdkInit( const char * )
{
    return &alpaoTimingTest::mirror;
}

/// Return an empty software SDK error state.
void asdkGetLastError( UInt *code, char *text, UInt size )
{
    *code = 0;
    if( text && size )
    {
        text[0] = 0;
    }
}

/// Report the two software actuators.
COMPL_STAT asdkGet( asdkDM *, const char *, Scalar *value )
{
    *value = 2;
    return 0;
}

/// Record the actual command-submission boundary without accessing hardware.
COMPL_STAT asdkSend( asdkDM *, const Scalar * )
{
    ++alpaoTimingTest::calls;
    alpaoTimingTest::firstEntry = MagAOX::app::dev::dmTiming::monotonicNowNs();
    alpaoTimingTest::lastReturn = MagAOX::app::dev::dmTiming::monotonicNowNs();

    return alpaoTimingTest::result;
}

/// Complete a software-only reset.
COMPL_STAT asdkReset( asdkDM * )
{
    return 0;
}

/// Complete software-only teardown.
COMPL_STAT asdkRelease( asdkDM * )
{
    return 0;
}

#include "../alpaoCtrl.hpp"

namespace MagAOX
{
namespace app
{
/// Prepare the real command implementation with two software actuators and no application threads.
class alpaoCtrl_test : public alpaoCtrl
{
    IMAGE m_shape{}; ///< Owned local output-shape stream.
  public:
    using dev::dm<alpaoCtrl, float>::m_dmTiming;
    /// Allocate a unique local stream and the production command scratch storage.
    alpaoCtrl_test();
    /// Remove only the test-owned stream before the production destructor releases scratch storage.
    ~alpaoCtrl_test() noexcept;
};

alpaoCtrl_test::alpaoCtrl_test()
{
    static const std::string directory = []()
    {
        char        path[] = "/tmp/alpao-timing-test-XXXXXX";
        const char *value  = mkdtemp( path );

        if( !value )
        {
            throw std::runtime_error( "cannot create local ALPAO stream directory" );
        }

        return std::string( value );
    }();
    setenv( "MILK_SHM_DIR", directory.c_str(), 1 );
    uint32_t sizes[]{ 2, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_shape, "alpao_timing_shape", 2, sizes, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0 ) == 0 );
    m_outputShape.open( "alpao_timing_shape" );
    m_imageStream         = {};
    m_nbAct               = 2;
    m_max_stroke          = 1;
    m_volume_factor       = 1;
    m_actuator_mapping    = static_cast<int *>( calloc( 2, sizeof( int ) ) );
    m_actuator_mapping[1] = 1;
    m_dminputs            = static_cast<Scalar *>( calloc( 2, sizeof( Scalar ) ) );
    m_instSatMap.resize( 2, 1 );
    m_instSatMap.setZero();
    m_dmTiming.configure( 2.0, 32 );
    alpaoTimingTest::calls = 0;
}

alpaoCtrl_test::~alpaoCtrl_test() noexcept
{
    m_outputShape.close();
    ImageStreamIO_destroyIm( &m_shape );
}
} // namespace app
} // namespace MagAOX

/// \endcond

namespace libXWCTest
{
namespace alpaoCtrlTest
{
/** \defgroup alpaoCtrl_unit_test alpaoCtrl Unit Tests
 * \brief Offline tests of ALPAO command integration.
 * \ingroup application_unit_test
 */

/// Qualify the SDK result immediately while preserving ALPAO's subsequent saturation-map processing.
/** \ingroup alpaoCtrl_unit_test */
TEST_CASE( "alpaoCtrl records actual SDK outcomes and phase endpoints", "[alpaoCtrl][timing]" )
{
    // clang-format off
    #ifdef ALPAOCTRL_TEST_DOXYGEN_REF
    MagAOX::app::alpaoCtrl::commandDM(nullptr);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;

    alpaoCtrl_test app;
    alpaoTimingTest::result = GENERATE( 0, -1 );
    float command[]{ -2, 2 };

    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.commandDM( command ) == alpaoTimingTest::result );
    app.m_dmTiming.mark( Marker::finish );
    app.m_dmTiming.finishEvent();
    app.m_dmTiming.update();

    const auto &summary = app.dmTimingSummary();

    REQUIRE( alpaoTimingTest::calls == 1 );
    REQUIRE( summary.total.submitted == ( alpaoTimingTest::result ? 0 : 1 ) );
    REQUIRE( summary.total.failed == ( alpaoTimingTest::result ? 1 : 0 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::actuator )].count == 1 );
    REQUIRE( summary.metrics[metricIndex( Metric::saturation )].count == 1 );

    const auto &sdk = summary.metrics[metricIndex( Metric::sdk )];

    REQUIRE( sdk.count == 1 );
    REQUIRE( sdk.mean >= ( alpaoTimingTest::lastReturn - alpaoTimingTest::firstEntry ) * 1e-9 );
}
} // namespace alpaoCtrlTest
} // namespace libXWCTest
