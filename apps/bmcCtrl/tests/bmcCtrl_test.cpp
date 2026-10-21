/** \file bmcCtrl_test.cpp
 * \brief Offline BMC SDK outcome and timing endpoint tests with isolated shared memory.
 * \ingroup bmcCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include "sdk/BMCApi.h"

/// \cond TEST_HARNESS
namespace bmcTimingTest
{
int      result{ 0 };     ///< Scripted SDK return value.
unsigned calls{ 0 };      ///< SDK submissions made by commandDM.
int64_t  firstEntry{ 0 }; ///< SDK entry after the start marker.
int64_t  lastReturn{ 0 }; ///< SDK completion before the end marker.
} // namespace bmcTimingTest

/// Complete software-only initialization.
BMCRC BMCOpen( DM *, const char * )
{
    return NO_ERR;
}

/// Supply the two test actuator addresses.
BMCRC BMCLoadMap( DM *, const char *, uint32_t *map )
{
    map[0] = 0;
    map[1] = 1;

    return NO_ERR;
}

/// Record the actual submission boundary without accessing hardware.
BMCRC BMCSetArray( DM *, const double *, const void * )
{
    ++bmcTimingTest::calls;
    bmcTimingTest::firstEntry = MagAOX::app::dev::dmTiming::monotonicNowNs();
    bmcTimingTest::lastReturn = MagAOX::app::dev::dmTiming::monotonicNowNs();

    return bmcTimingTest::result;
}

/// Describe a scripted failure.
const char *BMCErrorString( BMCRC )
{
    return "offline SDK failure";
}

/// Complete a software-only clear.
BMCRC BMCClearArray( DM * )
{
    return NO_ERR;
}

/// Complete software-only teardown.
BMCRC BMCClose( DM * )
{
    return NO_ERR;
}

#include "../bmcCtrl.hpp"

namespace MagAOX
{
namespace app
{
/// Prepare the real command implementation with two software actuators and no application threads.
class bmcCtrl_test : public bmcCtrl
{
    IMAGE m_shape{}; ///< Owned local output-shape stream.
  public:
    using dev::dm<bmcCtrl, float>::m_dmTiming;
    /// Allocate a unique local output stream and the production command scratch storage.
    bmcCtrl_test();
    /// Remove only the test-owned stream before the production destructor releases scratch storage.
    ~bmcCtrl_test() noexcept;
};

bmcCtrl_test::bmcCtrl_test()
{
    static const std::string directory = []()
    {
        char        path[] = "/tmp/bmc-timing-test-XXXXXX";
        const char *value  = mkdtemp( path );

        if( !value )
        {
            throw std::runtime_error( "cannot create local BMC stream directory" );
        }

        return std::string( value );
    }();
    setenv( "MILK_SHM_DIR", directory.c_str(), 1 );
    uint32_t sizes[]{ 2, 1 };
    REQUIRE( ImageStreamIO_createIm_gpu(
                 &m_shape, "bmc_timing_shape", 2, sizes, _DATATYPE_FLOAT, -1, 1, 10, 0, 0, 0 ) == 0 );
    m_outputShape.open( "bmc_timing_shape" );
    m_imageStream         = {};
    m_nbAct               = 2;
    m_act_gain            = 1;
    m_volume_factor       = 1;
    m_actuator_mapping    = static_cast<int *>( calloc( 2, sizeof( int ) ) );
    m_actuator_mapping[1] = 1;
    m_dminputs            = static_cast<double *>( calloc( 2, sizeof( double ) ) );
    m_instSatMap.resize( 2, 1 );
    m_instSatMap.setZero();
    m_dmTiming.configure( 2.0, 32 );
    bmcTimingTest::calls = 0;
}

bmcCtrl_test::~bmcCtrl_test() noexcept
{
    m_outputShape.close();
    ImageStreamIO_destroyIm( &m_shape );
}
} // namespace app
} // namespace MagAOX

/// \endcond

namespace libXWCTest
{
namespace bmcCtrlTest
{
/** \defgroup bmcCtrl_unit_test bmcCtrl Unit Tests
 * \brief Offline tests of BMC command integration.
 * \ingroup application_unit_test
 */

/// Preserve SDK return behavior and qualify submission before the later saturation-map epilogue.
/** \ingroup bmcCtrl_unit_test */
TEST_CASE( "bmcCtrl records actual SDK outcomes and phase endpoints", "[bmcCtrl][timing]" )
{
    // clang-format off
    #ifdef BMCCTRL_TEST_DOXYGEN_REF
    MagAOX::app::bmcCtrl::commandDM(nullptr);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;

    bmcCtrl_test app;
    bmcTimingTest::result = GENERATE( 0, 7 );
    float command[]{ 0.25f, 2.0f };

    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.commandDM( command ) == ( bmcTimingTest::result ? -1 : 0 ) );
    app.m_dmTiming.mark( Marker::finish );
    app.m_dmTiming.finishEvent();
    app.m_dmTiming.update();

    const auto &summary = app.dmTimingSummary();

    REQUIRE( bmcTimingTest::calls == 1 );
    REQUIRE( summary.total.submitted == ( bmcTimingTest::result ? 0 : 1 ) );
    REQUIRE( summary.total.failed == ( bmcTimingTest::result ? 1 : 0 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::actuator )].count == 1 );
    REQUIRE( summary.metrics[metricIndex( Metric::saturation )].count == ( bmcTimingTest::result ? 0 : 1 ) );

    const auto &sdk = summary.metrics[metricIndex( Metric::sdk )];

    REQUIRE( sdk.count == 1 );
    REQUIRE( sdk.mean >= ( bmcTimingTest::lastReturn - bmcTimingTest::firstEntry ) * 1e-9 );
}
} // namespace bmcCtrlTest
} // namespace libXWCTest
