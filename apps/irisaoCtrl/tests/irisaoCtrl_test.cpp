/** \file irisaoCtrl_test.cpp
 * \brief Offline IrisAO endpoint tests preserving unknown SDK outcome semantics.
 * \ingroup irisaoCtrl_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include "sdk/irisao.mirrors.h"

/// \cond TEST_HARNESS
namespace irisaoTimingTest
{
unsigned staged{ 0 };     ///< Segment staging calls preceding submission.
unsigned calls{ 0 };      ///< Final submissions made by commandDM.
int64_t  firstEntry{ 0 }; ///< SDK entry after the start marker.
int64_t  lastReturn{ 0 }; ///< SDK completion before the end marker.
} // namespace irisaoTimingTest

/// Return a software-only mirror identity.
MirrorHandle MirrorConnect( const char *, const char *, bool )
{
    return nullptr;
}

/// Expose one software segment to the production iterator.
bool MirrorIterate( MirrorHandle, SegmentNumber segment )
{
    return segment == 0;
}

/// Record one segment's preparation without hardware access.
void SetMirrorPosition( MirrorHandle, SegmentNumber, double, double, double )
{
    ++irisaoTimingTest::staged;
}

/// Make the software segment reachable.
void GetMirrorPosition( MirrorHandle, SegmentNumber, MirrorPosition *position )
{
    position->reachable = true;
}

/// Record the final submission without inventing success semantics for the vendor API.
void MirrorCommand( MirrorHandle, int command )
{
    REQUIRE( command == MirrorSendSettings );
    ++irisaoTimingTest::calls;
    irisaoTimingTest::firstEntry = MagAOX::app::dev::dmTiming::monotonicNowNs();
    irisaoTimingTest::lastReturn = MagAOX::app::dev::dmTiming::monotonicNowNs();
}

/// Complete software-only teardown.
void MirrorRelease( MirrorHandle )
{
}

#include "../irisaoCtrl.hpp"

namespace MagAOX
{
namespace app
{
/// Prepare the production IrisAO command path without connecting a mirror or starting threads.
class irisaoCtrl_test : public irisaoCtrl
{
  public:
    using dev::dm<irisaoCtrl, float>::m_dmTiming;
    /// Initialize the one-segment scratch state and a bounded collector.
    irisaoCtrl_test();
    /// Select the existing SDK hardware-disable configuration.
    void disabled( bool value /**< [in] Whether the SDK is configured to disable hardware. */ );
};

irisaoCtrl_test::irisaoCtrl_test()
{
    m_imageStream = {};
    m_dm          = {};
    m_instSatMap.resize( 3, 1 );
    m_instSatMap.setZero();
    m_dmTiming.configure( 2.0, 32 );
    irisaoTimingTest::calls  = 0;
    irisaoTimingTest::staged = 0;
}

void irisaoCtrl_test::disabled( bool value )
{
    m_hardwareDisable = value;
}
} // namespace app
} // namespace MagAOX

/// \endcond

namespace libXWCTest
{
namespace irisaoCtrlTest
{
/** \defgroup irisaoCtrl_unit_test irisaoCtrl Unit Tests
 * \brief Offline tests of IrisAO command integration.
 * \ingroup application_unit_test
 */

/// Measure the final SDK call while retaining explicit unqualified or disabled submission outcomes.
/** \ingroup irisaoCtrl_unit_test */
TEST_CASE( "irisaoCtrl never treats unqualified SDK returns as successful submissions", "[irisaoCtrl][timing]" )
{
    // clang-format off
    #ifdef IRISAOCTRL_TEST_DOXYGEN_REF
    MagAOX::app::irisaoCtrl::commandDM(nullptr);
    #endif
    // clang-format on
    using namespace MagAOX::app;
    using namespace MagAOX::dmTiming;

    irisaoCtrl_test app;
    const bool      disabled = GENERATE( false, true );
    app.disabled( disabled );
    float command[]{ 1, 2, 3 };

    app.m_dmTiming.beginEvent( true );
    REQUIRE( app.commandDM( command ) == 0 );
    REQUIRE( app.m_dmTiming.outcome() == ( disabled ? Outcome::noSend : Outcome::unknown ) );
    app.m_dmTiming.mark( Marker::finish );
    app.m_dmTiming.finishEvent();
    app.m_dmTiming.update();

    const auto &summary = app.dmTimingSummary();

    REQUIRE( irisaoTimingTest::staged == 1 );
    REQUIRE( irisaoTimingTest::calls == 1 );
    REQUIRE( summary.total.submitted == 0 );
    REQUIRE( summary.total.noSend == ( disabled ? 1 : 0 ) );
    REQUIRE( summary.metrics[metricIndex( Metric::receiptSend )].count == 0 );
    REQUIRE( summary.metrics[metricIndex( Metric::actuator )].count == 1 );
    REQUIRE( summary.metrics[metricIndex( Metric::saturation )].count == 0 );

    const auto &sdk = summary.metrics[metricIndex( Metric::sdk )];

    REQUIRE( sdk.count == 1 );
    REQUIRE( sdk.mean >= ( irisaoTimingTest::lastReturn - irisaoTimingTest::firstEntry ) * 1e-9 );
}
} // namespace irisaoCtrlTest
} // namespace libXWCTest
