/** \file confirmed_replay.cpp
 * \brief Actual accelSim numerical forcing and passive, causal replay through the real shadow input path.
 */
#include "confirmed_replay.hpp"
#include <accelSimModel.hpp>
#include <algorithm>
#include <chrono>
#include <stdexcept>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
constexpr std::int64_t origin = 1000000000, epoch = 1800000000000000000LL;

/// Owned temporal image adapter: no ImageStreamIO, files, publication API, runtime or hardware.
class MemoryAdc : public DDSPC::AdcSource
{
  public:
    DDSPC::AdcHeader                 m_header{ 3, 2, 1, 128, 0, 0, true, true, true, false }; ///< Exact source layout.
    std::array<DDSPC::AdcFrame, 128> m_frames{}; ///< Original ordered temporal planes.
    DDSPC::AdcClock                  m_clock;    ///< Explicit common-clock mapping observation.

    /// Copy immutable source metadata.
    DDSPC::AdcHeader header() const override;

    /// Copy one temporal plane and its original metadata.
    DDSPC::AdcFrame frame( std::size_t index /**< [in] Retained temporal slot. */ ) const override;

    /// Copy the explicitly synthetic clock observation.
    DDSPC::AdcClock clock() const override;
};
DDSPC::AdcHeader MemoryAdc::header() const
{
    return m_header;
}
DDSPC::AdcFrame MemoryAdc::frame( std::size_t index ) const
{
    return m_frames.at( index );
}
DDSPC::AdcClock MemoryAdc::clock() const
{
    return m_clock;
}

/// Construct independently known synthetic identities and availability times.
DDSPC::InputStamp stamp( const DDSPC::InputConfig &c /**< [in] Fixed synthetic contract. */,
                         std::uint64_t             generation /**< [in] Producer incarnation. */,
                         std::uint64_t             counter /**< [in] Immutable frame identity. */,
                         std::int64_t              available /**< [in] Publication in common monotonic ns. */ )
{
    DDSPC::InputStamp s;
    s.m_generation    = generation;
    s.m_counter       = counter;
    s.m_clockEpoch    = c.m_clockEpoch;
    s.m_layout        = c.m_layout;
    s.m_availableNs   = available;
    s.m_receivedNs    = available + 10000;
    s.m_snapshotValid = true;
    return s;
}
} // namespace

ConfirmedRecording record_accelsim( std::size_t frames, std::uint64_t seed, std::size_t restartTick )
{
    if( frames < 2 || frames > 100000 )
        throw std::invalid_argument( "Invalid offline recording size" );
    ConfirmedRecording r;
    r.m_inputs = ShadowReplayConfig{}.m_inputs;
    MagAOX::app::accelSimDetail::parameters parameters;
    parameters.m_fps  = 8000;
    parameters.m_seed = seed;
    MagAOX::app::accelSimDetail::model source;
    source.configure( parameters, MagAOX::app::accelSimDetail::referenceModes( parameters.m_modeSeed ) );
    // Fixed physical calibration, independent of subsequent observations/training. No future data normalization.
    r.m_arcsecPerUnit = .002 / std::abs( MagAOX::app::accelSimDetail::accelerationFactor( 28, parameters.m_leverArm ) );
    std::vector<std::array<double, 2>> pointing;
    for( std::size_t tick = 0; tick < frames * 4 + 6; ++tick )
    {
        const auto &s        = source.step(); // The exact simulator model, including all 214 modes and sensor noise.
        const auto  acquired = origin - 500000 + std::int64_t( tick ) * 125000;
        RecordedAdc a;
        a.m_frame      = { s.m_adc,
                           tick >= restartTick ? tick - restartTick + 1 : tick + 1,
                           epoch + acquired,
                           epoch + acquired + 1000 };
        a.m_receivedNs = acquired + 3000;
        a.m_restart    = tick == restartTick;
        r.m_adc.push_back( a );
        pointing.push_back( s.m_wfsPointing );
    }
    std::mt19937_64                  external( seed + 100 ), noise( seed + 101 );
    std::normal_distribution<double> externalNormal, wfsNormal;
    DDSPC::Matrix                    previous = DDSPC::Matrix::Zero( 2, 1 );
    for( std::size_t i = 0; i < frames; ++i )
    {
        const auto            start = origin + std::int64_t( i ) * 500000;
        DDSPC::TimedWfsRecord w;
        w.m_exposureStartNs = start;
        w.m_exposureEndNs   = start + 200000;
        w.m_decisionNs      = start + 250000;
        w.m_stamp           = stamp( r.m_inputs, r.m_inputs.m_wfsGeneration, 100 + i, start + 210000 );
        w.m_residual        = previous;
        for( int axis = 0; axis < 2; ++axis )
        {
            // Causal zero-order-held exposure average:125us of sample(start),75us of sample(start+125us).
            const double measured = .625 * pointing[4 + 4 * i][axis] + .375 * pointing[5 + 4 * i][axis];
            w.m_residual( axis ) += float( measured / r.m_arcsecPerUnit + .005 * wfsNormal( noise ) );
        }
        r.m_wfs.push_back( w );
        DDSPC::AppliedInputRecord u;
        u.m_requestedNs      = start + 50000;
        u.m_effectiveStartNs = start + 150000;
        u.m_effectiveEndNs   = start + 650000;
        u.m_stamp            = stamp( r.m_inputs, r.m_inputs.m_commandGeneration, 900 + i, u.m_effectiveEndNs + 10000 );
        u.m_evidence         = DDSPC::InputEvidence::SyntheticApplied;
        DDSPC::Matrix requested( 2, 1 );
        u.m_applied.resize( 2, 1 );
        for( int axis = 0; axis < 2; ++axis )
        {
            requested( axis )   = float( .6 * externalNormal( external ) );
            u.m_applied( axis ) = std::clamp( requested( axis ), -.4f, .4f );
        }
        r.m_requested.push_back( requested );
        r.m_holds.push_back( u );
        previous = u.m_applied; // The independently consumed limited input, never a DDSPC prediction.
    }
    return r;
}

ConfirmedReplayResult replay_confirmed( ReplaySink &sink, const ConfirmedRecording &r, const ConfirmedReplayOptions &o )
{
    if( r.m_wfs.size() < 2 || r.m_holds.size() != r.m_wfs.size() || o.m_deliveryDelayNs < 0 ||
        o.m_deliveryDelayNs > 1000000000 || o.m_adcOffsetCounts < -100 || o.m_adcOffsetCounts > 100 )
        throw std::invalid_argument( "Invalid confirmed replay recording or fault settings" );
    DDSPC::AdcConfig config;
    config.m_capacity = o.m_captureCapacity;
    DDSPC::AdcCapture capture( config );
    capture.attached( 7, 8 );
    MemoryAdc             source;
    AdcForwarder          forwarder( { r.m_inputs.m_clockEpoch + ( o.m_wrongClock ? 1 : 0 ), 1, 20 } );
    ConfirmedReplayResult result;
    std::size_t           a = 0, w = 0, u = 0;
    double                frameUs = 0;
    while( w < r.m_wfs.size() )
    {
        const auto at = a < r.m_adc.size() ? r.m_adc[a].m_receivedNs + o.m_deliveryDelayNs
                                           : std::numeric_limits<std::int64_t>::max();
        const auto ut =
            u < r.m_holds.size() ? r.m_holds[u].m_stamp.m_receivedNs : std::numeric_limits<std::int64_t>::max();
        const auto wt    = r.m_wfs[w].m_decisionNs;
        const auto now   = std::min( { at, ut, wt } );
        const auto begin = std::chrono::steady_clock::now();
        capture.tick( now );
        if( at == now )
        {
            const auto &event = r.m_adc[a];
            if( event.m_restart )
                capture.attached( 7, 9 );
            auto f = event.m_frame;
            if( o.m_adcOffsetCounts != 0 && a >= o.m_adcChangeStart )
                for( auto &code : f.m_raw )
                    code = std::uint16_t( std::clamp( int( code ) + o.m_adcOffsetCounts, 0, 65535 ) );
            source.m_header.m_counter                = f.m_counter;
            source.m_header.m_index                  = ( f.m_counter - 1 ) % 128;
            source.m_frames[source.m_header.m_index] = f;
            source.m_clock                           = { epoch + now, now, 10 };
            capture.drain( source );
            ++a;
        }
        else if( ut == now )
        {
            auto hold = r.m_holds[u];
            if( u >= o.m_badHoldStart && u - o.m_badHoldStart < o.m_badHoldCount )
                hold.m_evidence = DDSPC::InputEvidence::DriverPrepared;
            result.m_holdsAccepted += sink.applied( hold, now );
            ++u;
        }
        else
        {
            if( !( w >= o.m_skipForwardStart && w - o.m_skipForwardStart < o.m_skipForwardCount ) )
                forwarder.forward( capture, sink, now );
            if( !( w >= o.m_skipWfsStart && w - o.m_skipWfsStart < o.m_skipWfsCount ) )
                result.m_wfsAccepted += sink.wfs( r.m_wfs[w], now );
            ++w;
        }
        frameUs += std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - begin ).count();
        if( wt == now )
        {
            result.m_frameUs.push_back( frameUs );
            frameUs = 0;
        }
    }
    result.m_forwarding = forwarder.status();
    return result;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
