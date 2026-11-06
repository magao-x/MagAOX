/** \file async_test_data.hpp
 * \brief Deterministic synthetic WFS, held applied-input and timestamped 4/8-kHz sensor events for offline tests.
 */
#ifndef loPredCtrlAcc_async_test_data_hpp
#define loPredCtrlAcc_async_test_data_hpp
#include "async_shadow.hpp"
#include <random>
#include <stdexcept>

namespace libXWCTest::loPredCtrlAccTest
{
/// Synthetic sampling only: latest causal sample per WFS origin, without any live stream or calibration claim.
class AsyncTestInputs
{
    int           m_ratio;                        ///< Sensor events per 500-us WFS interval, two or four.
    std::uint64_t m_origin;                       ///< Monotonic synthetic exposure grid origin.
    std::mt19937  m_wfs{ 800 }, m_sensor{ 1080 }; ///< Independent, repeatable innovations.
    std::uint64_t m_sensorEvents{ 0 };            ///< Exact generated raw sample count.

  public:
    /// Select an offline sample rate and common time origin.
    AsyncTestInputs( int           sensorHz /**< [in] 4000 or 8000. */,
                     std::uint64_t originNs /**< [in] Exposure-grid origin. */ );

    /// Fill existing matrices and simulate individually timestamped causal sensor events.
    void next( std::uint64_t         frame /**< [in] Grid ordinal, independent of injected stream sequence gaps. */,
               DDSPC::OfflineRecord &record /**< [out] Preallocated declared synthetic data. */ );

    /// Count raw synthetic sensor events, before WFS-rate sample selection.
    std::uint64_t sensor_events() const;
};

inline AsyncTestInputs::AsyncTestInputs( int sensorHz, std::uint64_t originNs )
    : m_ratio( sensorHz / 2000 ), m_origin( originNs )
{
    if( sensorHz != 4000 && sensorHz != 8000 )
        throw std::invalid_argument( "Synthetic sensor rate must be 4 or 8 kHz" );
}
inline void AsyncTestInputs::next( std::uint64_t frame, DDSPC::OfflineRecord &record )
{
    record.m_streamEpoch     = 1;
    record.m_data.m_index    = frame;
    record.m_measurementNs   = m_origin + ( frame + 1 ) * 500000;
    record.m_availableNs     = record.m_measurementNs;
    record.m_exposureStartNs = m_origin + frame * 500000;
    record.m_appliedNs = record.m_appliedAvailableNs = record.m_measurementNs;
    record.m_data.m_wfsValid = record.m_data.m_appliedValid = record.m_data.m_timingValid = record.m_data.m_accelValid =
        true;
    for( auto *v : { &record.m_data.m_residual, &record.m_data.m_appliedCommand } )
        for( int j = 0; j < v->size(); ++j )
            ( *v )( j ) = ( int( m_wfs() % 2001 ) - 1000 ) / 1000.f;
    for( int sample = 0; sample < m_ratio; ++sample )
    {
        record.m_accelNs          = m_origin + frame * 500000 + sample * ( 500000 / m_ratio );
        record.m_accelAvailableNs = record.m_accelNs + 1000; // Known before the WFS origin for both synthetic rates.
        for( int j = 0; j < record.m_data.m_acceleration.size(); ++j )
            record.m_data.m_acceleration( j ) = ( int( m_sensor() % 2001 ) - 1000 ) / 1000.f;
        ++m_sensorEvents;
    }
}
inline std::uint64_t AsyncTestInputs::sensor_events() const
{
    return m_sensorEvents;
}
} // namespace libXWCTest::loPredCtrlAccTest
#endif
