/** \file adc_forwarder.hpp
 * \brief Replay-only bounded forwarding of actual capture/reader batches into the application ADC seam.
 */
#ifndef loPredCtrlAcc_adc_forwarder_hpp
#define loPredCtrlAcc_adc_forwarder_hpp
#include "shadow_replay.hpp"
#include "../adc_reader.hpp"

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Forwarding evidence is distinct from capture loss: missing counters may overlap decoder drops/rejections.
struct ForwardingStatus
{
    DDSPC::AdcStatus m_source; ///< Coherent source identity/layout and original capture health/counters.
    DDSPC::AdcCursor m_cursor; ///< Last attempted record in this reader incarnation.
    std::uint64_t    m_copied{ 0 }, m_accepted{ 0 }, m_rejected{ 0 }, m_missing{ 0 },
        m_breaks{ 0 };       ///< Lifetime forwarding counts.
    bool m_pending{ false }; ///< Eligible retained records remain after the per-call work budget.
};

/// Replay-only explicit provenance; transport health alone never supplies these assertions.
struct ForwardingContract
{
    std::uint64_t m_clockEpoch{ 0 }, m_calibration{ 0 }; ///< Established synthetic clock and calibration identity.
    std::int64_t  m_uncertaintyNs{ 0 }; ///< Known synthetic acquisition/mapping bound; not inferred from receipt times.
};

/// Single consumer with bounded preallocated batches; no native stream, clock, thread, output or model ownership.
class AdcForwarder
{
    ForwardingContract            m_contract;   ///< Explicit replay provenance, independent of native reader health.
    std::vector<DDSPC::AdcSample> m_batch;      ///< Preallocated bounded caller-owned copy storage.
    std::size_t                   m_maxBatches; ///< Hard per-call copy/delivery budget.
    ForwardingStatus              m_status;     ///< Source/cursor/loss diagnostics, owned by the replay consumer.

    /// Apply one owned batch after the capture/reader lock has been released.
    void deliver( const DDSPC::AdcBatch &batch /**< [in] Coherent bounded snapshot. */,
                  ReplaySink            &sink /**< [in,out] Actual application ADC integration. */,
                  std::int64_t           nowNs /**< [in] Actual virtual frontend availability. */ );

  public:
    /// Validate explicit provenance and allocate once; opens no native resources.
    AdcForwarder( const ForwardingContract &contract /**< [in] Synthetic timing/calibration evidence. */,
                  std::size_t               batchSize  = 8 /**< [in] Maximum owned samples per copy. */,
                  std::size_t               maxBatches = 4 /**< [in] Maximum copies per dispatch. */ );

    /// Copy/deliver a bounded budget, preserving order and never backdating frontend availability.
    template <typename Source>
    void forward( const Source &source /**< [in] Serialized capture or thread-safe reader with copy_after(). */,
                  ReplaySink   &sink /**< [in,out] Input-only actual app seam. */,
                  std::int64_t  nowNs /**< [in] Current virtual dispatch time. */ );

    /// Copy scalar forwarding diagnostics without affecting capture retention.
    ForwardingStatus status() const;
};

template <typename Source>
void AdcForwarder::forward( const Source &source, ReplaySink &sink, std::int64_t nowNs )
{
    for( std::size_t i = 0; i < m_maxBatches; ++i )
    {
        const auto batch = source.copy_after( m_status.m_cursor, m_batch, nowNs );
        deliver( batch, sink, nowNs ); // copy_after released the reader lock before any model/application call.
        if( !batch.m_more )
            break;
    }
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
