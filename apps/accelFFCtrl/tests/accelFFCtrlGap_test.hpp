/** \file accelFFCtrlGap_test.hpp
 * \brief Offline-only bounded-gap experiment using unchanged production input validation and harmonic numerics.
 * \ingroup accelFFCtrl_unit_test
 */
#ifndef accelFFCtrlGap_test_hpp
#define accelFFCtrlGap_test_hpp

#include "../accelFFCtrlInput.hpp"

namespace libXWCTest
{
namespace accelFFCtrlTest
{
/// \cond ACCELFFCTRL_GAP_FIXTURES
namespace gapValidation
{
using namespace MagAOX::app::accelFFCtrlDetail;

/// Select the production opt-in for the same offline replay interface as the historical prototypes.
class productionObserver : public inputProcessor
{
  public:
    /// Prepare the actual production reader with only its single-frame option enabled.
    explicit productionObserver( const parameters &config /**< [in] common numerical configuration */ );

    /// Expose the production admission count to the shared offline scorer.
    uint64_t retained() const;
};

inline productionObserver::productionObserver( const parameters &config ) : inputProcessor( config, 256, 1000000, true )
{
}

inline uint64_t productionObserver::retained() const
{
    return diagnostics().m_admittedGaps;
}

/// Immutable owned depth-one snapshot; this experiment has no native stream API.
struct snapshot
{
    inputHeader m_header; ///< Already copied source metadata.
    inputFrame  m_frame;  ///< Already copied UINT16 pair and original acquisition/write times.

    /// Return the owned metadata.
    inputHeader header() const;

    /// Return the only owned plane.
    inputFrame frame( size_t index /**< [in] plane, necessarily zero */ ) const;
};

inline inputHeader snapshot::header() const
{
    return m_header;
}

inline inputFrame snapshot::frame( size_t index ) const
{
    return index == 0 ? m_frame : inputFrame{};
}

/// An experimental numerical owner; only this test header can select gap tolerance.
/** A production inputProcessor first validates every record and translates its timestamp.
 * Its private numerical query retrieves the mapped timestamp even during warmup;
 * only the independent estimator's 2-kHz queries are experimental predictions.
 * Accepted ordinal numbers, rather than physical publication counters, enter
 * that estimator. Real counters remain in diagnostics and in the admission policy.
 * No samples, ADC values or timestamps are synthesized to fill a gap.
 */
template <unsigned maxMissing>
class observer
{
    inputProcessor    m_gate; ///< Unmodified production clock/data/source validation, with its strict reset policy.
    harmonicEstimator m_estimator;  ///< Unmodified numerical kernel, with unchanged cold-start warmup.
    parameters        m_config;     ///< Same physical/window/freshness settings as the production comparison.
    prediction        m_prediction; ///< Most recent experimental query, cleared immediately on any reset.
    uint64_t m_serial{ 0 }, m_counter{ 0 }, m_generation{ 0 }; ///< Accepted ordinal and last physical source identity.
    uint64_t m_resets{ 0 }, m_rejected{ 0 }, m_accepted{ 0 },
        m_retained{ 0 };                   ///< Experimental fit and admission counters.
    int64_t m_time{ -1 }, m_lastGap{ -1 }; ///< Previous accepted monotonic acquisition and last retained gap.

    /// Clear only the experimental fit; the validator retains its physical counter/clock history.
    void clear();

  public:
    /// Prepare both numerical owners without accessing streams.
    explicit observer( const parameters &config /**< [in] physical/window/freshness settings */ );

    /// Bind a new source generation and require the original cold-start warmup.
    void attached();

    /// Validate one immutable depth-one snapshot, retaining history only under the bounded-gap policy.
    template <typename sourceT, typename clockT>
    size_t drain( sourceT &source /**< [in] offline owned source */,
                  clockT   clockNow /**< [in] real receipt-clock callback */ );

    /// Query at actual current time; the numerical kernel still enforces sample freshness and target bounds.
    void predict( int64_t now /**< [in] monotonic query ns */ );

    /// Return physical-source counters plus experimental readiness/reset counts.
    inputDiagnostics diagnostics() const;

    /// Count admitted counter gaps, separately from physical samples lost.
    uint64_t retained() const;
};

template <unsigned maxMissing>
inline observer<maxMissing>::observer( const parameters &config )
    : m_gate( config ), m_estimator( config ), m_config( config )
{
}

template <unsigned maxMissing>
inline void observer<maxMissing>::clear()
{
    m_estimator.reset();
    m_prediction = {};
    m_serial     = 0;
    m_time = m_lastGap = -1;
    ++m_resets;
}

template <unsigned maxMissing>
inline void observer<maxMissing>::attached()
{
    m_gate.attached();
    clear();
}

template <unsigned maxMissing>
template <typename sourceT, typename clockT>
inline size_t observer<maxMissing>::drain( sourceT &source, clockT clockNow )
{
    const auto header = source.header();
    if( header.m_write )
        return 0; // A busy producer is not evidence that an old fresh acquisition is corrupt.
    snapshot owned{ header, source.frame( 0 ) };
    if( !( header == source.header() ) || header.m_depth != 1 )
    {
        clear();
        return 0;
    }
    const auto   clock    = clockNow();
    const auto   before   = m_gate.diagnostics();
    const size_t result   = m_gate.drain( owned, [clock] { return clock; } );
    const auto   after    = m_gate.diagnostics();
    const bool   accepted = after.m_processed == before.m_processed + 1;
    if( !accepted )
    {
        if( after.m_resets != before.m_resets )
            clear();
        return result;
    }
    m_gate.predict( clock.m_mono );
    const auto mapped = m_gate.diagnostics().m_prediction;
    if( mapped.m_state != estimatorState::warming && mapped.m_state != estimatorState::ready )
    {
        clear();
        ++m_rejected;
        return result;
    }
    const uint64_t delta   = after.m_counter > m_counter ? after.m_counter - m_counter : 0;
    const double   nominal = 1e9 / m_config.m_sampleRate;
    const int64_t  elapsed = mapped.m_sampleTimeNs - m_time;
    const bool     bound   = m_time >= 0 && after.m_generation == m_generation && delta;
    // Preserve prior timing across a gap: the strict validator intentionally forgets it when resetting.
    if( bound && ( elapsed <= 0 || std::abs( elapsed - delta * nominal ) > .05 * nominal ) )
    {
        clear();
        ++m_rejected;
        return result;
    }
    const bool gap = bound && delta > 1;
    const bool retain =
        gap && delta - 1 <= maxMissing && elapsed <= 1000000 && m_estimator.state() == estimatorState::ready &&
        ( m_lastGap < 0 || mapped.m_sampleTimeNs - m_lastGap >= std::llround( m_config.m_memoryTime * 1e9 ) );
    const bool onlyOverrun = after.m_resets == before.m_resets + 1 && after.m_lastEvent == inputState::overrun;
    if( ( after.m_resets != before.m_resets && !( retain && onlyOverrun ) ) ||
        ( m_estimator.state() != estimatorState::warming && m_estimator.state() != estimatorState::ready ) )
        clear();
    if( retain && onlyOverrun )
    {
        m_lastGap = mapped.m_sampleTimeNs;
        ++m_retained;
    }
    m_counter    = after.m_counter;
    m_generation = after.m_generation;
    m_time       = mapped.m_sampleTimeNs;
    if( m_estimator.consume( { owned.m_frame.m_adc, m_time, ++m_serial, m_generation }, clock.m_mono ) )
        ++m_accepted;
    else
    {
        clear();
        ++m_rejected;
    }
    return result;
}

template <unsigned maxMissing>
inline void observer<maxMissing>::predict( int64_t now )
{
    m_prediction = m_estimator.predict( now );
}

template <unsigned maxMissing>
inline inputDiagnostics observer<maxMissing>::diagnostics() const
{
    auto result         = m_gate.diagnostics();
    result.m_prediction = m_prediction;
    result.m_resets     = m_resets;
    result.m_processed  = m_accepted;
    result.m_dropped += m_rejected;
    result.m_invalid += m_rejected;
    return result;
}

template <unsigned maxMissing>
inline uint64_t observer<maxMissing>::retained() const
{
    return m_retained;
}
} // namespace gapValidation
/// \endcond
} // namespace accelFFCtrlTest
} // namespace libXWCTest
#endif
