/** \file adc_forwarder.cpp
 * \brief Bounded ADC replay forwarding with explicit loss invalidation and retained source provenance.
 */
#include "adc_forwarder.hpp"
#include <stdexcept>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
AdcForwarder::AdcForwarder( const ForwardingContract &contract, std::size_t batchSize, std::size_t maxBatches )
    : m_contract( contract ), m_maxBatches( maxBatches )
{
    if( !contract.m_clockEpoch || !contract.m_calibration || contract.m_uncertaintyNs < 0 || !batchSize ||
        batchSize > 65536 || !maxBatches || maxBatches > 65536 )
        throw std::invalid_argument( "Invalid offline ADC forwarding contract or work budget" );
    m_batch.resize( batchSize );
}

void AdcForwarder::deliver( const DDSPC::AdcBatch &batch, ReplaySink &sink, std::int64_t nowNs )
{
    m_status.m_source = batch.m_status;
    m_status.m_cursor = batch.m_next;
    m_status.m_missing += batch.m_missing;
    m_status.m_pending = batch.m_more;
    if( batch.m_discontinuity )
    {
        // Invalidate assistance even when a full new window would otherwise hide an intervening lost segment.
        DDSPC::ShadowAdcRecord invalid;
        sink.adc( invalid, nowNs );
        ++m_status.m_breaks;
    }
    for( std::size_t i = 0; i < batch.m_count; ++i )
    {
        DDSPC::ShadowAdcRecord record;
        record.m_sample        = m_batch[i]; // Preserve original clocks, counter, generation and segment unchanged.
        record.m_clockEpoch    = m_contract.m_clockEpoch;
        record.m_calibration   = m_contract.m_calibration;
        record.m_uncertaintyNs = m_contract.m_uncertaintyNs;
        record.m_snapshotValid =
            true; // The capture admitted this owned sample; mapping is the explicit replay contract.
        ++m_status.m_copied;
        if( sink.adc( record, nowNs ) )
            ++m_status.m_accepted;
        else
            ++m_status.m_rejected; // Attempted input is not retried forever; candidate eligibility remains fail-closed.
    }
}

ForwardingStatus AdcForwarder::status() const
{
    return m_status;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
