/** \file causal_input_test_data.hpp
 * \brief Explicitly synthetic exposures, clipped held inputs, and clock metadata for offline pairing tests.
 */
#ifndef loPredCtrlAcc_causal_input_test_data_hpp
#define loPredCtrlAcc_causal_input_test_data_hpp

#include "../causal_input.hpp"
#include <random>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{

/// Establish a synthetic 2-kHz cadence with 50-us response lag and a 100-us minimum delivery delay.
inline DDSPC::InputConfig input_config()
{
    DDSPC::InputConfig config;
    config.m_modes             = 2;
    config.m_accelChannels     = 1;
    config.m_wfsGeneration     = 11;
    config.m_commandGeneration = 22;
    config.m_clockEpoch        = 33;
    config.m_layout            = 44;
    config.m_periodNs          = 500000;
    config.m_exposureNs        = 200000;
    config.m_maxAgeNs          = 5000000;
    config.m_responseLagNs     = 50000;
    config.m_minCommandDelayNs = 100000;
    config.m_evidence          = DDSPC::InputEvidence::SyntheticApplied;
    config.m_timingVerified = config.m_mappingVerified = true;
    return config;
}

/// Generate deterministic independent signals without simulator internals or controller proposals.
inline DDSPC::Matrix input_signal( int           count /**< [in] Vector length. */,
                                   std::uint64_t seed /**< [in] Deterministic signal identity. */ )
{
    std::mt19937  random( seed );
    DDSPC::Matrix value( count, 1 );
    for( int mode = 0; mode < count; ++mode )
        value( mode ) = float( int( random() % 2001 ) - 1000 ) / 1000;
    return value;
}

/// Make a synthetic immutable y_i exposure with features available before its frozen origin deadline.
inline DDSPC::TimedWfsRecord input_wfs( const DDSPC::InputConfig &config /**< [in] Synthetic session. */,
                                        std::uint64_t             index /**< [in] Physical sample number. */ )
{
    DDSPC::TimedWfsRecord record;
    record.m_stamp.m_generation    = config.m_wfsGeneration;
    record.m_stamp.m_counter       = 100 + index;
    record.m_stamp.m_clockEpoch    = config.m_clockEpoch;
    record.m_stamp.m_layout        = config.m_layout;
    record.m_stamp.m_snapshotValid = true;
    record.m_exposureStartNs       = 1000000 + index * config.m_periodNs;
    record.m_exposureEndNs         = record.m_exposureStartNs + config.m_exposureNs;
    record.m_stamp.m_availableNs   = record.m_exposureEndNs + 10000;
    record.m_stamp.m_receivedNs    = record.m_exposureEndNs + 20000;
    record.m_decisionNs            = record.m_exposureEndNs + 50000;
    record.m_residual              = input_signal( config.m_modes, 9917 + index );
    record.m_acceleration          = input_signal( config.m_accelChannels, 1231 + index );
    record.m_accelAcquiredNs       = record.m_decisionNs - 20000;
    record.m_accelAvailableNs      = record.m_decisionNs - 10000;
    record.m_accelValid            = true;
    return record;
}

/// Certify the clipped input u_i actually held during the response-aligned next exposure, independent of requests.
inline DDSPC::AppliedInputRecord input_command( const DDSPC::InputConfig &config /**< [in] Synthetic session. */,
                                                std::uint64_t index /**< [in] Input affecting exposure index+1. */ )
{
    DDSPC::AppliedInputRecord record;
    record.m_stamp.m_generation    = config.m_commandGeneration;
    record.m_stamp.m_counter       = 900 + index;
    record.m_stamp.m_clockEpoch    = config.m_clockEpoch;
    record.m_stamp.m_layout        = config.m_layout;
    record.m_stamp.m_snapshotValid = true;
    record.m_effectiveStartNs      = 1000000 + ( index + 1 ) * config.m_periodNs - config.m_responseLagNs;
    record.m_effectiveEndNs        = record.m_effectiveStartNs + config.m_periodNs;
    record.m_requestedNs           = record.m_effectiveStartNs - 150000;
    record.m_stamp.m_availableNs   = record.m_effectiveEndNs + 10000;
    record.m_stamp.m_receivedNs    = record.m_effectiveEndNs + 20000;
    record.m_applied               = input_signal( config.m_modes, 7919 + index ).cwiseMax( -0.4f ).cwiseMin( 0.4f );
    record.m_evidence              = DDSPC::InputEvidence::SyntheticApplied;
    return record;
}

} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
