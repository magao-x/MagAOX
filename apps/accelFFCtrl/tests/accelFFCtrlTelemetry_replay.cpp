/** \file accelFFCtrlTelemetry_replay.cpp
 * \brief File-only ADC replay with unchanged estimation, optional earlier-fit phase calibration and shadow proposals.
 * \ingroup accelFFCtrl_files
 */
#include "../accelFFCtrlInput.hpp"
#include "../offline/accelFFCtrlShadow.hpp"
#include "accelFFCtrlPhase_test.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace MagAOX::app::accelFFCtrlDetail;
using namespace MagAOX::app::accelFFCtrlOffline;
using libXWCTest::accelFFCtrlTest::calibrated;
using libXWCTest::accelFFCtrlTest::phaseCalibration;

/// \cond ACCELFFCTRL_TELEMETRY_FIXTURE
namespace
{
/// Owned depth-one file snapshot, with no native transport dependency.
struct fileSource
{
    inputFrame m_frame; ///< One copied acquisition and its original publication identity.

    /// Return the local completed depth-one header.
    inputHeader header() const;

    /// Return the only local plane.
    inputFrame frame( size_t index /**< [in] plane zero */ ) const;
};

inputHeader fileSource::header() const
{
    return { 3, 2, 1, 1, m_frame.m_counter, 0, true, true, true, false };
}

inputFrame fileSource::frame( size_t ) const
{
    return m_frame;
}

/// Exact numeric file row, already translated by the caller to a declared common clock.
struct row
{
    int64_t  m_acquired, m_written, m_receipt; ///< Common-domain ns; never read from the machine clock.
    uint64_t m_counter, m_generation;          ///< Source identities preserved through missing frames.
    unsigned m_x, m_y;                         ///< Raw ADC codes, validated before UINT16 conversion.
};

/// Read one strict seven-field numeric CSV row without floating-point timestamp conversion.
bool readRow( std::istream &input /**< [in,out] CSV after header */, row &r /**< [out] exact owned row */ )
{
    std::string line;
    if( !std::getline( input, line ) )
        return false;
    std::replace( line.begin(), line.end(), ',', ' ' );
    std::istringstream fields( line );
    if( !( fields >> r.m_acquired >> r.m_written >> r.m_receipt >> r.m_counter >> r.m_generation >> r.m_x >> r.m_y ) ||
        r.m_x > 65535 || r.m_y > 65535 || !r.m_generation )
        throw std::invalid_argument( "invalid normalized ADC row" );
    std::string extra;
    if( fields >> extra )
        throw std::invalid_argument( "extra ADC field" );
    return true;
}

/// Convert a normalized nonnegative nanosecond timestamp to the production metadata representation.
timespec stamp( int64_t ns /**< [in] common-domain ns */ )
{
    return { time_t( ns / 1000000000 ), long( ns % 1000000000 ) };
}

} // namespace
/// \endcond

/// Replay only files: parameters, ADC rows, query times, and the output diagnostic CSV.
int main( int    argc /**< [in] argument count */,
          char **argv /**< [in] params.txt adc.csv queries.csv proposals.csv [phase-calibration.txt] */ )
try
{
    if( argc != 5 && argc != 6 )
        throw std::invalid_argument(
            "usage: accelFFCtrlTelemetry_replay params.txt adc.csv queries.csv proposals.csv [phase-calibration.txt]" );
    std::ifstream        settings( argv[1] ), data( argv[2] ), queries( argv[3] );
    std::ofstream        output( argv[4] );
    parameters           config;
    correctionParameters controls;
    Eigen::Matrix2d      coordinates, mapping;
    unsigned             evidence      = 0;
    uint64_t             calibrationId = 0;
    size_t               bands         = 0;
    settings >> config.m_sampleRate >> config.m_memoryTime >> config.m_leverArm >> config.m_rotationDegrees;
    for( auto *v : { &config.m_zero, &config.m_voltsPerBit, &config.m_sensitivity } )
        settings >> ( *v )[0] >> ( *v )[1];
    settings >> config.m_predictionNs >> controls.m_gain >> controls.m_maxModal[0] >> controls.m_maxModal[1];
    for( auto *matrix : { &coordinates, &mapping } )
        for( size_t row = 0; row < 2; ++row )
            for( size_t col = 0; col < 2; ++col )
                settings >> ( *matrix )( row, col );
    settings >> calibrationId >> evidence >> bands;
    if( !settings || evidence > 1 || bands == 0 || bands > 2 || !data || !queries || !output )
        throw std::invalid_argument( "invalid replay files/settings" );
    controls.m_horizonNs = config.m_predictionNs;
    shadowCommand shadow( coordinates, mapping, calibrationId, shadowCalibration( evidence ), controls );
    std::vector<std::unique_ptr<inputProcessor>> inputs;
    std::vector<double>                          frequencies( bands );
    for( auto &frequency : frequencies )
    {
        if( !( settings >> frequency ) )
            throw std::invalid_argument( "missing replay frequency" );
        config.m_frequency = frequency;
        inputs.emplace_back( std::make_unique<inputProcessor>( config, 256, 1000000, false ) );
    }
    std::vector<phaseCalibration> calibration;
    if( argc == 6 )
    {
        // This optional file mode is expressly count-normalized, never an SI or physical DM calibration.
        if( bands != 2 || config.m_zero != std::array<double, 2>{ 0, 0 } ||
            config.m_voltsPerBit != std::array<double, 2>{ 1, 1 } ||
            config.m_sensitivity != std::array<double, 2>{ 1, 1 } || config.m_leverArm != 1 ||
            config.m_rotationDegrees != 0 )
            throw std::invalid_argument( "phase calibration requires the declared count-normalized basis" );
        std::ifstream phaseFile( argv[5] );
        std::string   version;
        int64_t       horizon;
        if( !( phaseFile >> version >> horizon ) || version != "accelFFCtrl-count-phase-v1" ||
            horizon != config.m_predictionNs )
            throw std::invalid_argument( "invalid phase calibration basis/horizon" );
        calibration.resize( bands );
        for( size_t band = 0; band < bands; ++band )
        {
            double frequency;
            phaseFile >> frequency >> calibration[band].m_axis;
            for( size_t row = 0; row < 2; ++row )
                for( size_t col = 0; col < 2; ++col )
                    phaseFile >> calibration[band].m_weights( row, col );
            if( !phaseFile || frequency != frequencies[band] || calibration[band].m_axis != band ||
                !calibration[band].m_weights.allFinite() )
                throw std::invalid_argument( "invalid phase calibration frequency/axis/weights" );
        }
        std::string extra;
        if( phaseFile >> extra )
            throw std::invalid_argument( "extra phase calibration field" );
    }
    std::unique_ptr<dmResponse> plant;
    int64_t                     plantPeriod = 0, delay = 0;
    double                      bandwidth      = 0;
    unsigned                    syntheticPlant = 0;
    if( !( settings >> syntheticPlant ) || syntheticPlant > 1 )
        throw std::invalid_argument( "missing explicit synthetic-response choice" );
    if( syntheticPlant )
    {
        Eigen::Matrix2d response;
        settings >> plantPeriod >> delay >> bandwidth;
        for( size_t row = 0; row < 2; ++row )
            for( size_t col = 0; col < 2; ++col )
                settings >> response( row, col );
        if( !settings )
            throw std::invalid_argument( "invalid synthetic-response controls" );
        plant = std::make_unique<dmResponse>( response, plantPeriod, delay, bandwidth );
    }
    std::string header;
    std::getline( data, header );
    if( header != "acquired_ns,written_ns,receipt_ns,counter,generation,adc_x,adc_y" )
        throw std::invalid_argument( "unexpected normalized ADC header" );
    std::getline( queries, header );
    if( header != "query_ns" )
        throw std::invalid_argument( "unexpected query header" );
    output
        << std::setprecision( 17 )
        << "query_ns,sample_ns,target_ns,counter,generation,epoch,valid,state,fault,scale,theta_x,theta_y,mode0,mode1,"
           "processed,dropped,resets,synthetic_effect_x,synthetic_effect_y";
    for( size_t band = 0; band < bands; ++band )
        for( const auto *field : { "px", "py", "ax_re", "ax_im", "ay_re", "ay_im" } )
            output << ',' << field << band;
    output << '\n';
    row                     r{};
    bool                    have       = readRow( data, r );
    uint64_t                generation = 0, resets = 0;
    int64_t                 lastQuery = -1, lastReceipt = -1;
    int64_t                 now;
    std::vector<prediction> predictions( bands );
    std::vector<prediction> mappedPredictions( bands );
    while( queries >> now )
    {
        if( now < 0 || now <= lastQuery || ( plant && lastQuery >= 0 && now - lastQuery != plantPeriod ) )
            throw std::invalid_argument( "query times must strictly increase" );
        lastQuery = now;
        while( have && r.m_receipt <= now )
        {
            if( r.m_receipt < lastReceipt )
                throw std::invalid_argument( "receipt times must not decrease" );
            lastReceipt = r.m_receipt;
            if( generation != r.m_generation )
            {
                for( auto &input : inputs )
                    input->attached();
                generation = r.m_generation;
            }
            fileSource source{ { { uint16_t( r.m_x ), uint16_t( r.m_y ) },
                                 stamp( r.m_acquired ),
                                 stamp( r.m_written ),
                                 r.m_counter } };
            for( auto &input : inputs )
                input->drain( source, [&r] { return inputClock{ r.m_receipt, r.m_receipt, 0 }; } );
            have = readRow( data, r );
        }
        const auto &stats = inputs.front()->diagnostics();
        if( resets != stats.m_resets )
        {
            shadow.reset( now, 0 );
            resets = stats.m_resets;
        }
        for( size_t band = 0; band < bands; ++band )
        {
            inputs[band]->predict( now );
            predictions[band] = inputs[band]->diagnostics().m_prediction;
        }
        if( !calibration.empty() )
            for( size_t band = 0; band < bands; ++band )
                mappedPredictions[band] = calibrated( predictions[band], frequencies[band], calibration[band] );
        const auto record = shadow.propose( calibration.empty() ? predictions : mappedPredictions, now );
        const auto effect = plant ? plant->step( record.m_proposal.m_modal ) : std::array<double, 2>{};
        output << now << ',' << record.m_sampleNs << ',' << record.m_requestedTargetNs << ',' << record.m_sourceCounter
               << ',' << record.m_sourceGeneration << ',' << record.m_epoch << ',' << record.m_validPrediction << ','
               << int( record.m_proposal.m_state ) << ',' << int( record.m_fault ) << ',' << record.m_proposal.m_scale
               << ',' << record.m_pointing[0] << ',' << record.m_pointing[1] << ',' << record.m_proposal.m_modal[0]
               << ',' << record.m_proposal.m_modal[1] << ',' << stats.m_processed << ',' << stats.m_dropped << ','
               << stats.m_resets << ',' << effect[0] << ',' << effect[1];
        for( size_t band = 0; band < bands; ++band )
        {
            const auto &p = predictions[band];
            output << ',' << p.m_pointing[0] << ',' << p.m_pointing[1];
            for( size_t axis = 0; axis < 2; ++axis )
            {
                const double phase = p.m_phase[axis] + 6.28318530717958647692 * frequencies[band] *
                                                           ( ( p.m_targetTimeNs - p.m_sampleTimeNs ) * 1e-9 );
                output << ',' << p.m_amplitude[axis] * std::cos( phase ) << ','
                       << p.m_amplitude[axis] * std::sin( phase );
            }
        }
        output << '\n';
    }
    if( !queries.eof() || lastQuery < 0 )
        throw std::invalid_argument( "invalid/empty query file" );
    output.close();
    if( !output.good() )
        throw std::runtime_error( "failed telemetry replay export" );
    return 0;
}
catch( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
