/** \file accelFFCtrlResponse_replay.cpp
 * \brief File-only normalized forecast-to-shadow-to-DM response evaluation, never a feedback controller.
 * \ingroup accelFFCtrl_files
 */
#include "../offline/accelFFCtrlShadow.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace MagAOX::app::accelFFCtrlDetail;
using namespace MagAOX::app::accelFFCtrlOffline;

/// Apply existing shadow bounds and the existing synthetic DM model to qualified forecast snapshots in files.
/** The CSV validity flag comes from the earlier predictor replay, not a new estimator.
 * For comparator forecasts the snapshot is a declared numerical fixture. rcond=1 is only the
 * fixture marker; no fit condition number, physical acceleration or optical calibration is inferred.
 * Irregular row times are retained: ideal response is an exact sample-history delay, matching
 * aoSim lag+1. A finite-bandwidth response uses an explicitly nominal 0.5-ms integration step.
 */
int main( int argc /**< [in] argument count */, char **argv /**< [in] parameters.txt forecasts.csv response.csv */ )
try
{
    if( argc != 4 )
        throw std::invalid_argument( "usage: accelFFCtrlResponse_replay parameters.txt forecasts.csv response.csv" );
    std::ifstream         settings( argv[1] ), input( argv[2] );
    std::ofstream         output( argv[3] );
    correctionParameters  controls;
    std::array<double, 2> sigma;
    unsigned              steps;
    double                bandwidth, sign;
    settings >> controls.m_gain >> sigma[0] >> sigma[1] >> controls.m_maxModal[0] >> controls.m_maxModal[1] >>
        controls.m_horizonNs >> steps >> bandwidth >> sign;
    if( !settings || !input || !output || steps > 20 || ( sign != 1 && sign != -1 ) || !std::isfinite( sigma[0] ) ||
        !std::isfinite( sigma[1] ) || sigma[0] <= 0 || sigma[1] <= 0 )
        throw std::invalid_argument( "invalid normalized response settings" );
    Eigen::Matrix2d normalization = Eigen::Matrix2d::Zero(), response = normalization;
    for( size_t axis = 0; axis < 2; ++axis )
    {
        normalization( axis, axis ) = 1 / sigma[axis];
        response( axis, axis )      = sign * sigma[axis];
    }
    shadowCommand shadow( Eigen::Matrix2d::Identity(), normalization, 15, shadowCalibration::synthetic, controls );
    dmResponse    plant( response, 500000, int64_t( steps ) * 500000, bandwidth );
    std::string   line;
    std::getline( input, line );
    if( line != "query_ns,sample_ns,target_ns,counter,generation,valid,tip,tilt,reference_tip,reference_tilt" )
        throw std::invalid_argument( "unexpected forecast header" );
    output
        << std::setprecision( 17 )
        << "query_ns,target_ns,valid,state,fault,scale,mode0,mode1,effect_tip,effect_tilt,residual_tip,residual_tilt\n";
    int64_t previous = -1;
    while( std::getline( input, line ) )
    {
        std::replace( line.begin(), line.end(), ',', ' ' );
        std::istringstream    fields( line );
        prediction            p;
        int64_t               now;
        unsigned              valid;
        std::array<double, 2> reference;
        fields >> now >> p.m_sampleTimeNs >> p.m_targetTimeNs >> p.m_sequence >> p.m_generation >> valid >>
            p.m_pointing[0] >> p.m_pointing[1] >> reference[0] >> reference[1];
        std::string extra;
        if( !fields || fields >> extra || valid > 1 || now <= previous || !std::isfinite( reference[0] ) ||
            !std::isfinite( reference[1] ) )
            throw std::invalid_argument( "invalid forecast row or order" );
        previous  = now;
        p.m_state = valid ? estimatorState::ready : estimatorState::warming;
        p.m_rcond = 1; // Qualified forecast snapshot marker only; never claimed as a recorded fit condition number.
        const auto record = shadow.propose( { &p, 1 }, now );
        const auto effect = plant.step( record.m_proposal.m_modal );
        output << now << ',' << record.m_requestedTargetNs << ',' << record.m_validPrediction << ','
               << int( record.m_proposal.m_state ) << ',' << int( record.m_fault ) << ',' << record.m_proposal.m_scale
               << ',' << record.m_proposal.m_modal[0] << ',' << record.m_proposal.m_modal[1] << ',' << effect[0] << ','
               << effect[1] << ',' << reference[0] + effect[0] << ',' << reference[1] + effect[1] << '\n';
    }
    if( previous < 0 || !input.eof() )
        throw std::invalid_argument( "empty or invalid forecast file" );
    output.close();
    if( !output.good() )
        throw std::runtime_error( "failed response output" );
    return 0;
}
catch( const std::exception &error )
{
    std::cerr << error.what() << '\n';
    return 1;
}
