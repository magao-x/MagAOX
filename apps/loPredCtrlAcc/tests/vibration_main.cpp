/** \file vibration_main.cpp
 * \brief Export deterministic host-memory closed-loop experiments; no native application/stream linkage.
 */
#include "vibration.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace
{
using namespace libXWCTest::loPredCtrlAccTest;
/// Define positive, negative and timing controls before inspecting outcomes; identical seeds within every case.
std::vector<VibrationConfig> scenarios()
{
    std::vector<VibrationConfig> cases;
    VibrationConfig              c;
    c.m_name   = "28hz";
    c.m_peak56 = false;
    cases.push_back( c );
    c.m_name   = "56hz";
    c.m_peak28 = false;
    c.m_peak56 = true;
    cases.push_back( c );
    c = VibrationConfig{};
    cases.push_back( c );
    c.m_name      = "broadband";
    c.m_broadband = true;
    cases.push_back( c );
    auto add = [&]( const std::string &name, auto edit )
    {
        auto next   = c;
        next.m_name = name;
        edit( next );
        cases.push_back( next );
    };
    add( "accel_noise", []( auto &x ) { x.m_accelNoise = 6; } );
    for( int i = 0; i < 4; ++i )
        add( "coherence_" + std::to_string( i ), [i]( auto &x ) { x.m_coherence = i / 4.; } );
    add( "delay_jitter",
         []( auto &x )
         {
             x.m_wfsDelayFrames    = 1;
             x.m_sensorDelayFrames = 1;
             x.m_jitterNs          = 30000;
         } );
    add( "missing",
         []( auto &x )
         {
             x.m_missingFrame  = x.m_train + x.m_evaluate / 2;
             x.m_missingFrames = 20;
         } );
    for( int delay : { 1, 2, 3 } )
        add( "latency_" + std::to_string( delay ), [delay]( auto &x ) { x.m_dmExtraFrames = delay; } );
    for( double scale : { .7, 1.3, -1. } )
        add( scale < 0 ? "wrong_sign" : ( scale < 1 ? "gain_low" : "gain_high" ),
             [scale]( auto &x ) { x.m_dmScale = scale; } );
    add( "zero_gain", []( auto &x ) { x.m_controlGain = 0; } );
    add( "command_limit", []( auto &x ) { x.m_limit = .2; } );
    for( int delay : { 1, 3, 6 } )
        add( "publication_" + std::to_string( delay ), [delay]( auto &x ) { x.m_publicationFrames = delay; } );
    add( "reset", []( auto &x ) { x.m_resetFrame = x.m_train + x.m_evaluate / 2; } );
    add( "overflow",
         []( auto &x )
         {
             x.m_capacity    = 4;
             x.m_stallFrame  = x.m_train + x.m_evaluate / 2;
             x.m_stallFrames = 32;
         } );
    add( "broadband_qr", []( auto &x ) { x.m_qrd = true; } );
    for( unsigned seed : { 200, 300 } )
        add( "broadband_seed" + std::to_string( seed ), [seed]( auto &x ) { x.m_seed = seed; } );
    const auto original = cases;
    for( auto next : original )
        if( !next.m_qrd && next.m_name != "broadband" )
        {
            next.m_qrd = true;
            next.m_name += "_qr";
            cases.push_back( next );
        }
    return cases;
}
} //namespace

/// Write finite trace/configuration artifacts to a caller-selected offline directory only.
int main( int    argc /**< [in] Argument count. */,
          char **argv /**< [in] Output directory, optional case name or quick. */ )
{
    if( argc < 2 || argc > 3 )
        return 2;
    const std::filesystem::path root( argv[1] );
    std::filesystem::create_directories( root );
    std::ofstream config( root / "config.csv" );
    config << "scenario,seed,train,evaluate,history,future,accel_history,peak28,peak56,broadband,qrd,coherence,accel_"
              "noise,wfs_noise,lambda,prior,"
              "gamma,dm_pole,dm_scale,control_gain,limit,dm_extra_frames,wfs_delay_frames,sensor_delay_frames,"
              "publication_frames,jitter_ns,max_age_ns,capacity,missing_frame,missing_frames,reset_frame,stall_frame,"
              "stall_frames\n";
    bool ran = false;
    for( auto c : scenarios() )
    {
        if( argc == 3 && std::string( argv[2] ) != "quick" && c.m_name != argv[2] )
            continue;
        if( argc == 3 && std::string( argv[2] ) == "quick" )
        {
            c.m_train    = 600;
            c.m_evaluate = 1200;
            if( c.m_name != "mixed" && c.m_name != "broadband" )
                continue;
        }
        const auto s = vibration_source( c );
        const auto r = vibration_run( c, s );
        config << std::setprecision( 12 ) << c.m_name << ',' << c.m_seed << ',' << c.m_train << ',' << c.m_evaluate
               << ',' << c.m_history << ',' << c.m_future << ',' << c.m_accelHistory << ',' << c.m_peak28 << ','
               << c.m_peak56 << ',' << c.m_broadband << ',' << c.m_qrd << ',' << c.m_coherence << ',' << c.m_accelNoise
               << ',' << c.m_wfsNoise << ',' << c.m_lambda << ',' << c.m_prior << ',' << c.m_gamma << ',' << c.m_dmPole
               << ',' << c.m_dmScale << ',' << c.m_controlGain << ',' << c.m_limit << ',' << c.m_dmExtraFrames << ','
               << c.m_wfsDelayFrames << ',' << c.m_sensorDelayFrames << ',' << c.m_publicationFrames << ','
               << c.m_jitterNs << ',' << c.m_maxAgeNs << ',' << c.m_capacity << ',' << c.m_missingFrame << ','
               << c.m_missingFrames << ',' << c.m_resetFrame << ',' << c.m_stallFrame << ',' << c.m_stallFrames << '\n';
        std::ofstream file( root / ( c.m_name + ".csv" ) );
        file << std::setprecision( 10 )
             << "controller,frame,dx,dy,ax,ay,rx,ry,yx,yy,request_x,request_y,applied_x,"
                "applied_y,valid,ready,new_command,version,age_ns,model_sequence,origin,reason,prediction_squared,"
                "prediction_count,overflows,invalid_changes,theta_norm,d_norm,rank,sensor_index,sensor_time_ns,"
                "sensor_available_ns,wfs_available_ns\n";
        for( int model = 0; model < 5; ++model )
            for( std::size_t k = 0; k < r.m_frames[model].size(); ++k )
            {
                const auto &f = r.m_frames[model][k];
                const int   a = s.m_sensorIndex[k];
                file << model << ',' << k;
                for( int j = 0; j < 2; ++j )
                    file << ',' << s.m_disturbance[k]( j );
                for( int j = 0; j < 2; ++j )
                    file << ',' << ( a >= 0 ? s.m_acceleration[a]( j ) : std::numeric_limits<float>::quiet_NaN() );
                for( auto *v : { &f.m_residual, &f.m_measured, &f.m_requested, &f.m_applied } )
                    for( int j = 0; j < 2; ++j )
                        file << ',' << ( *v )( j );
                file << ',' << f.m_valid << ',' << f.m_ready << ',' << f.m_newCommand << ',' << f.m_version << ','
                     << f.m_ageNs << ',' << f.m_modelSequence << ',' << f.m_origin << ',' << f.m_reason << ','
                     << f.m_predictionSquared << ',' << f.m_predictionCount << ',' << r.m_overflows[model] << ','
                     << r.m_invalidChanges[model] << ',' << f.m_thetaNorm << ',' << f.m_dNorm << ',' << f.m_rank << ','
                     << a << ',' << ( a >= 0 ? s.m_sensorNs[a] : 0 ) << ',' << ( a >= 0 ? s.m_sensorAvailableNs[a] : 0 )
                     << ',' << s.m_availableNs[k] << '\n';
            }
        if( !file || !config )
            throw std::runtime_error( "Cannot write offline experiment artifacts" );
        std::cerr << "completed " << c.m_name << '\n';
        ran = true;
    }
    return ran ? 0 : 2;
}
