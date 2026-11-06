/** \file comparison.cpp
 * \brief Matched deterministic replay and numerical feedback without hardware interfaces.
 */
#include "comparison.hpp"
#include "../ar_controller.hpp"
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace libXWCTest::loPredCtrlAccTest
{
namespace
{
using DDSPC::Matrix;
using Series        = std::vector<Matrix>;
constexpr int modes = 2, history = 2, accelHistory = 3, train = 1800, evaluate = 600;

/// Portable innovations plus multiple frequencies on an explicitly synthetic 2-kHz clock.
Series disturbance( unsigned seed /**< [in] Fixed sequence identity. */ )
{
    std::mt19937 rng( seed );
    Series       result( train + evaluate + 4, Matrix::Zero( modes, 1 ) );
    for( std::size_t t = 0; t < result.size(); ++t )
        for( int m = 0; m < modes; ++m )
            result[t]( m ) = ( int( rng() % 2001 ) - 1000 ) / 1000.f +
                             0.15f * std::sin( 2 * 3.141592653589793 * ( m ? 211 : 37 ) * t / 2000 ) +
                             0.1f * std::sin( 2 * 3.141592653589793 * 83 * t / 2000 );
    return result;
}

/// Independent zero-initialized newest-first indexed history oracle.
Matrix stack( const Series &values /**< [in] Trajectory. */,
              int           origin /**< [in] Latest included time. */,
              int           count /**< [in] Block count. */ )
{
    Matrix result = Matrix::Zero( modes * count, 1 );
    for( int j = 0; j < count; ++j )
        if( origin >= j )
            result.middleRows( modes * j, modes ) = values.at( origin - j );
    return result;
}

/// Canonical causal state built independently of production history getters.
Matrix state( const Series &y /**< [in] Residual trajectory. */,
              const Series &u /**< [in] Actual synthetic input trajectory. */,
              const Series &a /**< [in] Sensor trajectory. */,
              int           origin /**< [in] WFS/input origin. */,
              int           accelOrigin /**< [in] Sensor origin in its trajectory. */,
              bool          assisted /**< [in] Include sensor history. */ )
{
    Matrix result( ( 2 * history - 1 ) * modes + ( assisted ? modes * accelHistory : 0 ), 1 );
    result.topRows( history * modes )                             = stack( y, origin, history );
    result.middleRows( history * modes, ( history - 1 ) * modes ) = stack( u, origin - 1, history - 1 );
    if( assisted )
        result.bottomRows( modes * accelHistory ) = stack( a, accelOrigin, accelHistory );
    return result;
}

/// Stable coupled plant: absolute u_t held over [t,t+1) produces y_(t+1); units are synthetic.
Matrix plant( const Matrix &y /**< [in] Present residual. */,
              const Matrix &previousY /**< [in] Previous residual. */,
              const Matrix &u /**< [in] Actual input. */,
              const Matrix &previousU /**< [in] Previous input. */,
              const Matrix &a /**< [in] External disturbance, possibly unobserved. */,
              const Matrix &noise /**< [in] Independent process innovation. */,
              bool          coupled /**< [in] Enable disturbance coupling. */ )
{
    Matrix A( 2, 2 ), B( 2, 2 ), C( 2, 2 ), D( 2, 2 );
    A << 0.25f, 0.04f, -0.03f, 0.2f;
    B << 0.08f, -0.02f, 0.03f, 0.06f;
    C << 0.65f, 0.1f, -0.08f, 0.55f;
    D << 0.5f, -0.15f, 0.12f, 0.4f;
    return A * y - 0.04f * previousY + B * previousU + C * u + ( coupled ? 1.f : 0.f ) * D * a + 0.01f * noise;
}
} // namespace

ComparisonResult compare_models( int horizon, int scenario, bool qrd )
{
    const auto  excitation = disturbance( 201 ), latent = disturbance( 202 ), unrelated = disturbance( 203 ),
                noise  = disturbance( 204 );
    const auto &sensor = scenario == 2 ? unrelated : latent;
    Series      y( excitation.size(), Matrix::Zero( modes, 1 ) ), u = y;
    std::array<std::unique_ptr<DDSPC::PredictiveController>, 3> models;
    std::array<Matrix, 3>                                       theta, gains;
    for( int k = 0; k < 3; ++k )
    {
        models[k]          = std::make_unique<DDSPC::PredictiveController>( modes,
                                                                            history,
                                                                            horizon,
                                                                            0,
                                                                            1,
                                                                            0.3f,
                                                                            0.02f,
                                                                            k == 2 ? modes : 0,
                                                                            k == 2 ? accelHistory : 0,
                                                                            k == 0 ? DDSPC::HistoryMode::Legacy
                                                                                   : DDSPC::HistoryMode::Causal );
        models[k]->use_qrd = qrd;
    }
    for( int t = 0; t < train + evaluate; ++t )
    {
        // Actual synthetic input: clip and hold BEFORE recording. No inferred/calculated-command substitution.
        u[t] = t > 0 && t % 17 == 0 ? u[t - 1] : excitation[t].array().max( -0.7f ).min( 0.7f ).matrix().eval();
        if( t < train )
        {
            // Gain=0, no controller solve: legacy exploration records exactly the same applied input.
            const Matrix recorded = models[0]->calculate_command( y[t], u[t] );
            if( !( recorded.array() == u[t].array() ).all() )
                throw std::runtime_error( "Synthetic matched-input precondition failed" );
            models[1]->append_sample( t, y[t], u[t], Matrix() );
            models[2]->append_sample( t, y[t], u[t], sensor[t] );
            if( t >= horizon + accelHistory - 1 ) // Match mature update counts across all models.
                for( auto &model : models )
                    model->update_system();
        }
        y[t + 1] = plant( y[t],
                          t ? y[t - 1] : Matrix::Zero( modes, 1 ).eval(),
                          u[t],
                          t ? u[t - 1] : Matrix::Zero( modes, 1 ).eval(),
                          latent[t],
                          noise[t],
                          scenario != 1 );
    }
    for( int k = 0; k < 3; ++k )
    {
        theta[k] = models[k]->get_reference_prediction_matrix();
        if( k )
            gains[k] = models[k]->get_reference_controller( 0.3f ).bottomRows( modes );
    }
    ComparisonResult result;
    result.m_dNorm = theta[2].middleCols( ( 2 * history - 1 ) * modes, modes * accelHistory ).norm();
    for( int t = train; t < train + evaluate - horizon; ++t )
        for( int k = 0; k < 3; ++k )
        {
            const Matrix s = state( y, u, sensor, t, t, k == 2 );
            Matrix       phi( s.rows() + modes * horizon, 1 );
            // Future applied inputs CONDITION the offline score; they are never available decision features.
            phi << s, stack( u, t + horizon - 1, horizon );
            result.m_prediction[k] += ( theta[k] * phi - stack( y, t + horizon, horizon ) ).squaredNorm();
        }
    models[0]->update_controller(); // Retain existing scaled regularization and inverse, gain=0 in this study.
    models[0]->reset_buffers();
    for( int k = 0; k < 3; ++k )
    {
        Series cy( evaluate + 1, Matrix::Zero( modes, 1 ) ), cu = cy;
        for( int t = 0; t < evaluate; ++t )
        {
            const int i        = train + t;
            Matrix    proposal = k == 0 ? models[0]->calculate_command( cy[t], Matrix::Zero( modes, 1 ) )
                                        : Matrix( gains[k] * state( cy, cu, sensor, t, i, k == 2 ) );
            cu[t]              = t && t % 17 == 0 ? cu[t - 1] : proposal.array().max( -1.f ).min( 1.f ).matrix().eval();
            cy[t + 1]          = plant( cy[t],
                                        t ? cy[t - 1] : Matrix::Zero( modes, 1 ).eval(),
                                        cu[t],
                                        t ? cu[t - 1] : Matrix::Zero( modes, 1 ).eval(),
                                        latent[i],
                                        noise[i],
                                        scenario != 1 );
            result.m_residual[k] += cy[t + 1].squaredNorm();
            result.m_command[k] += cu[t].squaredNorm();
        }
        result.m_prediction[k] = std::sqrt( result.m_prediction[k] / ( ( evaluate - horizon ) * horizon * modes ) );
        result.m_residual[k]   = std::sqrt( result.m_residual[k] / ( evaluate * modes ) );
        result.m_command[k]    = std::sqrt( result.m_command[k] / ( evaluate * modes ) );
    }
    return result;
}
} // namespace libXWCTest::loPredCtrlAccTest
