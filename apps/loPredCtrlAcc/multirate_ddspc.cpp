/** \file multirate_ddspc.cpp
 * \brief Shared-factor phase-conditioned DDSPC learner and per-slot control law.
 */

#include "multirate_ddspc.hpp"

#include <cmath>
#include <stdexcept>

namespace DDSPC
{

int MultirateConfig::lift() const
{
    return m_slots * m_modes;
}

int MultirateConfig::futureInputs() const
{
    return m_future * lift();
}

int MultirateConfig::window() const
{
    return m_ratio * m_accelHistory;
}

int MultirateConfig::state( int phase ) const
{
    return m_history * m_modes + ( m_history - 1 ) * lift() + 2 * ( window() + phase );
}

int MultirateConfig::features( int phase ) const
{
    return futureInputs() + state( phase );
}

int MultirateConfig::outputs() const
{
    return m_future * m_modes;
}

void MultirateConfig::validate() const
{
    if( m_modes < 1 || m_modes > 16 || m_history < 1 || m_history > 64 || m_future < 1 || m_future > 16 ||
        m_slots < 1 || m_slots > 64 || m_ratio < 1 || m_ratio > 64 || m_accelHistory < 1 || m_accelHistory > 64 ||
        !( m_gamma > 0 && m_gamma <= 1 ) || !( m_prior > 0 ) || !std::isfinite( m_prior ) || !( m_lambda > 0 ) ||
        !std::isfinite( m_lambda ) || features( m_ratio - 1 ) > 4096 )
        throw std::invalid_argument( "Invalid multirate DDSPC dimensions or parameters" );
}

InformationQR::InformationQR( int features, int outputs, double gamma, double prior )
    : m_features( features ), m_outputs( outputs ), m_rootGamma( std::sqrt( gamma ) ), m_rootPrior( std::sqrt( prior ) )
{
    if( features < 1 || outputs < 1 || !( gamma > 0 && gamma <= 1 ) || !( prior > 0 ) )
        throw std::invalid_argument( "Invalid information factor" );
    m_factor.resize( features, features + outputs );
    m_row.resize( features + outputs );
    reset();
}

void InformationQR::reset()
{
    m_factor.setZero();
    m_factor.leftCols( m_features ).diagonal().setConstant( m_rootPrior );
}

void InformationQR::update( const VectorD &regressor, const VectorD &target )
{
    if( regressor.size() != m_features || target.size() != m_outputs || !regressor.allFinite() || !target.allFinite() )
        throw std::invalid_argument( "Invalid information-factor example" );
    const int width = m_features + m_outputs;
    m_factor *= m_rootGamma;
    m_row.head( m_features ) = regressor.transpose();
    m_row.tail( m_outputs )  = target.transpose();
    double *row              = m_row.data();
    for( int j = 0; j < m_features; ++j )
    {
        const double b = row[j];
        if( b == 0 )
            continue;
        double      *rj = m_factor.data() + std::size_t( j ) * width;
        const double a  = rj[j];
        const double r  = std::sqrt( a * a + b * b );
        const double c = a / r, s = b / r;
        rj[j]  = r;
        row[j] = 0;
        for( int l = j + 1; l < width; ++l )
        {
            const double x = rj[l], y = row[l];
            rj[l]  = c * x + s * y;
            row[l] = c * y - s * x;
        }
    }
}

void InformationQR::solve( int leading, MatrixD &coefficients ) const
{
    if( leading < 1 || leading > m_features || coefficients.rows() != leading || coefficients.cols() != m_outputs )
        throw std::invalid_argument( "Invalid information-factor solve" );
    coefficients = m_factor.block( 0, m_features, leading, m_outputs );
    m_factor.topLeftCorner( leading, leading ).triangularView<Eigen::Upper>().solveInPlace( coefficients );
}

int InformationQR::features() const
{
    return m_features;
}

MultirateDDSPC::MultirateDDSPC( const MultirateConfig &config )
    : m_config( ( config.validate(), config ) ),
      m_qr( config.features( config.m_ratio - 1 ), config.outputs(), config.m_gamma, config.m_prior ),
      m_lambda( config.m_lambda )
{
    const auto &c = m_config;
    m_frames.resize( c.m_history + c.m_future + 2 );
    for( auto &frame : m_frames )
    {
        frame.m_y    = VectorD::Zero( c.m_modes );
        frame.m_lift = VectorD::Zero( c.lift() );
    }
    std::size_t size = 1;
    while( size < std::size_t( 8 * ( c.window() + c.m_ratio * ( c.m_future + 4 ) ) ) )
        size <<= 1;
    m_samples.resize( size );
    m_phi    = VectorD::Zero( c.features( c.m_ratio - 1 ) );
    m_target = VectorD::Zero( c.outputs() );
    m_error  = VectorD::Zero( c.outputs() );
    m_ctc    = MatrixD::Zero( c.futureInputs(), c.futureInputs() );
    m_buffer.assign( 2 * ( c.window() + c.m_ratio - 1 ), 0 );
    m_theta.resize( c.m_ratio );
    for( int k = 0; k < c.m_ratio; ++k )
        m_theta[k] = MatrixD::Zero( c.features( k ), c.outputs() );
    m_status.m_rmse.assign( c.m_ratio, 0 );
    m_status.m_scored.assign( c.m_ratio, 0 );
}

const MultirateConfig &MultirateDDSPC::config() const
{
    return m_config;
}

void MultirateDDSPC::reset()
{
    m_qr.reset();
    break_history();
    const bool training = m_training;
    m_status            = {};
    m_status.m_rmse.assign( m_config.m_ratio, 0 );
    m_status.m_scored.assign( m_config.m_ratio, 0 );
    m_status.m_training = training;
    m_extracted         = false;
    m_dirty             = true;
}

void MultirateDDSPC::break_history()
{
    m_status.m_breaks += m_count > 0;
    m_count   = 0;
    m_example = false;
}

void MultirateDDSPC::training( bool enabled )
{
    m_training = m_status.m_training = enabled;
}

void MultirateDDSPC::regularization( double lambda )
{
    if( !( lambda > 0 ) || !std::isfinite( lambda ) )
        throw std::invalid_argument( "Invalid multirate DDSPC penalty" );
    if( lambda != m_lambda )
        m_dirty = true;
    m_lambda = lambda;
}

void MultirateDDSPC::sample( std::uint64_t counter, std::uint64_t segment, const double *value )
{
    auto &entry      = m_samples[counter & ( m_samples.size() - 1 )];
    entry.m_counter  = counter;
    entry.m_segment  = segment;
    entry.m_value[0] = value[0];
    entry.m_value[1] = value[1];
}

bool MultirateDDSPC::samples( std::uint64_t first, int count, std::uint64_t segment, double *out ) const
{
    for( int n = 0; n < count; ++n )
    {
        const auto &entry = m_samples[( first + n ) & ( m_samples.size() - 1 )];
        if( entry.m_counter != first + n || entry.m_segment != segment )
            return false;
        out[2 * n]     = entry.m_value[0];
        out[2 * n + 1] = entry.m_value[1];
    }
    return true;
}

bool MultirateDDSPC::append( std::uint64_t    index,
                             const VectorD   &y,
                             const MatrixD   &applied,
                             const VectorD   &window,
                             std::uint64_t    anchor,
                             std::uint64_t    segment,
                             MultirateOrigin &origin )
{
    const auto &c = m_config;
    const int   m = c.m_modes, q = c.m_slots, N = c.m_history, F = c.m_future, L = c.window(), lift = c.lift();
    const auto  ring  = m_frames.size();
    auto        frame = [&]( std::uint64_t i ) -> Frame        &{ return m_frames[i % ring]; };
    origin.m_valid    = false;
    if( y.size() != m || !y.allFinite() )
    {
        break_history();
        return false;
    }
    if( m_count && index != m_last + 1 )
        break_history();
    if( m_count )
    {
        if( applied.rows() == m && applied.cols() == q && applied.allFinite() )
            for( int j = 0; j < q; ++j ) // Newest slot first.
                frame( m_last ).m_lift.segment( ( q - 1 - j ) * m, m ) = applied.col( j );
        else
            break_history();
    }
    auto &current     = frame( index );
    current.m_y       = y;
    current.m_anchor  = anchor;
    current.m_segment = segment;
    current.m_accel   = window.size() == 2 * L && window.allFinite() && anchor + 1 >= std::uint64_t( L );
    if( current.m_accel )
        for( int j = 0; j < L; ++j )
            sample( anchor - j, segment, window.data() + 2 * j );
    m_last = index;
    ++m_count;

    origin.m_index   = index;
    origin.m_anchor  = anchor;
    origin.m_segment = segment;
    origin.m_valid   = current.m_accel && m_count >= std::uint64_t( N );
    if( origin.m_valid )
    {
        for( int h = 0; h < N; ++h )
            origin.m_state.segment( h * m, m ) = frame( index - h ).m_y;
        for( int h = 1; h < N; ++h )
            origin.m_state.segment( N * m + ( h - 1 ) * lift, lift ) = frame( index - h ).m_lift;
        origin.m_state.tail( 2 * L ) = window;
    }

    // The example of origin i = index-F matures with y_index: phi = [u_f; y_p; u_p; base window; extras].
    m_example = false;
    if( m_count < std::uint64_t( N + F ) )
        return origin.m_valid;
    const auto  i = index - F;
    const auto &o = frame( i );
    const int   r = c.m_ratio, uf = c.futureInputs();
    double     *buffer = m_buffer.data();
    if( !o.m_accel || !samples( o.m_anchor - L + 1, L + r - 1, o.m_segment, buffer ) )
    {
        ++m_status.m_skipped;
        return origin.m_valid;
    }
    for( int h = 0; h < F; ++h )
        m_phi.segment( h * lift, lift ) = frame( i + F - 1 - h ).m_lift;
    for( int h = 0; h < N; ++h )
        m_phi.segment( uf + h * m, m ) = frame( i - h ).m_y;
    for( int h = 1; h < N; ++h )
        m_phi.segment( uf + N * m + ( h - 1 ) * lift, lift ) = frame( i - h ).m_lift;
    const int base = uf + N * m + ( N - 1 ) * lift;
    for( int j = 0; j < L; ++j ) // Newest first: counter anchor-j is buffer entry L-1-j.
    {
        m_phi( base + 2 * j )     = buffer[2 * ( L - 1 - j )];
        m_phi( base + 2 * j + 1 ) = buffer[2 * ( L - 1 - j ) + 1];
    }
    for( int e = 0; e < 2 * ( r - 1 ); ++e ) // Extras oldest first: anchor+1, anchor+2, ...
        m_phi( base + 2 * L + e ) = buffer[2 * L + e];
    for( int h = 0; h < F; ++h )
        m_target.segment( h * m, m ) = frame( index - h ).m_y;
    m_example = true;
    return origin.m_valid;
}

bool MultirateDDSPC::learn()
{
    if( !m_example )
        return false;
    m_example = false;
    if( m_extracted )
        for( int k = 0; k < m_config.m_ratio; ++k )
        {
            m_error.noalias()    = m_theta[k].transpose() * m_phi.head( m_theta[k].rows() );
            const double squared = ( m_target - m_error ).squaredNorm();
            if( !std::isfinite( squared ) )
                continue;
            const double total = m_status.m_rmse[k] * m_status.m_rmse[k] * m_status.m_scored[k] + squared;
            m_status.m_scored[k] += m_target.size();
            m_status.m_rmse[k] = std::sqrt( total / m_status.m_scored[k] );
        }
    if( !m_training )
        return false;
    m_qr.update( m_phi, m_target );
    ++m_status.m_updates;
    m_dirty = true;
    return true;
}

void MultirateDDSPC::slot_gain(
    const MultirateConfig &config, const MatrixD &thetaT, int prefix, double lambda, MatrixD &gain )
{
    const int uf = config.futureInputs();
    if( thetaT.rows() <= uf || thetaT.cols() != config.outputs() )
        throw std::invalid_argument( "Invalid multirate coefficients" );
    const MatrixD ctc = thetaT.topRows( uf ) * thetaT.topRows( uf ).transpose();
    slot_gain( config, thetaT, ctc, prefix, config.m_scaledPenalty ? lambda * ctc.maxCoeff() : lambda, gain );
}

void MultirateDDSPC::slot_gain( const MultirateConfig &config,
                                const MatrixD         &thetaT,
                                const MatrixD         &ctc,
                                int                    prefix,
                                double                 penalty,
                                MatrixD               &gain )
{
    const int uf = config.futureInputs(), m = config.m_modes, fixed = prefix * m, v = uf - fixed;
    const int state = thetaT.rows() - uf;
    if( prefix < 0 || prefix >= config.m_slots || state < 1 )
        throw std::invalid_argument( "Invalid multirate slot gain" );
    // Within-frame variation P u_f, with P = I - (1/q) 1 1^T per mode in every future lift (P^T P = P).
    const int    q = config.m_slots, lift = config.lift();
    const double weight = config.m_ripple * ( config.m_scaledPenalty ? ctc.maxCoeff() : 1 );
    MatrixD      ripple = MatrixD::Zero( uf, uf );
    if( config.m_ripple > 0 )
        for( int f = 0; f < config.m_future; ++f )
            for( int a = 0; a < q; ++a )
                for( int b = 0; b < q; ++b )
                    ripple.block( f * lift + a * m, f * lift + b * m, m, m )
                        .diagonal()
                        .setConstant( weight * ( ( a == b ) - 1. / q ) );
    MatrixD hessian = ctc.topLeftCorner( v, v ) + ripple.topLeftCorner( v, v );
    hessian.diagonal().array() += penalty;
    const MatrixD rows     = hessian.ldlt().solve( MatrixD::Identity( v, v ).rightCols( m ) ); // H^-1 E
    const MatrixD response = thetaT.topRows( v ).transpose() * rows;                           // C_v H^-1 E
    gain.resize( m, state + fixed );
    gain.leftCols( state ).noalias() = -response.transpose() * thetaT.bottomRows( state ).transpose();
    if( fixed )
        gain.rightCols( fixed ).noalias() = -response.transpose() * thetaT.middleRows( v, fixed ).transpose() -
                                            rows.transpose() * ripple.block( 0, v, v, fixed );
}

bool MultirateDDSPC::refresh( MultirateGains &out )
{
    const auto &c    = m_config;
    m_status.m_ready = m_status.m_updates >= std::uint64_t( c.features( c.m_ratio - 1 ) );
    if( !m_dirty )
        return false;
    m_dirty = false;
    if( !m_status.m_ready )
    {
        const bool changed = out.m_ready;
        out.m_ready        = false;
        if( changed )
            out.m_version = ++m_version;
        return changed;
    }
    bool finite = true;
    for( int k = 0; k < c.m_ratio; ++k )
    {
        m_qr.solve( c.features( k ), m_theta[k] );
        finite = finite && m_theta[k].allFinite();
        if( !finite )
            break;
        const auto uf        = c.futureInputs();
        m_ctc.noalias()      = m_theta[k].topRows( uf ) * m_theta[k].topRows( uf ).transpose();
        const double penalty = c.m_scaledPenalty ? m_lambda * m_ctc.maxCoeff() : m_lambda;
        for( int p = 0; p < c.m_slots && finite; ++p )
        {
            auto &gain = out.m_slot[k * c.m_slots + p];
            slot_gain( c, m_theta[k], m_ctc, p, penalty, gain );
            finite = gain.allFinite();
        }
    }
    m_extracted   = finite;
    out.m_ready   = finite;
    out.m_version = ++m_version;
    ++m_status.m_refreshes;
    return true;
}

void MultirateDDSPC::allocate( MultirateGains &gains ) const
{
    const auto &c = m_config;
    gains.m_slot.resize( c.m_ratio * c.m_slots );
    for( int k = 0; k < c.m_ratio; ++k )
        for( int p = 0; p < c.m_slots; ++p )
            gains.m_slot[k * c.m_slots + p] = MatrixD::Zero( c.m_modes, c.state( k ) + p * c.m_modes );
    gains.m_ready = false;
}

void MultirateDDSPC::allocate( MultirateOrigin &origin ) const
{
    origin.m_state = VectorD::Zero( m_config.state( 0 ) );
    origin.m_valid = false;
}

const MultirateStatus &MultirateDDSPC::status() const
{
    return m_status;
}

const VectorD &MultirateDDSPC::regressor() const
{
    return m_phi;
}

const VectorD &MultirateDDSPC::target() const
{
    return m_target;
}

const MatrixD &MultirateDDSPC::coefficients( int phase ) const
{
    return m_theta.at( phase );
}

const InformationQR &MultirateDDSPC::factor() const
{
    return m_qr;
}

void MultirateDDSPC::command( const MultirateConfig &config,
                              const MultirateGains  &gains,
                              const MultirateOrigin &origin,
                              int                    phase,
                              int                    prefix,
                              const double          *extras,
                              const double          *executed,
                              double                *out )
{
    const int           m = config.m_modes, s = config.state( 0 );
    const MatrixD      &gain = gains.m_slot[phase * config.m_slots + prefix];
    Eigen::Map<VectorD> command( out, m );
    command.noalias() = gain.leftCols( s ) * origin.m_state;
    if( phase )
        command.noalias() += gain.middleCols( s, 2 * phase ) * Eigen::Map<const VectorD>( extras, 2 * phase );
    if( prefix )
        command.noalias() += gain.rightCols( prefix * m ) * Eigen::Map<const VectorD>( executed, prefix * m );
}

} // namespace DDSPC
