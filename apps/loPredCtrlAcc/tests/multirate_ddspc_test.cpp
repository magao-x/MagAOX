/** \file multirate_ddspc_test.cpp
 * \brief T2/T3/T5: shared-factor phase predictors equal independent and batch least squares; slot-law consistency.
 * \ingroup loPredCtrlAcc_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../multirate_ddspc.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>

namespace libXWCTest::loPredCtrlAccTest
{
/// \cond MULTIRATE_TEST_SUPPORT
namespace
{
using DDSPC::MatrixD;
using DDSPC::VectorD;

/// Exponentially weighted batch least squares with prior rho*I, solved from the normal equations.
class BatchLS
{
    MatrixD m_gram, m_cross; ///< Weighted Gram matrix and cross products.
    double  m_gamma;         ///< Forgetting factor.

  public:
    /// Start from the prior.
    BatchLS( int    features /**< [in] Regressor size. */,
             int    outputs /**< [in] Target size. */,
             double gamma /**< [in] Forgetting factor. */,
             double prior /**< [in] Prior information. */ )
        : m_gram( prior * MatrixD::Identity( features, features ) ), m_cross( MatrixD::Zero( features, outputs ) ),
          m_gamma( gamma )
    {
    }

    /// Add one example.
    void add( const VectorD &x /**< [in] Regressor. */, const VectorD &y /**< [in] Target. */ )
    {
        m_gram  = m_gamma * m_gram + x * x.transpose();
        m_cross = m_gamma * m_cross + x * y.transpose();
    }

    /// Theta^T of the leading regressor entries.
    MatrixD solve( int leading /**< [in] Leading entries. */ ) const
    {
        return m_gram.topLeftCorner( leading, leading ).ldlt().solve( m_cross.topRows( leading ) );
    }
};

/// Largest absolute difference relative to the largest coefficient.
double relative( const MatrixD &a /**< [in] Estimate. */, const MatrixD &b /**< [in] Reference. */ )
{
    return ( a - b ).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff();
}

/// RTC dimensions of the assisted model.
DDSPC::MultirateConfig rtc()
{
    DDSPC::MultirateConfig config;
    config.m_prior = 1e-3;
    return config;
}

/// Drive the learner with a vibrating plant, random probes and ACC samples correlated with the disturbance.
/** Returns the number of updates; every matured regressor/target is passed to the callback. */
template <class callbackT>
std::uint64_t drive( DDSPC::MultirateDDSPC &learner /**< [in,out] Learner under test. */,
                     int                    frames /**< [in] WFS frames. */,
                     callbackT              matured /**< [in] Called with each learned example. */ )
{
    const auto                      &c = learner.config();
    std::mt19937                     rng( 3 );
    std::normal_distribution<double> noise( 0, 1 );
    DDSPC::MultirateOrigin           origin;
    learner.allocate( origin );
    std::vector<double> acc;
    MatrixD             applied( c.m_modes, c.m_slots );
    VectorD             y( c.m_modes ), window( 2 * c.window() );
    std::uint64_t       updates = 0;
    for( int i = 0; i < frames; ++i )
    {
        for( int n = 0; n < c.m_ratio; ++n )
        {
            const double t = double( i * c.m_ratio + n ) / c.m_ratio;
            acc.push_back( std::sin( 1.9 * t ) + .3 * noise( rng ) );
            acc.push_back( std::cos( 1.3 * t ) + .3 * noise( rng ) );
        }
        const std::uint64_t anchor = acc.size() / 2 - 1;
        for( int j = 0; j < c.window(); ++j )
            for( int a = 0; a < 2; ++a )
                window( 2 * j + a ) = anchor >= std::uint64_t( j ) ? acc[2 * ( anchor - j ) + a] : 0;
        for( int mode = 0; mode < c.m_modes; ++mode )
            y( mode ) = std::sin( 1.9 * i + mode ) + .2 * applied.row( mode ).sum() + .1 * noise( rng );
        for( int j = 0; j < c.m_slots; ++j )
            for( int mode = 0; mode < c.m_modes; ++mode )
                applied( mode, j ) = .5 * noise( rng );
        learner.append( i + 1,
                        y,
                        i ? applied : MatrixD(),
                        anchor + 1 >= std::uint64_t( c.window() ) ? window : VectorD(),
                        anchor,
                        1,
                        origin );
        const VectorD x = learner.regressor(), target = learner.target();
        if( learner.learn() )
        {
            ++updates;
            matured( x, target );
        }
    }
    return updates;
}
} // namespace
/// \endcond

/** \brief T2: the leading block of one information factor is exactly the factor of the leading-entry problem. */
TEST_CASE( "Information factor leading subsets equal independent and batch least squares", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::InformationQR::update;
    DDSPC::InformationQR::solve;
#endif
    // clang-format on
    const double                      gamma = GENERATE( 1.0, 0.995 );
    const int                         n = 30, o = 4;
    std::mt19937                      rng( 11 );
    std::normal_distribution<double>  noise( 0, 1 );
    DDSPC::InformationQR              full( n, o, gamma, 1e-3 );
    std::vector<DDSPC::InformationQR> nested;
    for( int k = 0; k < 4; ++k )
        nested.emplace_back( n - 6 + 2 * k, o, gamma, 1e-3 );
    BatchLS batch( n, o, gamma, 1e-3 );
    for( int t = 0; t < 500; ++t )
    {
        VectorD x( n ), y( o );
        for( int j = 0; j < n; ++j )
            x( j ) = noise( rng ) + ( j % 3 ? 0. : .5 * x( 0 ) );
        y = MatrixD::Ones( o, n ) * x / n + .1 * VectorD::NullaryExpr( o, [&] { return noise( rng ); } );
        full.update( x, y );
        for( auto &qr : nested )
            qr.update( x.head( qr.features() ), y );
        batch.add( x, y );
    }
    double worstNested = 0, worstBatch = 0;
    for( const auto &qr : nested )
    {
        MatrixD shared( qr.features(), o ), alone( qr.features(), o );
        full.solve( qr.features(), shared );
        qr.solve( qr.features(), alone );
        worstNested = std::max( worstNested, relative( shared, alone ) );
        worstBatch  = std::max( worstBatch, relative( shared, batch.solve( qr.features() ) ) );
    }
    std::printf(
        "T2 core gamma %.3f: shared vs independent %.3g, shared vs batch %.3g\n", gamma, worstNested, worstBatch );
    REQUIRE( worstNested <= 1e-12 );
    REQUIRE( worstBatch <= 1e-8 );
}

/** \brief T2/T3 at RTC dimensions: one update per genuine WFS target; every phase equals its own least squares. */
TEST_CASE( "Multirate learner phase coefficients equal independent least squares", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateDDSPC::append;
    DDSPC::MultirateDDSPC::learn;
    DDSPC::MultirateDDSPC::refresh;
#endif
    // clang-format on
    auto                  config = rtc();
    DDSPC::MultirateDDSPC learner( config );
    learner.training( true );
    const int                         r = config.m_ratio, o = config.outputs();
    BatchLS                           batch( config.features( r - 1 ), o, config.m_gamma, config.m_prior );
    std::vector<DDSPC::InformationQR> nested;
    for( int k = 0; k < r; ++k )
        nested.emplace_back( config.features( k ), o, config.m_gamma, config.m_prior );
    std::vector<VectorD> targets;
    const int            frames  = 700;
    const auto           updates = drive( learner,
                                          frames,
                                          [&]( const VectorD &x, const VectorD &y )
                                          {
                                    batch.add( x, y );
                                    for( auto &qr : nested )
                                        qr.update( x.head( qr.features() ), y );
                                    targets.push_back( y );
                                          } );
    // Each matured origin is learned once; the first N+F-1 frames only fill the histories.
    REQUIRE( updates == std::uint64_t( frames - ( config.m_history + config.m_future - 1 ) ) );
    REQUIRE( learner.status().m_updates == updates );
    for( std::size_t t = 1; t < targets.size(); ++t ) // Consecutive targets are distinct genuine frames.
        REQUIRE( targets[t].head( config.m_modes ) != targets[t - 1].head( config.m_modes ) );
    DDSPC::MultirateGains gains;
    learner.allocate( gains );
    REQUIRE( learner.refresh( gains ) );
    REQUIRE( gains.m_ready );
    double worstNested = 0, worstBatch = 0;
    for( int k = 0; k < r; ++k )
    {
        MatrixD alone( config.features( k ), o );
        nested[k].solve( config.features( k ), alone );
        worstNested = std::max( worstNested, relative( learner.coefficients( k ), alone ) );
        worstBatch = std::max( worstBatch, relative( learner.coefficients( k ), batch.solve( config.features( k ) ) ) );
    }
    std::printf( "T2 learner (%d features): shared vs four independent QR %.3g, vs batch LS %.3g\n",
                 config.features( r - 1 ),
                 worstNested,
                 worstBatch );
    REQUIRE( worstNested <= 1e-12 );
    REQUIRE( worstBatch <= 1e-6 );
}

/** \brief T5 and law check: each slot gain is the next-slot row of the fixed-prefix DDSPC solution, and with no extra
 * ACC information re-solving every slot reproduces the frame-rate plan exactly.
 */
TEST_CASE( "Multirate slot law re-solves the original DDSPC cost", "[multirate]" )
{
    // clang-format off
#ifdef LOPREDCTRLACC_TEST_DOXYGEN_REF
    DDSPC::MultirateDDSPC::slot_gain;
    DDSPC::MultirateDDSPC::command;
#endif
    // clang-format on
    for( const double ripple : { 0., 10. } )
    {
        auto config           = rtc();
        config.m_history      = 4;
        config.m_future       = 2;
        config.m_accelHistory = 2;
        config.m_ripple       = ripple;
        const int    m = config.m_modes, q = config.m_slots, uf = config.futureInputs(), s0 = config.state( 0 );
        const double lambda = .7;
        std::mt19937 rng( 5 );
        std::normal_distribution<double> noise( 0, 1 );
        // Theta_0 random; Theta_k appends zero extra-ACC coefficients (no additional information).
        MatrixD theta0 = MatrixD::NullaryExpr( config.features( 0 ), config.outputs(), [&] { return noise( rng ); } );
        DDSPC::MultirateGains gains;
        DDSPC::MultirateDDSPC( config ).allocate( gains );
        gains.m_ready = true;
        for( int k = 0; k < config.m_ratio; ++k )
        {
            MatrixD theta                  = MatrixD::Zero( config.features( k ), config.outputs() );
            theta.topRows( theta0.rows() ) = theta0;
            for( int p = 0; p < q; ++p )
                DDSPC::MultirateDDSPC::slot_gain( config, theta, p, lambda, gains.m_slot[k * q + p] );
        }
        DDSPC::MultirateOrigin origin;
        origin.m_state      = VectorD::NullaryExpr( s0, [&] { return noise( rng ); } );
        const MatrixD c     = theta0.topRows( uf ).transpose();
        const VectorD x     = theta0.bottomRows( s0 ).transpose() * origin.m_state;
        const double  scale = ( c.transpose() * c ).maxCoeff();
        // Full quadratic of the original cost plus the within-frame variation term: (C^T C + lambda I + mu P).
        MatrixD hessian = c.transpose() * c;
        hessian.diagonal().array() += lambda * scale;
        for( int f = 0; f < config.m_future; ++f )
            for( int a = 0; a < q; ++a )
                for( int b = 0; b < q; ++b )
                    for( int mode = 0; mode < m; ++mode )
                        hessian( f * q * m + a * m + mode, f * q * m + b * m + mode ) +=
                            ripple * scale * ( ( a == b ) - 1. / q );
        const VectorD plan = -hessian.ldlt().solve( c.transpose() * x ); // Frame-rate plan, newest first.
        // Re-solve at every slot with the executed prefix fixed at the planned values and k = p extra samples.
        VectorD executed = VectorD::Zero( q * m ), extras = VectorD::NullaryExpr( 2 * q, [&] { return noise( rng ); } );
        double  worst = 0;
        for( int p = 0; p < q; ++p )
        {
            VectorD prefix( p * m );
            for( int j = 0; j < p; ++j ) // Newest first.
                prefix.segment( ( p - 1 - j ) * m, m ) = executed.segment( j * m, m );
            double command[16];
            DDSPC::MultirateDDSPC::command(
                config, gains, origin, std::min( p, config.m_ratio - 1 ), p, extras.data(), prefix.data(), command );
            const VectorD planned = plan.segment( uf - ( p + 1 ) * m, m ); // Slot p of the current frame.
            worst = std::max( worst, ( Eigen::Map<VectorD>( command, m ) - planned ).cwiseAbs().maxCoeff() );
            executed.segment( p * m, m ) = planned;
            // The gain is the next-slot row of the explicit fixed-prefix solution for an arbitrary prefix.
            const VectorD arbitrary = VectorD::NullaryExpr( p * m, [&] { return noise( rng ); } );
            const int     v         = uf - p * m;
            const VectorD rest      = -hessian.topLeftCorner( v, v ).ldlt().solve(
                c.leftCols( v ).transpose() * x + hessian.block( 0, v, v, p * m ) * arbitrary );
            DDSPC::MultirateDDSPC::command( config, gains, origin, 0, p, extras.data(), arbitrary.data(), command );
            REQUIRE( ( Eigen::Map<VectorD>( command, m ) - rest.tail( m ) ).cwiseAbs().maxCoeff() <=
                     1e-9 * rest.cwiseAbs().maxCoeff() );
        }
        std::printf( "T5 ripple %g: no-information re-solve vs frame-rate plan: %.3g\n", ripple, worst );
        REQUIRE( worst <= 1e-9 * plan.cwiseAbs().maxCoeff() );
    }
}
} // namespace libXWCTest::loPredCtrlAccTest

namespace libXWCTest::loPredCtrlAccTest
{
/** \brief Learner cost per WFS frame at RTC dimensions (hidden benchmark). */
TEST_CASE( "Multirate learner frame cost", "[.][multirate-benchmark]" )
{
    DDSPC::MultirateDDSPC learner( rtc() );
    learner.training( true );
    DDSPC::MultirateGains gains;
    learner.allocate( gains );
    std::vector<double> update, refresh;
    drive( learner,
           2000,
           [&]( const VectorD &, const VectorD & )
           {
               const auto start = std::chrono::steady_clock::now();
               learner.refresh( gains );
               refresh.push_back(
                   std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - start ).count() );
           } );
    std::sort( refresh.begin(), refresh.end() );
    std::printf( "refresh (extract 4 phases + 16 gains): median %.1f us p99 %.1f us\n",
                 refresh[refresh.size() / 2],
                 refresh[refresh.size() * 99 / 100] );
}
} // namespace libXWCTest::loPredCtrlAccTest
