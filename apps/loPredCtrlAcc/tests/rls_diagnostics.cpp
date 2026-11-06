/** \file rls_diagnostics.cpp
 * \brief Identical-record covariance/weighted-LS replay of Phase 10 traces; no control or application interfaces.
 */
#include "../ar_controller.hpp"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

namespace
{
using namespace DDSPC;
using Double = Eigen::MatrixXd;
/// One actual synthetic record, copied from the persisted Phase 10 trace rather than a model's proposed action.
struct Row
{
    Matrix m_y{ 2, 1 }, m_u{ 2, 1 }, m_a{ 2, 1 }; ///< Measured residual, applied input and causal sensor features.
    double m_norm{ 0 };      ///< Origin coefficient norm observed in the next frame, zero if no snapshot was read.
    bool   m_valid{ false }; ///< Selected snapshot was available.
};
/// Split a simple numerical trace row without any string interpretation as code or external commands.
std::vector<std::string> split( const std::string &line /**< [in] CSV record. */ )
{
    std::vector<std::string> fields;
    std::istringstream       in( line );
    std::string              field;
    while( std::getline( in, field, ',' ) )
        fields.push_back( field );
    return fields;
}
/// Convert internal [C B D A] to the exact canonical snapshot layout before comparing norm bits.
Matrix canonical( const Matrix &theta /**< [in] Internal coefficients. */, int d /**< [in] Sensor feature rows. */ )
{
    Matrix result( theta.rows(), theta.cols() );
    result << theta.rightCols( 10 ), theta.middleCols( 6, 8 ), theta.middleCols( 14, d ), theta.leftCols( 6 );
    return result;
}
} //namespace

/// Replay normal contiguous source histories and report numerical evolution for multiple estimators on identical phi/y.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Trace, model=3/4, optional gamma/prior. */ )
{
    if( argc < 3 || argc > 5 )
        return 2;
    const int model = std::stoi( argv[2] );
    if( model != 3 && model != 4 )
        return 2;
    const float   gamma = argc > 3 ? std::stof( argv[3] ) : float( .9995 );
    const float   prior = argc > 4 ? std::stof( argv[4] ) : float( .001 );
    std::ifstream input( argv[1] );
    std::string   line;
    std::getline( input, line );
    const auto header = split( line );
    const auto col    = [&]( const std::string &name )
    {
        const auto it = std::find( header.begin(), header.end(), name );
        if( it == header.end() )
            throw std::invalid_argument( "Missing numerical trace column" );
        return std::size_t( it - header.begin() );
    };
    std::vector<Row> data;
    while( std::getline( input, line ) )
    {
        const auto fields = split( line );
        if( std::stoi( fields[0] ) != model )
            continue;
        if( std::stoi( fields[1] ) != int( data.size() ) )
            throw std::invalid_argument( "Noncontiguous trace" );
        Row r;
        for( int j = 0; j < 2; ++j )
        {
            r.m_y( j ) = std::stof( fields[col( j ? "yy" : "yx" )] );
            r.m_u( j ) = std::stof( fields[col( j ? "applied_y" : "applied_x" )] );
            r.m_a( j ) = std::stof( fields[col( j ? "ay" : "ax" )] );
        }
        r.m_valid = std::stoi( fields[col( "valid" )] );
        r.m_norm  = std::stod( fields[col( "theta_norm" )] );
        data.push_back( r );
    }
    if( data.empty() )
        return 2;
    const int            sensor = model == 4 ? 10 : 0, p = 24 + sensor;
    PredictiveController history(
        2, 5, 3, 0, gamma, .02f, prior, model == 4 ? 2 : 0, model == 4 ? 5 : 0, HistoryMode::Causal );
    auto                     workspace = history.make_workspace();
    RecursiveLeastSquares    native( 6, p, gamma, prior ), sym( 6, p, gamma, prior ), joseph( 6, p, gamma, prior );
    QRDRecursiveLeastSquares qr( 6, p, gamma, prior, true );
    Matrix                   norm( 1, 1 ), weight( 6, p ), covariance( p, p );
    Double                   pd = Double::Identity( p, p ) / double( prior ), wd = Double::Zero( 6, p );
    Double                   info = double( prior ) * Double::Identity( p, p ), rhs = Double::Zero( 6, p );
    const bool               sameTrace = argc == 3 && std::string( argv[1] ).find( "_qr.csv" ) == std::string::npos;
    double                   replayDifference = sameTrace ? 0 : std::numeric_limits<double>::quiet_NaN();
    std::cout << std::setprecision( 12 )
              << "frame,updates,gamma,prior,theta_native,theta_qr,theta_double,"
                 "condition_info,p_min,p_max,p_asymmetry,p_inverse_error,denominator,theta_error_native,theta_error_qr,"
                 "theta_error_double,theta_error_symmetric,theta_error_joseph,sym_min,joseph_min,double_min,replay_"
                 "norm_difference\n";
    int updates = 0;
    for( std::size_t i = 0; i < data.size(); ++i )
    {
        history.append_sample( i, data[i].m_y, data[i].m_u, model == 4 ? data[i].m_a : Matrix() );
        if( !history.prepare_training( *workspace ) )
            continue;
        const auto  &x = workspace->regressor(), &y = workspace->target();
        const Double xd = x.cast<double>(), yd = y.cast<double>();
        info                = double( gamma ) * info + xd * xd.transpose();
        rhs                 = double( gamma ) * rhs + yd * xd.transpose();
        const Double priorD = pd / double( gamma ), kd = priorD * xd / ( 1 + ( xd.transpose() * priorD * xd )( 0 ) );
        wd += ( yd - wd * xd ) * kd.transpose();
        pd = priorD - kd * ( xd.transpose() * priorD );
        native.update_preallocated( x, y, norm, weight, covariance );
        const double denominator = double( 1.f + norm( 0 ) );
        sym.update_preallocated( x, y, norm, weight, covariance );
        sym.inverse_covariance = ( .5f * ( sym.inverse_covariance + sym.inverse_covariance.transpose() ) ).eval();
        const Matrix priorJ    = joseph.inverse_covariance / gamma;
        joseph.update_preallocated( x, y, norm, weight, covariance );
        const Matrix left         = Matrix::Identity( p, p ) - joseph.K.transpose() * x.transpose();
        joseph.inverse_covariance = left * priorJ * left.transpose() + joseph.K.transpose() * joseph.K;
        Matrix mutableX           = x;
        qr.update_preallocated( mutableX, y, weight );
        ++updates;
        const float thetaNorm = canonical( native.prediction_matrix, sensor ).norm();
        if( sameTrace && i + 1 < data.size() && data[i + 1].m_valid && data[i + 1].m_norm )
            replayDifference =
                std::max( replayDifference, std::abs( double( thetaNorm ) - double( float( data[i + 1].m_norm ) ) ) );
        if( i % 20 && i + 1 != data.size() )
            continue;
        const Double                          pc = native.inverse_covariance.cast<double>();
        Eigen::SelfAdjointEigenSolver<Double> ip( info ), eig( ( .5 * ( pc + pc.transpose() ) ).eval() );
        Eigen::SelfAdjointEigenSolver<Double> se( sym.inverse_covariance.cast<double>() ),
            je( ( .5 * ( joseph.inverse_covariance + joseph.inverse_covariance.transpose() ) ).cast<double>() ),
            de( pd );
        const Double exactP = info.ldlt().solve( Double::Identity( p, p ) ), exactW = rhs * exactP;
        std::cout << i << ',' << updates << ',' << gamma << ',' << prior << ',' << thetaNorm << ','
                  << qr.prediction_matrix.norm() << ',' << wd.norm() << ','
                  << ip.eigenvalues().maxCoeff() / ip.eigenvalues().minCoeff() << ',' << eig.eigenvalues().minCoeff()
                  << ',' << eig.eigenvalues().maxCoeff() << ',' << ( pc - pc.transpose() ).norm() / pc.norm() << ','
                  << ( pc - exactP ).norm() / exactP.norm() << ',' << denominator << ','
                  << ( native.prediction_matrix.cast<double>() - exactW ).norm() << ','
                  << ( qr.prediction_matrix.cast<double>() - exactW ).norm() << ',' << ( wd - exactW ).norm() << ','
                  << ( sym.prediction_matrix.cast<double>() - exactW ).norm() << ','
                  << ( joseph.prediction_matrix.cast<double>() - exactW ).norm() << ',' << se.eigenvalues().minCoeff()
                  << ',' << je.eigenvalues().minCoeff() << ',' << de.eigenvalues().minCoeff() << ',' << replayDifference
                  << '\n';
    }
    std::cerr << "max_trace_norm_difference=" << replayDifference << " gamma=" << gamma << " prior=" << prior << '\n';
}
