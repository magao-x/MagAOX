/** \file operational_law.cpp
 * \brief Preserve the Legacy full-Gram, scaled-inverse control expression in an offline causal-record adapter.
 */
#include "operational_law.hpp"
#include <cmath>
#include <stdexcept>

namespace DDSPC
{
Matrix operational_legacy_gain( const Matrix &theta, int modes, int history, realT regularization, realT feedback )
{
    const int outputs = theta.rows(), a = modes * history, b = modes * ( history - 1 );
    const int acceleration = theta.cols() - outputs - a - b;
    if( modes < 1 || history < 1 || outputs < modes || outputs % modes || acceleration < 0 || !theta.allFinite() ||
        !std::isfinite( regularization ) || regularization <= 0 || !std::isfinite( feedback ) )
        throw std::invalid_argument( "Invalid offline Legacy law dimensions/parameters" );
    Matrix internal( outputs, theta.cols() );
    internal << theta.rightCols( outputs ), theta.middleCols( a, b ), theta.middleCols( a + b, acceleration ),
        theta.leftCols( a );
    // Match PredictiveController::update_controller(), including full Gram and explicit inverse.
    Matrix h   = internal.transpose() * internal;
    Matrix h11 = h.topLeftCorner( outputs, outputs );
    Matrix h21 = h.block( 0, outputs, outputs, h.cols() - outputs );
    if( !h.allFinite() || h11.maxCoeff() <= 0 )
        throw std::runtime_error( "Legacy scaled penalty is degenerate" );
    Matrix penalty = regularization * Matrix::Identity( outputs, outputs );
    Matrix full    = -1 * ( h11 + h11.maxCoeff() * penalty ).inverse() * h21;
    Matrix result( outputs, a + b + acceleration );
    result << full.rightCols( a ), full.leftCols( b ), full.middleCols( b, acceleration );
    // Legacy adds feedback only to its next (last newest-first) command block.
    result.block( outputs - modes, 0, modes, modes ).diagonal().array() += feedback;
    if( !result.allFinite() )
        throw std::runtime_error( "Nonfinite offline Legacy control law" );
    return result;
}
} //namespace DDSPC
