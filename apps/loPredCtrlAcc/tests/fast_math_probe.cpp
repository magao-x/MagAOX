/** \file fast_math_probe.cpp
 * \brief Offline isolation of shared reciprocal overflow in float Givens normalization under unsafe optimization.
 */
#include "../qrd_rls.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

namespace
{
/// Inspect exponent bits even when finite-only compilation invalidates standard isfinite assumptions.
bool finite( float value /**< [in] IEEE single-precision result. */ )
{
    std::uint32_t bits;
    std::memcpy( &bits, &value, sizeof( bits ) );
    return ( bits & 0x7f800000U ) != 0x7f800000U;
}
/// Return normalized coefficients without reference-argument aliasing that would suppress the optimization under test.
struct Rotation
{
    float m_r, m_c, m_s; ///< Hypot and normalized row coefficients.
};
/// Preserve the original two-division normalization as an inspectable function in disassembly.
__attribute__( ( noinline ) ) Rotation original( float a /**< [in] Upper entry. */, float b /**< [in] Lower entry. */ )
{
    const float r = std::hypot( a, b );
    return { r, a / r, -b / r };
}
/// Test the same normalization with an exponent range accommodating reciprocal(FLT_TRUE_MIN).
__attribute__( ( noinline ) ) Rotation wider( float a /**< [in] Upper entry. */, float b /**< [in] Lower entry. */ )
{
    const double h = std::hypot( double( a ), double( b ) );
    return { float( h ), float( double( a ) / h ), float( -double( b ) / h ) };
}
#if defined( __GNUC__ ) && !defined( __clang__ )
    #define SCOPED_DIVISION __attribute__( ( noinline, optimize( "no-reciprocal-math" ) ) )
#else
    #define SCOPED_DIVISION __attribute__( ( noinline ) )
#endif
/// GNU-only local experiment: retain divisions without changing any global or production compiler flag.
SCOPED_DIVISION Rotation scoped( float a /**< [in] Upper entry. */, float b /**< [in] Lower entry. */ )
{
    const float r = std::hypot( a, b );
    return { r, a / r, -b / r };
}
/// Exercise a complete adjacent-row rotation with an isolated candidate normalizer; never replace the real kernel.
template <bool useWide /**< [in] Use double normalization rather than locally scoped float division. */>
DDSPC::Matrix candidate_rotation( DDSPC::Matrix a /**< [in] Same augmented test matrix as the original kernel. */ )
{
    const int n = a.cols();
    for( int j = 0; j < n; ++j )
        for( int i = n; i > j; --i )
        {
            if( a( i, j ) == 0 )
                continue;
            const auto r = useWide ? wider( a( i - 1, j ), a( i, j ) ) : scoped( a( i - 1, j ), a( i, j ) );
            for( int k = j; k < n; ++k )
            {
                const float upper = a( i - 1, k ), lower = a( i, k );
                a( i - 1, k ) = r.m_c * upper - r.m_s * lower;
                a( i, k )     = r.m_s * upper + r.m_c * lower;
            }
        }
    return a;
}
/// Validate raw exponents of an entire result independently of finite-only optimizer assumptions.
bool finite_matrix( const DDSPC::Matrix &a /**< [in] Rotation output. */ )
{
    for( int i = 0; i < a.size(); ++i )
        if( !finite( a.data()[i] ) )
            return false;
    return true;
}
} //namespace

/// Export both the tiny failure and a finite exponent grid; all corrections are diagnostic-only functions.
int main()
{
    std::cout << "exponent,r,original_c,original_s,original_finite,wide_finite,scoped_finite,wide_unit_error\n"
              << std::setprecision( 12 );
    for( int exponent : { -140, -130, -126, -100, -10, 0, 100, 120 } )
    {
        volatile float av = std::ldexp( 1.f, exponent ), bv = -2 * av;
        const float    a = av, b = bv;
        const auto     o = original( a, b ), w = wider( a, b ), s = scoped( a, b );
        std::cout << exponent << ',' << o.m_r << ',' << o.m_c << ',' << o.m_s << ','
                  << ( finite( o.m_c ) && finite( o.m_s ) ) << ',' << ( finite( w.m_c ) && finite( w.m_s ) ) << ','
                  << ( finite( s.m_c ) && finite( s.m_s ) ) << ','
                  << std::abs( double( w.m_c ) * w.m_c + double( w.m_s ) * w.m_s - 1 ) << '\n';
    }
    DDSPC::QRDRecursiveLeastSquares old( 3, 4, 1, 1 );
    old.A_.setZero();
    old.A_.row( 0 ) << 1.e-40f, .123f, -.578f, -.578f;
    old.A_.row( 1 ) << -2.e-40f, .456f, .382f, .382f;
    const auto input = old.A_;
    old.apply_givens();
    bool valid = true;
    for( int j = 0; j < old.A_.size(); ++j )
        valid = valid && finite( old.A_.data()[j] );
    std::cerr << "compiler=" << __VERSION__ << " original_tiny_rotation_finite=" << valid
              << " wide_tiny_rotation_finite=" << finite_matrix( candidate_rotation<true>( input ) )
              << " scoped_tiny_rotation_finite=" << finite_matrix( candidate_rotation<false>( input ) ) << '\n';
}
