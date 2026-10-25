/** \file ocamUtils.hpp
 * \brief Utilities for parsing OCAM camera query responses.
 *
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamUtils_hpp
#define ocamUtils_hpp

#include <array>
#include <charconv>
#include <string_view>

#include <mx/ioutils/stringUtils.hpp>
#include <mx/math/floatUtils.hpp>

namespace MagAOX
{
namespace app
{

/// Structure to hold the OCAM camera temperature readings returned by the device.
/**
 * \ingroup ocam2KCtrl
 */
struct ocamTemps
{
    float CCD{ 0 };           ///< Detector temperature in degrees Celsius.
    float CPU{ 0 };           ///< Camera CPU temperature in degrees Celsius.
    float POWER{ 0 };         ///< Power supply temperature in degrees Celsius.
    float BIAS{ 0 };          ///< Bias temperature in degrees Celsius.
    float WATER{ 0 };         ///< Cooling water temperature in degrees Celsius.
    float LEFT{ 0 };          ///< Left amplifier temperature in degrees Celsius.
    float RIGHT{ 0 };         ///< Right amplifier temperature in degrees Celsius.
    float SET{ 0 };           ///< Reported detector setpoint in degrees Celsius.
    float COOLING_POWER{ 0 }; ///< Cooling power in the camera's reported units.

    /// Compare the temperatures and cooling power used for change detection.
    /** The CPU reading is intentionally excluded to preserve existing telemetry behavior.
     * \returns true if the compared readings are equal.
     */
    bool operator==( const ocamTemps &t /**< [in] Readings to compare. */ ) const;

    /// Set all values to the invalid value, -999.
    /** \returns 0.
     */
    int setInvalid();
};

inline bool ocamTemps::operator==( const ocamTemps &t ) const
{
    return ( CCD == t.CCD && POWER == t.POWER && BIAS == t.BIAS && WATER == t.WATER && LEFT == t.LEFT &&
             RIGHT == t.RIGHT && SET == t.SET && COOLING_POWER == t.COOLING_POWER );
}

inline int ocamTemps::setInvalid()
{
    CCD           = -999;
    CPU           = -999;
    POWER         = -999;
    BIAS          = -999;
    WATER         = -999;
    LEFT          = -999;
    RIGHT         = -999;
    SET           = -999;
    COOLING_POWER = -999;

    return 0;
}

/// Implementation details shared by the OCAM response parsers.
/** \cond OCAM_UTILS_DETAIL
 */
namespace ocamUtilsDetail
{

/// Remove surrounding serial-response whitespace without copying the token.
inline std::string_view trim( std::string_view token /**< [in] Response token. */ )
{
    constexpr std::string_view whitespace{ " \t\r\n" };
    size_t                     first = token.find_first_not_of( whitespace );

    if( first == std::string_view::npos )
    {
        return {};
    }

    return token.substr( first, token.find_last_not_of( whitespace ) - first + 1 );
}

/// Parse a complete finite decimal token, preserving the output on failure.
inline bool parseFloat( float           &value, /**< [out] Parsed value on success. */
                        std::string_view token /**< [in] Numeric response token. */ )
{
    token = trim( token );

    if( !token.empty() && token.front() == '+' )
    {
        token.remove_prefix( 1 );

        if( !token.empty() && token.front() == '-' )
        {
            return false;
        }
    }

    if( token.empty() )
    {
        return false;
    }

    float parsed;
    auto  result = std::from_chars( token.data(), token.data() + token.size(), parsed );

    if( result.ec != std::errc{} || result.ptr != token.data() + token.size() || !mx::math::isFinite( parsed ) )
    {
        return false;
    }

    value = parsed;
    return true;
}

} // namespace ocamUtilsDetail
/** \endcond
 */

/// Parse a complete, labeled OCAM temperature query response.
/** All fields must be finite; temperatures below absolute zero and negative cooling power are rejected.
 * The returned SET field is converted from tenths of a degree to degrees Celsius.
 *
 * \returns 0 on success, with every output field updated.
 * \returns -1 on error, leaving the output unchanged.
 *
 * \ingroup ocam2KCtrl
 */
inline int parseTemps( ocamTemps         &temps, /**< [out] Temperature readings on success. */
                       const std::string &tstr /**< [in] Device response to parse. */ )
{
    std::vector<std::string> v;
    mx::ioutils::parseStringVector( v, tstr, "[]" );

    constexpr std::array<std::string_view, 10> labels{
        "Temperatures : CCD", "CPU", "POWER", "BIAS", "WATER", "LEFT", "RIGHT", "SET", "Cooling Power", "mW." };

    if( v.size() != 19 )
    {
        return -1;
    }

    std::array<float, 9> values;

    for( size_t i = 0; i < labels.size(); ++i )
    {
        if( ocamUtilsDetail::trim( v[2 * i] ) != labels[i] )
        {
            return -1;
        }

        if( i < values.size() && !ocamUtilsDetail::parseFloat( values[i], v[2 * i + 1] ) )
        {
            return -1;
        }
    }

    values[7] /= 10.0F;

    for( size_t i = 0; i < 8; ++i )
    {
        if( values[i] < -273.15F )
        {
            return -1;
        }
    }

    if( values[8] < 0 )
    {
        return -1;
    }

    temps.CCD           = values[0];
    temps.CPU           = values[1];
    temps.POWER         = values[2];
    temps.BIAS          = values[3];
    temps.WATER         = values[4];
    temps.LEFT          = values[5];
    temps.RIGHT         = values[6];
    temps.SET           = values[7];
    temps.COOLING_POWER = values[8];

    return 0;
}

/// Parse the finite positive rate in an OCAM "fps [value] Hz" query response.
/** \returns 0 on success.
 * \returns -1 on error, leaving the output unchanged.
 *
 * \ingroup ocam2KCtrl
 */
inline int parseFPS( float             &fps, /**< [out] Camera rate on success, in Hz. */
                     const std::string &fstr /**< [in] Device response to parse. */ )
{
    std::vector<std::string> v;
    mx::ioutils::parseStringVector( v, fstr, "[]" );

    float parsed;

    if( v.size() != 3 || ocamUtilsDetail::trim( v[0] ) != "fps" || ocamUtilsDetail::trim( v[2] ) != "Hz" ||
        !ocamUtilsDetail::parseFloat( parsed, v[1] ) || parsed <= 0 )
    {
        return -1;
    }

    fps = parsed;

    return 0;
}

/// Parse the integer gain in an OCAM "Gain set to N " response.
/** A trailing space before the final response whitespace is required by the legacy response contract.
 * The supported range is 1 through 600.
 *
 * \returns 0 on success, and emGain set to a value >= 1
 * \returns -1 on error, and emGain will be set to 0.
 *
 * \ingroup ocam2KCtrl
 */
inline int parseEMGain( unsigned          &emGain, /**< [out] Valid gain, or 0 on error. */
                        const std::string &fstr /**< [in] Device response to parse. */ )
{
    std::vector<std::string> v;
    mx::ioutils::parseStringVector( v, fstr, " " );

    emGain = 0;

    if( v.size() != 5 || v[0] != "Gain" || v[1] != "set" || v[2] != "to" || !ocamUtilsDetail::trim( v[4] ).empty() ||
        v[3].empty() )
    {
        return -1;
    }

    unsigned parsed;
    auto     result = std::from_chars( v[3].data(), v[3].data() + v[3].size(), parsed );

    if( result.ec != std::errc{} || result.ptr != v[3].data() + v[3].size() || parsed < 1 || parsed > 600 )
    {
        return -1;
    }

    emGain = parsed;
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // ocamUtils_hpp
