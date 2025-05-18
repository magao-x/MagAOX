/** \file Types.hpp
 *
 * Declarations for the INDI Types.
 *
 * @author Jared Males (@Steward Observatory)
 */

#ifndef libxindi_Types_hpp
#define libxindi_Types_hpp

#include <string>
#include <complex>
#include <format>

namespace xindi
{

/// \name Types
/** @{
 *
 */

/// The INDI Types
/**
 */
enum class Type
{
    Unknown = 0, ///< Type is not known, generally indicates an error
    Number,      ///< The INDI Number type
    Text,        ///< The INDI Text
    Switch,      ///< The INDI Switch type
    Light,       ///< The INDI Light type
    BLOB         ///< The binary large object (BLOB) type
};

enum class Switch
{
    Unknown = 0,
    Off,
    On
};

enum class Light
{
    Unknown = 0,
    Idle,
    Ok,
    Busy,
    Alert
};

struct BLOB
{
};

template <typename TT>
Type type2Type()
{
    return Type::Number;
}

template <>
Type type2Type<std::string>()
{
    return Type::Text;
}

template <>
Type type2Type<char *>()
{
    return Type::Text;
}

template <>
Type type2Type<const char *>()
{
    return Type::Text;
}

template <>
Type type2Type<Switch>()
{
    return Type::Switch;
}

template <>
Type type2Type<Light>()
{
    return Type::Light;
}

template <>
Type type2Type<BLOB>()
{
    return Type::BLOB;
}

///@}

/// \name Type to String Conversions
/**@{
 */

/** \todo long double? */
/** \todo quad? */

/// Get the string representation of the value of a general type.
/** Specializations for strings, floating points, and Switches and Lights are provided.
 *
 * \returns the string corresponding to the value
 */
template <typename TT>
std::string value2string( const TT &value /**< [in] the value to convert */ )
{
    return std::format( "{}", value );
}

/// Get the string from a string
/** This just returns the value without conversion
 *
 * \returns the string
 */
template<>
inline std::string value2string<std::string>( const std::string &val /**< [in] the string to return */ )
{
    return val;
}

/// Get the string representing a float without losing precision
/**
 *
 * \returns the string containing the decimal representation with full precision
 */
template<>
inline std::string value2string<float>( const float &value /**< [in] the value to convert */ )
{
    return std::format( "{:.{}f}", value, std::numeric_limits<float>::digits10 + 1 );
}

/// Get the string representing a double without losing precision
/**
 *
 * \returns the string containing the decimal representation with full precision
 */
template<>
inline std::string value2string<double>( const double &value /**< [in] the value to convert */ )
{
    return std::format( "{:.{}f}", value, std::numeric_limits<double>::digits10 + 1 );
}

/// Get the string representing a complex<float> without losing precision
/** Uses the format a+bi or a-bi.
 *
 * \returns the string containing the decimal representation with full precision
 */
template<>
inline std::string value2string<std::complex<float>>( const std::complex<float> &value /**< [in] the value to convert */ )
{
    return std::format( "{:.{}f}{:+.{}f}i",
                        value.real(),
                        std::numeric_limits<float>::digits10 + 1,
                        value.imag(),
                        std::numeric_limits<float>::digits10 + 1 );
}

/// Get the string representing a complex<double> without losing precision
/** Uses the format a+bi or a-bi.
 *
 * \returns the string containing the decimal representation with full precision
 */
template<>
inline std::string value2string<std::complex<double>>( const std::complex<double> &value /**< [in] the value to convert */ )
{
    return std::format( "{:.{}f}{:+.{}f}i",
                        value.real(),
                        std::numeric_limits<double>::digits10 + 1,
                        value.imag(),
                        std::numeric_limits<double>::digits10 + 1 );
}

/// Get the string representation of a switch state.
/** A switch can be "On" or "Off".  If in the unknown state the empty string "" is returned
 *
 * \returns the string corresponding to the switch state
 */
template<>
inline std::string value2string<Switch>( const Switch &sw /**< [in] the switch state to convert*/ )
{
    std::string str = "";

    switch( sw )
    {
    case Switch::Unknown:
        str = "";
        break;
    case Switch::Off:
        str = "Off";
        break;
    case Switch::On:
        str = "On";
        break;
    }

    return str;
}

/// Get the string representation of a light state.
/** A light can be "Idle" or "Ok", "Busy", or "Alert".  If in the unknown state the empty string "" is returned
 *
 * \returns the string corresponding to the light state
 */
template<>
inline std::string value2string<Light>( const Light &light /**< [in] the light state to convert*/ )
{
    std::string str = "";

    switch( light )
    {
    case Light::Unknown:
        str = "";
        break;
    case Light::Idle:
        str = "Idle";
        break;
    case Light::Ok:
        str = "Ok";
        break;
    case Light::Busy:
        str = "Busy";
        break;
    case Light::Alert:
        str = "Alert";
        break;
    }

    return str;
}

///@}

/// \name String to Type Conversions
/**@{
 */

template <typename T>
T string2value( const std::string &str )
{
    std::stringstream ss( str );

    T val;

    ss >> std::boolalpha >> val;

    return val;
}

template <>
inline std::string string2value<std::string>( const std::string &str )
{
    return str;
}

template <>
inline std::complex<float> string2value<std::complex<float>>( const std::string &str )
{
    std::stringstream ss( str );

    std::complex<float> val;

    ss >> reinterpret_cast<float ( & )[2]>( val )[0] >> reinterpret_cast<float ( & )[2]>( val )[1];

    return val;
}

template <>
inline std::complex<double> string2value<std::complex<double>>( const std::string &str )
{
    std::stringstream ss( str );

    std::complex<double> val;

    ss >> reinterpret_cast<double ( & )[2]>( val )[0] >> reinterpret_cast<double ( & )[2]>( val )[1];

    return val;
}

/// Get the switch state from its string representation
/** The valid states are "On" and "Off".  Anything else returns `Switch::Unknown`
 *
 * \returns the \ref Switch state corresponding to a string
 */
template <>
inline Switch string2value<Switch>( const std::string &str /**< [in] a string containing a switch state */ )
{
    Switch sw = Switch::Unknown;

    if( str == "Off" )
    {
        sw = Switch::Off;
    }
    else if( str == "On" )
    {
        sw = Switch::On;
    }

    return sw;
}

/// Get the light state from its string representation
/** The valid states are "Idle", "Ok", "Busy", and "Alert".  Anything else returns `Light::Unknown`
 *
 * \returns the \ref Switch state corresponding to a string
 */
template <>
inline Light string2value<Light>( const std::string &str )
{
    Light light = Light::Unknown;

    if( str == "Idle" )
    {
        light = Light::Idle;
    }
    else if( str == "Ok" )
    {
        light = Light::Ok;
    }
    else if( str == "Busy" )
    {
        light = Light::Busy;
    }
    else if( str == "Alert" )
    {
        light = Light::Alert;
    }

    return light;
}

///@}

} // namespace xindi

#endif // libxindi_Types_hpp
