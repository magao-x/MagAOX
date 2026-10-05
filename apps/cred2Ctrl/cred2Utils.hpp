/** \file cred2Utils.hpp
 * \brief Utilities for the C-RED 2 camera controller.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup cred2Ctrl_files
 */

#ifndef cred2Utils_hpp
#define cred2Utils_hpp

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <mx/ioutils/stringUtils.hpp>

namespace MagAOX
{
namespace app
{

/// Structure holding the temperature values reported by the C-RED 2.
/**
 * \ingroup cred2Ctrl
 */
struct cred2Temps
{
    float motherboard{ 0 }; ///< Motherboard temperature [C].
    float frontend{ 0 };    ///< Front-end temperature [C].
    float powerboard{ 0 };  ///< Power-board temperature [C].
    float snake{ 0 };       ///< Detector temperature [C].
    float setpoint{ 0 };    ///< Detector temperature setpoint [C].
    float peltier{ 0 };     ///< External TEC temperature [C].
    float heatsink{ 0 };    ///< Heatsink temperature [C].

    /// Compare two cached temperature sets.
    bool operator==( const cred2Temps &t /**< [in] the values to compare against */ ) const
    {
        return motherboard == t.motherboard && frontend == t.frontend && powerboard == t.powerboard &&
               snake == t.snake && setpoint == t.setpoint && peltier == t.peltier && heatsink == t.heatsink;
    }

    /// Mark all temperature values invalid.
    int setInvalid()
    {
        motherboard = -999;
        frontend    = -999;
        powerboard  = -999;
        snake       = -999;
        setpoint    = -999;
        peltier     = -999;
        heatsink    = -999;

        return 0;
    }
};

/// C-RED 2 ROI expressed as 0-based inclusive column and row limits.
/**
 * \ingroup cred2Ctrl
 */
struct cred2Roi
{
    int  startColumn{ 0 };  ///< First included column.
    int  endColumn{ 0 };    ///< Last included column.
    int  startRow{ 0 };     ///< First included row.
    int  endRow{ 0 };       ///< Last included row.
    bool fullFrame{ true }; ///< True when the ROI spans the full detector.
};

/// Strip an optional prompt and surrounding whitespace from a C-RED 2 CLI response.
inline std::string cred2CleanResponse( const std::string &response /**< [in] raw CLI response */ )
{
    std::string clean = response;

    size_t promptPos = clean.find( "fli-cli>" );
    if( promptPos != std::string::npos )
    {
        clean.erase( promptPos );
    }

    size_t first = 0;
    while( first < clean.size() && std::isspace( static_cast<unsigned char>( clean[first] ) ) )
    {
        ++first;
    }

    size_t last = clean.size();
    while( last > first && std::isspace( static_cast<unsigned char>( clean[last - 1] ) ) )
    {
        --last;
    }

    return clean.substr( first, last - first );
}

/// Parse a raw numeric response into a float.
inline int cred2ParseFloat( float             &value,   ///< [out] parsed floating-point value
                            const std::string &response /**< [in] raw or cleaned CLI response */
)
{
    std::string clean = cred2CleanResponse( response );
    if( clean.empty() )
    {
        return -1;
    }

    char *end = nullptr;
    value     = std::strtof( clean.c_str(), &end );

    if( end == clean.c_str() )
    {
        return -1;
    }

    while( end != nullptr && *end != '\0' && std::isspace( static_cast<unsigned char>( *end ) ) )
    {
        ++end;
    }

    if( end != nullptr && *end != '\0' )
    {
        return -1;
    }

    return 0;
}

/// Parse a delimited list of raw numeric responses into a float vector.
inline int cred2ParseFloatVector( std::vector<float> &values,        ///< [out] parsed floating-point values
                                  const std::string  &response,      /**< [in] raw or cleaned CLI response */
                                  size_t              expectedValues /**< [in] expected number of parsed values, or 0 */
)
{
    std::string              clean = cred2CleanResponse( response );
    std::vector<std::string> tokens;

    mx::ioutils::parseStringVector( tokens, clean, ":, \t\r\n" );

    if( tokens.empty() )
    {
        return -1;
    }

    if( expectedValues > 0 && tokens.size() != expectedValues )
    {
        return -1;
    }

    values.clear();
    values.reserve( tokens.size() );

    for( const auto &token : tokens )
    {
        float value = 0;

        if( cred2ParseFloat( value, token ) < 0 )
        {
            values.clear();
            return -1;
        }

        values.push_back( value );
    }

    return 0;
}

/// Parse a raw on/off response into a boolean.
inline int cred2ParseBool( bool              &value,   ///< [out] parsed boolean value
                           const std::string &response /**< [in] raw or cleaned CLI response */
)
{
    std::string clean = cred2CleanResponse( response );
    std::transform( clean.begin(),
                    clean.end(),
                    clean.begin(),
                    []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

    if( clean == "on" || clean == "true" || clean == "1" )
    {
        value = true;
        return 0;
    }

    if( clean == "off" || clean == "false" || clean == "0" )
    {
        value = false;
        return 0;
    }

    return -1;
}

/// Parse a raw range response such as `0-639`.
inline int cred2ParseRange( int               &firstValue,  ///< [out] first parsed range value
                            int               &secondValue, ///< [out] second parsed range value
                            const std::string &response     /**< [in] raw or cleaned CLI response */
);

/// Parse a raw cropping status response such as `on` or `on:192-447:128-383`.
inline int cred2ParseCropState( bool              &enabled,     ///< [out] parsed cropping-enabled flag
                                int               &startColumn, ///< [out] parsed first included column
                                int               &endColumn,   ///< [out] parsed last included column
                                int               &startRow,    ///< [out] parsed first included row
                                int               &endRow,      ///< [out] parsed last included row
                                const std::string &response     /**< [in] raw or cleaned CLI response */
)
{
    std::string              clean = cred2CleanResponse( response );
    std::vector<std::string> tokens;

    mx::ioutils::parseStringVector( tokens, clean, ":" );

    if( tokens.empty() )
    {
        return -1;
    }

    if( cred2ParseBool( enabled, tokens[0] ) < 0 )
    {
        return -1;
    }

    if( tokens.size() == 1 )
    {
        startColumn = 0;
        endColumn   = 0;
        startRow    = 0;
        endRow      = 0;
        return 0;
    }

    if( tokens.size() != 3 )
    {
        return -1;
    }

    if( cred2ParseRange( startColumn, endColumn, tokens[1] ) < 0 || cred2ParseRange( startRow, endRow, tokens[2] ) < 0 )
    {
        return -1;
    }

    return 0;
}

/// Parse a raw range response such as `0-639`.
inline int cred2ParseRange( int               &firstValue,  ///< [out] first parsed range value
                            int               &secondValue, ///< [out] second parsed range value
                            const std::string &response     /**< [in] raw or cleaned CLI response */
)
{
    std::string              clean = cred2CleanResponse( response );
    std::vector<std::string> tokens;
    mx::ioutils::parseStringVector( tokens, clean, "-, \t\r\n" );

    if( tokens.size() != 2 )
    {
        return -1;
    }

    try
    {
        firstValue  = mx::ioutils::stoT<int>( tokens[0] );
        secondValue = mx::ioutils::stoT<int>( tokens[1] );
    }
    catch( ... )
    {
        return -1;
    }

    return 0;
}

/// Check whether a command response looks successful.
inline bool cred2ResponseOK( const std::string &response /**< [in] raw or cleaned CLI response */ )
{
    std::string clean = cred2CleanResponse( response );
    std::string lower = clean;
    std::transform( lower.begin(),
                    lower.end(),
                    lower.begin(),
                    []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

    if( lower.find( "error" ) != std::string::npos || lower.find( "fail" ) != std::string::npos )
    {
        return false;
    }

    return true;
}

/// Convert a MagAO-X ROI center/size description into C-RED 2 corners.
inline int cred2RoiFromCenter( cred2Roi &roi,       ///< [out] the corresponding C-RED 2 ROI
                               float     centerX,   /**< [in] ROI x center coordinate */
                               float     centerY,   /**< [in] ROI y center coordinate */
                               int       width,     /**< [in] ROI width in pixels */
                               int       height,    /**< [in] ROI height in pixels */
                               int       fullWidth, /**< [in] detector full-frame width */
                               int       fullHeight /**< [in] detector full-frame height */
)
{
    if( width < 1 || height < 1 || fullWidth < 1 || fullHeight < 1 )
    {
        return -1;
    }

    roi.startColumn = static_cast<int>( std::lround( centerX - 0.5f * ( static_cast<float>( width ) - 1.0f ) ) );
    roi.endColumn   = roi.startColumn + width - 1;
    roi.startRow    = static_cast<int>( std::lround( centerY - 0.5f * ( static_cast<float>( height ) - 1.0f ) ) );
    roi.endRow      = roi.startRow + height - 1;

    if( roi.startColumn < 0 || roi.startRow < 0 || roi.endColumn >= fullWidth || roi.endRow >= fullHeight )
    {
        return -1;
    }

    roi.fullFrame =
        roi.startColumn == 0 && roi.endColumn == fullWidth - 1 && roi.startRow == 0 && roi.endRow == fullHeight - 1;

    return 0;
}

/// Convert C-RED 2 ROI corners into a MagAO-X ROI center/size description.
inline int cred2RoiToCenter( float          &centerX,   ///< [out] ROI x center coordinate
                             float          &centerY,   ///< [out] ROI y center coordinate
                             int            &width,     ///< [out] ROI width in pixels
                             int            &height,    ///< [out] ROI height in pixels
                             const cred2Roi &roi,       /**< [in] the C-RED 2 ROI to convert */
                             int             fullWidth, /**< [in] detector full-frame width */
                             int             fullHeight /**< [in] detector full-frame height */
)
{
    if( roi.startColumn < 0 || roi.startRow < 0 || roi.endColumn < roi.startColumn || roi.endRow < roi.startRow ||
        roi.endColumn >= fullWidth || roi.endRow >= fullHeight )
    {
        return -1;
    }

    width  = roi.endColumn - roi.startColumn + 1;
    height = roi.endRow - roi.startRow + 1;

    centerX = roi.startColumn + 0.5f * ( static_cast<float>( width ) - 1.0f );
    centerY = roi.startRow + 0.5f * ( static_cast<float>( height ) - 1.0f );

    return 0;
}

/// Format the column command payload for `set cropping columns`.
inline std::string cred2ColumnsSpec( const cred2Roi &roi /**< [in] the ROI to format */ )
{
    return std::to_string( roi.startColumn ) + "-" + std::to_string( roi.endColumn );
}

/// Format the row command payload for `set cropping rows`.
inline std::string cred2RowsSpec( const cred2Roi &roi /**< [in] the ROI to format */ )
{
    return std::to_string( roi.startRow ) + "-" + std::to_string( roi.endRow );
}

} // namespace app
} // namespace MagAOX

#endif // cred2Utils_hpp
