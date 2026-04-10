/** \file ocamUtils.hpp
 * \brief Utilities for the OCAM camera
 *
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamUtils_hpp
#define ocamUtils_hpp

#include <mx/ioutils/stringUtils.hpp>

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
    float CCD{ 0 };           ///< The detector temperature.
    float CPU{ 0 };           ///< The CPU temperature
    float POWER{ 0 };         ///< Power supply temperature
    float BIAS{ 0 };          ///< Bias temperature
    float WATER{ 0 };         ///< Cooling water temperature
    float LEFT{ 0 };          ///< The left amplifier temperature
    float RIGHT{ 0 };         ///< The right amplifier temperature
    float SET{ 0 };           ///< The CCD set temeperature
    float COOLING_POWER{ 0 }; ///< the cooling power in 100 mw.

    /// Test for equality between two ocamTemps structures
    /**
     * \returns true if all members are equal
     * \returns false otherwise
     */
    bool operator==( const ocamTemps &t /**< [in] the struct to compare to*/ ) const
    {
        return ( CCD == t.CCD && POWER == t.POWER && BIAS == t.BIAS && WATER == t.WATER && LEFT == t.LEFT &&
                 RIGHT == t.RIGHT && SET == t.SET && COOLING_POWER == t.COOLING_POWER );
    }

    /// Set all values to the invalid value, -999.
    int setInvalid()
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
};

/// Parse the OCAM temp query and fill the ocamTemps structure.
/**
 * \b Tests
 *     - Parsing the temp response \ref tests_MagAOX_app_ocamUtils_parse_temp_response "[test doc]"
 *
 * \returns 0 on success
 * \returns -1 on error
 *
 * \ingroup ocam2KCtrl
 */
int parseTemps( ocamTemps         &temps, ///< [out] the struture of temperature readings
                const std::string &tstr   ///< [in] the device response to parse.
)
{
    std::vector<std::string> v;
    mx::ioutils::parseStringVector( v, tstr, "[]" );

    if( v.size() < 18 )
        return -1;

    temps.CCD           = mx::ioutils::stoT<float>( v[1] );
    temps.CPU           = mx::ioutils::stoT<float>( v[3] );
    temps.POWER         = mx::ioutils::stoT<float>( v[5] );
    temps.BIAS          = mx::ioutils::stoT<float>( v[7] );
    temps.WATER         = mx::ioutils::stoT<float>( v[9] );
    temps.LEFT          = mx::ioutils::stoT<float>( v[11] );
    temps.RIGHT         = mx::ioutils::stoT<float>( v[13] );
    temps.SET           = mx::ioutils::stoT<float>( v[15] ) / 10.0;
    temps.COOLING_POWER = mx::ioutils::stoT<float>( v[17] );

    return 0;
}

/// Parse the FPS response
/** Parses the OCAM 2K response to the "fps" query.
 *
 * \returns 0 on success
 * \returns -1 on error
 *
 * \todo add test for FPS
 *
 * \ingroup ocam2KCtrl
 */
int parseFPS( float             &fps, ///< [out] the fps returned by the camera
              const std::string &fstr ///< [in] the response to parse
)
{
    std::vector<std::string> v;
    mx::ioutils::parseStringVector( v, fstr, "[]" );

    if( v.size() < 3 )
        return -1;

    fps = mx::ioutils::stoT<float>( v[1] );

    return 0;
}

/// Parse the EM gain response
/** Example response: "Gain set to 2 \n\n", with the trailing space.
 * Expects gain >=1 and <= 600, otherwise returns an error.
 *
 * \b Tests
 *     - Parsing the gain response \ref tests_MagAOX_app_ocamUtils_parse_gain_response "[test doc]"
 *
 * \returns 0 on success, and emGain set to a value >= 1
 * \returns -1 on error, and emGain will be set to 0.
 *
 * \ingroup ocam2KCtrl
 */
int parseEMGain( unsigned          &emGain, ///< [out] the value of gain returned by the camera
                 const std::string &fstr    ///< [in] the query response from the camera.
)
{
    std::vector<std::string> v;
    mx::ioutils::parseStringVector( v, fstr, " " );

    if( v.size() != 5 )
    {
        emGain = 0;
        return -1;
    }

    emGain = mx::ioutils::stoT<unsigned>( v[3] );

    if( emGain < 1 || emGain > 600 )
    {
        emGain = 0;
        return -1;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // ocamUtils_hpp
