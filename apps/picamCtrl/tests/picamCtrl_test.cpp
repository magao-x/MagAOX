/** \file picamCtrl_test.cpp
 * \brief Offline regression tests for picamCtrl temperature limits and request rejection.
 * \ingroup picamCtrl_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"
#include <picam_advanced.h>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace picamHarness
{
/// Scripted camera constraints and observable SDK calls.
struct Fake
{
    /// Required range returned by the SDK.
    PicamRangeConstraint m_range{};

    /// Result of querying the required range.
    PicamError m_rangeError{ PicamError_None };

    /// Result of checking a candidate temperature.
    PicamError m_validationError{ PicamError_None };

    /// Whether the SDK accepts a candidate, including constraints beyond its extrema.
    pibln m_settable{ true };

    /// All SDK calls, used to detect writes on invalid reconfiguration.
    unsigned m_calls{ 0 };

    /// Range allocations made by the SDK double.
    unsigned m_rangeGets{ 0 };

    /// Range allocations returned to the SDK double.
    unsigned m_rangeDestroys{ 0 };

    /// Candidate validation calls.
    unsigned m_validations{ 0 };

    /// Parameter write calls.
    unsigned m_writes{ 0 };
};

/// Current test's offline camera state.
inline Fake g_fake;
} // namespace picamHarness

#define MagAOXApp outletTestApp
#define telemeter outletTestTelemeter
#define protected public
#include "../picamCtrl.hpp"
#undef protected
#undef telemeter
#undef MagAOXApp

namespace picamHarness
{
/// Production app with isolated INDI transport and exposed temperature state.
struct Fixture : outletHarness::Controller<MagAOX::app::picamCtrl>
{
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_ccdTempSetpt;
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_minTemp;
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_maxTemp;
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_indiP_temp;
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_startupTemp;
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_configMinTemp;
    using MagAOX::app::dev::stdCamera<MagAOX::app::picamCtrl>::m_configMaxTemp;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::picamCtrl>::m_reconfig;
    using MagAOX::app::MagAOXApp<true>::m_powerState;
    using MagAOX::app::MagAOXApp<true>::m_powerTargetState;

    /// Initialize camera state and a real private INDI driver without starting acquisition.
    Fixture();

    /// Send a temperature request through the production stdCamera callback.
    int request( float target /**< [in] Requested temperature, in C. */ );
};

Fixture::Fixture()
{
    g_fake                  = {};
    g_fake.m_range.minimum  = -55;
    g_fake.m_range.maximum  = 20;
    outletHarness::g_faults = {};
    m_serialNumber          = "fake";
    m_powerState            = 1;
    m_powerTargetState      = 1;
    powerOnDefaults();
    driver();
    createStandardIndiNumber<float>( m_indiP_temp, "temp_ccd", m_minTemp, m_maxTemp, 0, "%0.1f" );
    REQUIRE( powerOnTemp() == 0 );
}

int Fixture::request( float target )
{
    pcf::IndiProperty received( pcf::IndiProperty::Number, m_indiP_temp.getDevice(), m_indiP_temp.getName() );
    received.add( pcf::IndiElement( "target" ) );
    received["target"].set( target );
    return newCallBack_temp( received );
}
} // namespace picamHarness
/// \endcond

namespace libXWCTest
{
/** \addtogroup picamCtrl_unit_test */
namespace picamCtrlTest
{
using namespace picamHarness;
using namespace MagAOX::app;

/// Configured limits default to inclusive -55 and 20 C and reject incompatible startup settings.
/** \ingroup picamCtrl_unit_test */
TEST_CASE( "picamCtrl configures user temperature limits", "[picamCtrl]" )
{
    // clang-format off
    #ifdef PICAMCTRL_TEST_DOXYGEN_REF
    picamCtrl::picamCtrl(); picamCtrl::setupConfig(); picamCtrl::loadConfig();
    picamCtrl::powerOnDefaults(); dev::stdCamera<picamCtrl>::powerOnTemp();
    #endif
    // clang-format on
    Fixture f;
    REQUIRE( f.m_minTemp == -55 );
    REQUIRE( f.m_maxTemp == 20 );

    SECTION( "defaults" )
    {
        f.configText( "" );
        f.loadConfig();
        REQUIRE_FALSE( f.m_shutdown );
        REQUIRE( f.m_minTemp == -55 );
        REQUIRE( f.m_maxTemp == 20 );
    }
    SECTION( "configured bounds" )
    {
        f.configText( "[camera]\nminTemp=-40\nmaxTemp=10\nstartupTemp=-30\n" );
        f.loadConfig();
        REQUIRE_FALSE( f.m_shutdown );
        REQUIRE( f.m_minTemp == -40 );
        REQUIRE( f.m_maxTemp == 10 );
        REQUIRE( f.m_startupTemp == -30 );
    }
    SECTION( "equal inclusive bounds" )
    {
        f.configText( "[camera]\nminTemp=-30\nmaxTemp=-30\nstartupTemp=-30\n" );
        f.loadConfig();
        REQUIRE_FALSE( f.m_shutdown );
    }
    SECTION( "reversed bounds" )
    {
        f.configText( "[camera]\nminTemp=10\nmaxTemp=-40\n" );
        f.loadConfig();
        REQUIRE( f.m_shutdown );
    }
    SECTION( "startup target outside user limits" )
    {
        f.configText( "[camera]\nstartupTemp=21\n" );
        f.loadConfig();
        REQUIRE( f.m_shutdown );
    }
    SECTION( "default startup target outside user limits" )
    {
        f.configText( "[camera]\nminTemp=-40\n" );
        f.loadConfig();
        REQUIRE( f.m_shutdown );
    }
    SECTION( "nonfinite bounds" )
    {
        f.configText( "" );
        f.m_configMaxTemp = std::numeric_limits<float>::infinity();
        f.loadConfig();
        REQUIRE( f.m_shutdown );
    }
}

/// The configured startup target replaces the old fixed power-on value on every power cycle.
/** \ingroup picamCtrl_unit_test */
TEST_CASE( "picamCtrl reuses the configured power-on temperature", "[picamCtrl]" )
{
    // clang-format off
    #ifdef PICAMCTRL_TEST_DOXYGEN_REF
    picamCtrl::loadConfigImpl(); picamCtrl::powerOnDefaults();
    dev::stdCamera<picamCtrl>::powerOnTemp();
    #endif
    // clang-format on
    Fixture f;
    f.configText( "[camera]\nstartupTemp=-30\n" );
    REQUIRE( f.loadConfigImpl() == 0 );
    for( int cycle = 0; cycle < 2; ++cycle )
    {
        f.m_ccdTempSetpt = 10;
        REQUIRE( f.powerOnDefaults() == 0 );
        REQUIRE( f.powerOnTemp() == 0 );
        REQUIRE( f.m_ccdTempSetpt == -30 );
        REQUIRE( f.m_indiP_temp["target"].get<float>() == -30 );
    }
}

/// SDK ranges narrow user limits and are recomputed from configuration on reconnect.
/** \ingroup picamCtrl_unit_test */
TEST_CASE( "picamCtrl intersects user and SDK temperature ranges", "[picamCtrl]" )
{
    // clang-format off
    #ifdef PICAMCTRL_TEST_DOXYGEN_REF
    picamCtrl::getTempRange();
    #endif
    // clang-format on
    Fixture f;
    f.m_cameraHandle       = reinterpret_cast<PicamHandle>( 1 );
    g_fake.m_range.minimum = -80;
    g_fake.m_range.maximum = 30;
    REQUIRE( f.getTempRange() == 0 );
    REQUIRE( f.m_minTemp == -55 );
    REQUIRE( f.m_maxTemp == 20 );

    g_fake.m_range.minimum = -40;
    g_fake.m_range.maximum = 10;
    REQUIRE( f.getTempRange() == 0 );
    REQUIRE( f.m_minTemp == -40 );
    REQUIRE( f.m_maxTemp == 10 );
    REQUIRE( f.m_indiP_temp["target"].getMin() == "-40" );
    REQUIRE( f.m_indiP_temp["target"].getMax() == "10" );
    REQUIRE( f.m_indiP_temp["current"].getMin() == "-40" );
    REQUIRE( f.m_indiP_temp["current"].getMax() == "10" );

    g_fake.m_range.minimum = -80;
    g_fake.m_range.maximum = 30;
    REQUIRE( f.getTempRange() == 0 );
    REQUIRE( f.m_minTemp == -55 );
    REQUIRE( f.m_maxTemp == 20 );
    REQUIRE( g_fake.m_rangeGets == g_fake.m_rangeDestroys );
    auto messages = f.messages();
    REQUIRE( messages.size() >= 3 );

    SECTION( "no overlap" )
    {
        g_fake.m_range.minimum = 21;
        REQUIRE( f.getTempRange() == -1 );
    }
    SECTION( "one common endpoint" )
    {
        g_fake.m_range.minimum = 20;
        REQUIRE( f.getTempRange() == 0 );
        REQUIRE( f.m_minTemp == 20 );
        REQUIRE( f.m_maxTemp == 20 );
    }
    SECTION( "empty SDK range" )
    {
        g_fake.m_range.empty_set = true;
        REQUIRE( f.getTempRange() == -1 );
    }
    SECTION( "nonfinite SDK range" )
    {
        g_fake.m_range.minimum = std::numeric_limits<piflt>::quiet_NaN();
        REQUIRE( f.getTempRange() == -1 );
    }
    SECTION( "query failure" )
    {
        g_fake.m_rangeError = PicamError_InvalidParameterValue;
        REQUIRE( f.getTempRange() == -1 );
    }
    SECTION( "inward rounding" )
    {
        g_fake.m_range.minimum = -40.123456789;
        g_fake.m_range.maximum = 10.123456789;
        REQUIRE( f.getTempRange() == 0 );
        REQUIRE( piflt( f.m_minTemp ) >= g_fake.m_range.minimum );
        REQUIRE( piflt( f.m_maxTemp ) <= g_fake.m_range.maximum );
    }
    SECTION( "custom user bounds remain authoritative" )
    {
        f.m_configMinTemp      = -70;
        f.m_configMaxTemp      = 5;
        g_fake.m_range.minimum = -80;
        g_fake.m_range.maximum = 30;
        REQUIRE( f.getTempRange() == 0 );
        REQUIRE( f.m_minTemp == -70 );
        REQUIRE( f.m_maxTemp == 5 );
        g_fake.m_range.minimum = -40;
        REQUIRE( f.getTempRange() == 0 );
        REQUIRE( f.m_minTemp == -40 );
        REQUIRE( f.m_maxTemp == 5 );
    }
}

/// Rejected callback targets restore accepted state without interrupting acquisition or a pending valid request.
/** \ingroup picamCtrl_unit_test */
TEST_CASE( "picamCtrl rejects invalid temperature requests without reconfiguration", "[picamCtrl]" )
{
    // clang-format off
    #ifdef PICAMCTRL_TEST_DOXYGEN_REF
    picamCtrl::setTempSetPt(); picamCtrl::validateTempSetPt();
    dev::stdCamera<picamCtrl>::newCallBack_temp();
    #endif
    // clang-format on
    Fixture f;
    f.m_cameraHandle = reinterpret_cast<PicamHandle>( 1 );
    f.state( stateCodes::OPERATING );
    REQUIRE( f.request( -55 ) == 0 );
    REQUIRE( f.m_reconfig );
    REQUIRE( f.request( 20 ) == 0 );
    REQUIRE( f.m_ccdTempSetpt == 20 );
    f.m_reconfig = false;

    SECTION( "below minimum" )
    {
        REQUIRE( f.request( -56 ) == -1 );
    }
    SECTION( "above maximum" )
    {
        REQUIRE( f.request( 21 ) == -1 );
    }
    SECTION( "SDK rejects an in-range value" )
    {
        g_fake.m_settable = false;
        REQUIRE( f.request( -30 ) == -1 );
    }
    SECTION( "SDK validation fails" )
    {
        g_fake.m_validationError = PicamError_InvalidParameterValue;
        REQUIRE( f.request( -30 ) == -1 );
    }
    SECTION( "nonfinite internal values" )
    {
        unsigned validations = g_fake.m_validations;
        for( float value : { std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity() } )
            REQUIRE( f.validateTempSetPt( value ) == -1 );
        REQUIRE( g_fake.m_validations == validations );
        REQUIRE_FALSE( f.m_reconfig );
        REQUIRE( f.m_ccdTempSetpt == 20 );
        return;
    }
    SECTION( "preserve an already pending valid request" )
    {
        REQUIRE( f.request( -30 ) == 0 );
        REQUIRE( f.request( 21 ) == -1 );
        REQUIRE( f.m_reconfig );
        REQUIRE( f.m_ccdTempSetpt == -30 );
        REQUIRE( f.m_indiP_temp["target"].get<float>() == -30 );
        return;
    }
    REQUIRE_FALSE( f.m_reconfig );
    REQUIRE( f.m_ccdTempSetpt == 20 );
    REQUIRE( f.m_indiP_temp["target"].get<float>() == 20 );
    REQUIRE( f.m_indiP_temp.getState() == INDI_ALERT );
    REQUIRE( f.state() == stateCodes::OPERATING );
    REQUIRE( g_fake.m_writes == 0 );
}

/// Connection rejects incompatible startup targets and configuration rejects invalid values before SDK writes.
/** \ingroup picamCtrl_unit_test */
TEST_CASE( "picamCtrl validates temperature before connection and acquisition", "[picamCtrl]" )
{
    // clang-format off
    #ifdef PICAMCTRL_TEST_DOXYGEN_REF
    picamCtrl::connect(); picamCtrl::configureAcquisition();
    #endif
    // clang-format on
    Fixture f;
    SECTION( "compatible startup" )
    {
        REQUIRE( f.connect() == 0 );
        REQUIRE( f.state() == stateCodes::CONNECTED );
        REQUIRE( g_fake.m_writes == 0 );
    }
    SECTION( "SDK excludes startup target" )
    {
        g_fake.m_range.minimum = -40;
        REQUIRE( f.connect() == -1 );
        REQUIRE( f.state() == stateCodes::ERROR );
        REQUIRE( g_fake.m_writes == 0 );
    }
    SECTION( "invalid acquisition target" )
    {
        f.m_cameraHandle     = reinterpret_cast<PicamHandle>( 1 );
        f.m_ccdTempSetpt     = 21;
        f.m_camera_timestamp = 5;
        REQUIRE( f.configureAcquisition() == -1 );
        REQUIRE( g_fake.m_calls == 0 );
        REQUIRE( f.m_camera_timestamp == 5 );
    }
}
} // namespace picamCtrlTest
} // namespace libXWCTest

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
extern "C"
{
    /// Offline SDK double.
    PicamError PicamAdvanced_GetCameraModel( PicamHandle camera, PicamHandle *model )
    {
        static_cast<void>( camera );
        static_cast<void>( model );
        ++picamHarness::g_fake.m_calls;
        *model = reinterpret_cast<PicamHandle>( 2 );
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError PicamAdvanced_GetParameterRangeConstraints( PicamHandle                  camera_or_accessory,
                                                           PicamParameter               parameter,
                                                           const PicamRangeConstraint **constraint_array,
                                                           piint                       *constraint_count )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( constraint_array );
        static_cast<void>( constraint_count );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError PicamAdvanced_OpenCameraDevice( const PicamCameraID *id, PicamHandle *device )
    {
        static_cast<void>( id );
        static_cast<void>( device );
        ++picamHarness::g_fake.m_calls;
        *device = reinterpret_cast<PicamHandle>( 1 );
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError PicamAdvanced_SetAcquisitionBuffer( PicamHandle device, const PicamAcquisitionBuffer *buffer )
    {
        static_cast<void>( device );
        static_cast<void>( buffer );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_CanReadParameter( PicamHandle camera_or_accessory, PicamParameter parameter, pibln *readable )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( readable );
        ++picamHarness::g_fake.m_calls;
        *readable = true;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_CanSetParameterFloatingPointValue( PicamHandle    camera_or_accessory,
                                                        PicamParameter parameter,
                                                        piflt          value,
                                                        pibln         *settable )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( value );
        static_cast<void>( settable );
        ++picamHarness::g_fake.m_calls;
        ++picamHarness::g_fake.m_validations;
        if( picamHarness::g_fake.m_validationError != PicamError_None )
            return picamHarness::g_fake.m_validationError;
        *settable = picamHarness::g_fake.m_settable;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError
    Picam_CanSetParameterOnline( PicamHandle camera_or_accessory, PicamParameter parameter, pibln *onlineable )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( onlineable );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_CloseCamera( PicamHandle camera )
    {
        static_cast<void>( camera );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_CommitParameters( PicamHandle            camera_or_accessory,
                                       const PicamParameter **failed_parameter_array,
                                       piint                 *failed_parameter_count )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( failed_parameter_array );
        static_cast<void>( failed_parameter_count );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_DestroyCameraIDs( const PicamCameraID *id_array )
    {
        static_cast<void>( id_array );
        ++picamHarness::g_fake.m_calls;
        delete[] id_array;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_DestroyParameters( const PicamParameter *parameter_array )
    {
        static_cast<void>( parameter_array );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_DestroyRangeConstraints( const PicamRangeConstraint *constraint_array )
    {
        static_cast<void>( constraint_array );
        ++picamHarness::g_fake.m_calls;
        ++picamHarness::g_fake.m_rangeDestroys;
        delete constraint_array;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_DestroyRois( const PicamRois *rois )
    {
        static_cast<void>( rois );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_DestroyString( const pichar *s )
    {
        static_cast<void>( s );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_DoesParameterExist( PicamHandle camera_or_accessory, PicamParameter parameter, pibln *exists )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( exists );
        ++picamHarness::g_fake.m_calls;
        *exists = true;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_GetAvailableCameraIDs( const PicamCameraID **id_array, piint *id_count )
    {
        static_cast<void>( id_array );
        static_cast<void>( id_count );
        ++picamHarness::g_fake.m_calls;
        *id_array = new PicamCameraID[1]{};
        std::strcpy( const_cast<PicamCameraID *>( *id_array )->serial_number, "fake" );
        *id_count = 1;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_GetEnumerationString( PicamEnumeratedType type, piint value, const pichar **s )
    {
        static_cast<void>( type );
        static_cast<void>( value );
        static_cast<void>( s );
        ++picamHarness::g_fake.m_calls;
        *s = "fake";
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError
    Picam_GetParameterFloatingPointValue( PicamHandle camera_or_accessory, PicamParameter parameter, piflt *value )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_GetParameterIntegerValue( PicamHandle camera_or_accessory, PicamParameter parameter, piint *value )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_GetParameterLargeIntegerValue( PicamHandle camera, PicamParameter parameter, pi64s *value )
    {
        static_cast<void>( camera );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_GetParameterRangeConstraint( PicamHandle                  camera_or_accessory,
                                                  PicamParameter               parameter,
                                                  PicamConstraintCategory      category,
                                                  const PicamRangeConstraint **constraint )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( category );
        static_cast<void>( constraint );
        ++picamHarness::g_fake.m_calls;
        ++picamHarness::g_fake.m_rangeGets;
        if( picamHarness::g_fake.m_rangeError != PicamError_None )
            return picamHarness::g_fake.m_rangeError;
        *constraint = new PicamRangeConstraint( picamHarness::g_fake.m_range );
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_GetParameterRoisValue( PicamHandle camera, PicamParameter parameter, const PicamRois **value )
    {
        static_cast<void>( camera );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_InitializeLibrary()
    {
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_IsAcquisitionRunning( PicamHandle camera, pibln *running )
    {
        static_cast<void>( camera );
        static_cast<void>( running );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError
    Picam_SetParameterFloatingPointValue( PicamHandle camera_or_accessory, PicamParameter parameter, piflt value )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        ++picamHarness::g_fake.m_writes;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_SetParameterFloatingPointValueOnline( PicamHandle camera, PicamParameter parameter, piflt value )
    {
        static_cast<void>( camera );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_SetParameterIntegerValue( PicamHandle camera_or_accessory, PicamParameter parameter, piint value )
    {
        static_cast<void>( camera_or_accessory );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_SetParameterIntegerValueOnline( PicamHandle camera, PicamParameter parameter, piint value )
    {
        static_cast<void>( camera );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_SetParameterLargeIntegerValue( PicamHandle camera, PicamParameter parameter, pi64s value )
    {
        static_cast<void>( camera );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_SetParameterRoisValue( PicamHandle camera, PicamParameter parameter, const PicamRois *value )
    {
        static_cast<void>( camera );
        static_cast<void>( parameter );
        static_cast<void>( value );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_StartAcquisition( PicamHandle camera )
    {
        static_cast<void>( camera );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_StopAcquisition( PicamHandle camera )
    {
        static_cast<void>( camera );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_UninitializeLibrary()
    {
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }

    /// Offline SDK double.
    PicamError Picam_WaitForAcquisitionUpdate( PicamHandle             camera,
                                               piint                   readout_time_out,
                                               PicamAvailableData     *available,
                                               PicamAcquisitionStatus *status )
    {
        static_cast<void>( camera );
        static_cast<void>( readout_time_out );
        static_cast<void>( available );
        static_cast<void>( status );
        ++picamHarness::g_fake.m_calls;
        return PicamError_None;
    }
}
/// \endcond
