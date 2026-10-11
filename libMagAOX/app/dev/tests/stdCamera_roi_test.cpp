/** \file stdCamera_roi_test.cpp
 * \brief Offline relative ROI, preset classification, and completed-check tests.
 * \ingroup stdCamera_unit_test
 */
#include "../../../../tests/testXWC.hpp"
#include "../../../../tests/outletAppTest.hpp"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace roiHarness
{
/// Threadless camera exercising the actual stdCamera callback dispatcher.
class Camera : public MagAOX::app::outletTestApp<>, public MagAOX::app::dev::stdCamera<Camera>
{
    friend class MagAOX::app::dev::stdCamera<Camera>;
    using stdCameraT = MagAOX::app::dev::stdCamera<Camera>; ///< Shared camera implementation.
  public:
    static constexpr bool c_stdCamera_tempControl     = false; ///< Expose temperature controls.
    static constexpr bool c_stdCamera_tempLimits      = false; ///< Enable configurable temperature limits.
    static constexpr bool c_stdCamera_temp            = false; ///< Expose temperature reporting.
    static constexpr bool c_stdCamera_readoutSpeed    = false; ///< Disable readout-speed control.
    static constexpr bool c_stdCamera_vShiftSpeed     = false; ///< Disable vertical-shift control.
    static constexpr bool c_stdCamera_fanSpeed        = false; ///< Disable fan control.
    static constexpr bool c_stdCamera_emGain          = false; ///< Disable EM-gain control.
    static constexpr bool c_stdCamera_exptimeCtrl     = false; ///< Disable exposure-time control.
    static constexpr bool c_stdCamera_fpsCtrl         = false; ///< Disable frame-rate control.
    static constexpr bool c_stdCamera_fps             = false; ///< Disable frame-rate reporting.
    static constexpr bool c_stdCamera_synchro         = false; ///< Disable synchronization control.
    static constexpr bool c_stdCamera_usesModes       = false; ///< Disable camera modes.
    static constexpr bool c_stdCamera_usesROI         = true;  ///< Enable ROI control.
    static constexpr bool c_stdCamera_cropMode        = false; ///< Disable crop mode.
    static constexpr bool c_stdCamera_hasShutter      = false; ///< Disable shutter control.
    static constexpr bool c_stdCamera_usesStateString = false; ///< Disable persistent state descriptions.

    using stdCameraT::m_currentROI;
    using stdCameraT::m_default_h;
    using stdCameraT::m_default_w;
    using stdCameraT::m_default_x;
    using stdCameraT::m_default_y;
    using stdCameraT::m_indiP_roi_check_result;
    using stdCameraT::m_indiP_roi_region_state;
    using stdCameraT::m_indiP_roi_x;
    using stdCameraT::m_indiP_roi_x_relative;
    using stdCameraT::m_nextROI;
    using stdCameraT::m_roiUnits;
    using stdCameraT::updateROIGeometry;
    bool m_reconfig{ false }; ///< Unused mode reconfiguration flag.
    bool m_snap{ false };     ///< Inject a camera width/center adjustment.
    int  m_checkResult{ 0 };  ///< Inject a failed validator.
    int  m_sets{ 0 };         ///< Actual derived apply calls.

    /// Initialize a noncentral camera ROI.
    Camera();

    /// Register the real camera properties.
    int appStartup() override;

    /// Exercise the shared periodic state publication.
    int appLogic() override;

    /// Provide the shared shutdown interface.
    int appShutdown() override;

    /// Leave the shared power defaults unchanged.
    int powerOnDefaults();

    /// Simulate camera validation without hardware.
    int checkNextROI();

    /// Record an actual apply request.
    int setNextROI();

    /// Discard telemetry without starting a logger.
    template <class T>
    int telem( const typename T::messageT &message /**< [in] Telemetry payload. */ );
};
Camera::Camera() : outletTestApp<>( "test", false )
{
    m_currentROI = { 511.5, 511.5, 64, 64, 1, 1 };
    m_nextROI    = m_currentROI;
    m_full_x = m_full_y = 511.5;
    m_full_w = m_full_h = 1024;
    m_default_x = m_default_y = 511.5;
    m_default_w = m_default_h = 64;
}
int Camera::appStartup()
{
    return stdCameraT::appStartup();
}
int Camera::appLogic()
{
    return stdCameraT::appLogic();
}
int Camera::appShutdown()
{
    return stdCameraT::appShutdown();
}
int Camera::powerOnDefaults()
{
    return 0;
}
int Camera::checkNextROI()
{
    if( m_snap )
    {
        m_nextROI.w = 32;
        m_nextROI.x = 510.5;
    }
    return m_checkResult;
}
int Camera::setNextROI()
{
    ++m_sets;
    return 0;
}
template <class T>
int Camera::telem( const typename T::messageT & )
{
    return 0;
}

/// Fixture with private transport and no active app threads.
class Fixture : public outletHarness::Controller<Camera>
{
  public:
    /// Initialize real INDI properties and mark the camera ready.
    Fixture();

    /// Dispatch a numeric request through the actual shared dispatcher.
    int number( const std::string &name,  /**< [in] Property name. */
                const std::string &value, /**< [in] Raw element value. */
                const std::string &element = "target" /**< [in] Element name. */ );

    /// Dispatch a standard request switch.
    int request( const std::string &name /**< [in] Request property name. */ );
};
Fixture::Fixture()
{
    outletHarness::g_faults = {};
    driver();
    REQUIRE( appStartup() == 0 );
    state( MagAOX::app::stateCodes::READY );
    updateROIGeometry();
}
int Fixture::number( const std::string &name, const std::string &value, const std::string &element )
{
    pcf::IndiProperty p( pcf::IndiProperty::Number, configName(), name );
    p.add( pcf::IndiElement( element, value ) );
    return newCallBack_stdCamera( p );
}
int Fixture::request( const std::string &name )
{
    pcf::IndiProperty p( pcf::IndiProperty::Switch, configName(), name );
    p.add( pcf::IndiElement( "request" ) );
    p["request"].setSwitchState( pcf::IndiElement::On );
    return newCallBack_stdCamera( p );
}
} // namespace roiHarness
/// \endcond

namespace libXWCTest
{
/** \brief Relative requests stage absolute centers once while old clients retain their meaning.
 * \ingroup stdCamera_unit_test
 */
TEST_CASE( "Image-relative ROI callbacks preserve absolute targets", "[stdCamera][roi]" )
{
    // clang-format off
#ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera<Camera>::newCallBack_roi_relative;
    MagAOX::app::dev::stdCamera<Camera>::newCallBack_stdCamera;
#endif
    // clang-format on
    roiHarness::Fixture cam;
    REQUIRE( cam.number( "roi_region_x_relative", "30.5" ) == 0 );
    CHECK( cam.m_nextROI.x == 510.5f );
    CHECK( cam.m_currentROI.x == 511.5f );
    CHECK( cam.m_sets == 0 );
    REQUIRE( cam.number( "roi_region_x_relative", "30.5" ) == 0 );
    CHECK( cam.m_nextROI.x == 510.5f );
    REQUIRE( cam.number( "roi_region_x", "17.5" ) == 0 );
    CHECK( cam.m_nextROI.x == 17.5f );
    REQUIRE( cam.number( "roi_region_x", "10.5", "current" ) == 0 );
    CHECK( cam.m_nextROI.x == 10.5f );
    for( auto value : { "", "nan", "inf", "1e99", "17oops" } )
    {
        CHECK( cam.number( "roi_region_x_relative", value ) < 0 );
        CHECK( cam.m_nextROI.x == 10.5f );
    }
    cam.state( MagAOX::app::stateCodes::CONFIGURING );
    CHECK( cam.number( "roi_region_x_relative", "20" ) < 0 );
    REQUIRE( cam.appLogic() == 0 );
    CHECK( cam.m_indiP_roi_region_state.getState() == pcf::IndiProperty::Idle );
}

/** \brief Pixel centers round-trip for parity, noncentral regions, and different axis/binning conventions.
 * \ingroup stdCamera_unit_test
 */
TEST_CASE( "ROI coordinates round-trip across binning and parity", "[stdCamera][roi]" )
{
    using namespace MagAOX::roi;
    for( auto u : { units::sensor, units::binned } )
        for( int bin : { 1, 2, 3 } )
            for( int targetBin : { 1, 2, 3 } )
                for( int size : { 1, 2, 63, 64, 65 } )
                {
                    int    extent = u == units::sensor ? size * bin : size;
                    double center = 371.5;
                    for( double r : { -1., 0., ( size - 1 ) / 2., static_cast<double>( size ) } )
                    {
                        double a = absolute( r, targetBin, center, extent, bin, u );
                        CHECK( relative( a, targetBin, center, extent, bin, u ) == Approx( r ).margin( 1e-10 ) );
                    }
                    double at = absolute( ( size - 1 ) / 2. - 1, targetBin, center, extent, bin, u );
                    CHECK( physical( at, targetBin, u ) == Approx( physical( center, bin, u ) - bin ) );
                }
    region invalid{0, 0, 1, 1, 1, 1};
    invalid.x = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(valid(invalid));
    invalid.x = std::numeric_limits<float>::infinity();
    CHECK_FALSE(valid(invalid));
    roiHarness::Fixture cam;
    cam.m_currentROI = { 0, 40, 65, 63, 1, 1 };
    cam.m_nextROI    = cam.m_currentROI;
    REQUIRE( cam.number( "roi_region_x_relative", "31" ) == 0 );
    CHECK( cam.m_nextROI.x == -1 );
    REQUIRE( cam.number( "roi_region_y_relative", "30" ) == 0 );
    CHECK( cam.m_nextROI.y == 39 );
}

/** \brief A completed Check captures before/after geometry even for no-op or failed validators.
 * \ingroup stdCamera_unit_test
 */
TEST_CASE( "ROI Check publishes a coherent completed result", "[stdCamera][roi]" )
{
    // clang-format off
#ifdef STDCAMERA_TEST_DOXYGEN_REF
    MagAOX::app::dev::stdCamera<Camera>::newCallBack_roi_check;
    MagAOX::app::dev::stdCamera<Camera>::updateROIGeometry;
#endif
    // clang-format on
    roiHarness::Fixture cam;
    CHECK( cam.m_indiP_roi_region_state["is_default"].get<int>() == 1 );
    CHECK( cam.m_indiP_roi_region_state["is_full"].get<int>() == 0 );
    REQUIRE( cam.request( "roi_region_check" ) == 0 );
    CHECK( cam.m_indiP_roi_check_result["sequence"].get<int>() == 1 );
    CHECK( cam.m_indiP_roi_check_result["requested_w"].get<int>() == 64 );
    CHECK( cam.m_indiP_roi_check_result["checked_w"].get<int>() == 64 );
    CHECK( cam.m_sets == 0 );
    cam.m_snap = true;
    REQUIRE( cam.request( "roi_region_check" ) == 0 );
    CHECK( cam.m_indiP_roi_check_result["sequence"].get<int>() == 2 );
    CHECK( cam.m_indiP_roi_check_result["requested_w"].get<int>() == 64 );
    CHECK( cam.m_indiP_roi_check_result["checked_w"].get<int>() == 32 );
    CHECK( cam.m_indiP_roi_check_result["current_w"].get<int>() == 64 );
    CHECK( cam.m_sets == 0 );
    cam.m_checkResult = -1;
    CHECK( cam.request( "roi_region_check" ) == -1 );
    CHECK( cam.m_indiP_roi_check_result.getState() == pcf::IndiProperty::Alert );
    CHECK( cam.m_indiP_roi_check_result["sequence"].get<int>() == 3 );
    cam.m_currentROI.x += 1;
    cam.updateROIGeometry();
    CHECK( cam.m_indiP_roi_region_state["is_default"].get<int>() == 0 );
    REQUIRE( cam.request( "roi_set_default" ) == 0 );
    CHECK( cam.m_nextROI.w == 64 );
    CHECK( cam.m_sets == 1 );
    REQUIRE( cam.MagAOX::app::dev::stdCamera<roiHarness::Camera>::onPowerOff() == 0 );
    CHECK( cam.m_indiP_roi_check_result.getState() == pcf::IndiProperty::Idle );
    CHECK( cam.m_indiP_roi_x_relative.getState() == pcf::IndiProperty::Idle );
}
} // namespace libXWCTest
