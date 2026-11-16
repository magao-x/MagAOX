/** \file ocam2KCtrl_descramble_test.cpp
 * \brief Offline tests of frame-boundary OCAM descramble selection and fallback.
 * \ingroup ocam2KCtrl_descramble_unit_test
 */

/// \cond OCAM2KCTRL_TEST_SUPPORT
#define OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "ocam2KCtrl_test.cpp"
#undef OCAM2KCTRL_TEST_SUPPORT_ONLY
#include "../../../tests/outletAppTest.hpp"
/// \endcond

#include "../../../tests/testXWC.hpp"
#include "ocamDescrambleFixture.hpp"

namespace libXWCTest
{
namespace ocam2KCtrlTest
{

/// \cond OCAM2KCTRL_TEST_SUPPORT
namespace
{

/// Supply the SDK stub's response to identity-input map preparation for bin2.
void prepareBin2Map( ocam2KCtrl_test &app /**< [in,out] Controller receiving the structured bin2 map. */ )
{
    const auto table              = ocamDescrambleTable();
    g_ocam2StubState.reportedMode = OCAM2_BINNING2x2;
    g_ocam2StubState.outputImage.resize( OCAM2_PIXELS_IMAGE_BINNING2x2 );

    for( size_t i = 0; i < OCAM2_PIXELS_IMAGE_BINNING2x2 / 2; ++i )
    {
        g_ocam2StubState.outputImage[i]        = table[2 * i];
        g_ocam2StubState.outputImage[7200 + i] = table[43200 + 2 * i];
    }

    app.prepareDescramble();
}

/// Construct an INDI method request, leaving the other switch implicit.
pcf::IndiProperty methodRequest( ocam2KCtrl_test   &app /**< [in] Controller receiving the request. */,
                                 const std::string &method /**< [in] Requested switch element. */ )
{
    pcf::IndiProperty request( pcf::IndiProperty::Switch, app.configName(), "descramble_method" );
    request.add( pcf::IndiElement( method, pcf::IndiElement::On ) );
    return request;
}

} // namespace
/// \endcond

/// Default, explicit and invalid configuration values cannot silently select an unknown implementation.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl validates descramble method configuration", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::setupConfig;
    MagAOX::app::ocam2KCtrl::loadConfigImpl;
    #endif
    // clang-format on
    for( const std::string method : { "", "sdk", "optimized", "parallel", "typo" } )
    {
        CAPTURE( method );
        resetStubState();
        ocam2KCtrl_test app;
        app.setupConfig();
        const std::string path = uniqueConfigPath( "descramble" );

        if( method.empty() )
        {
            mx::app::writeConfigFile( path, { "unused" }, { "value" }, { "0" } );
        }
        else
        {
            mx::app::writeConfigFile( path, { "camera" }, { "descrambleMethod" }, { method } );
        }

        app.config.readConfig( path );
        const int result = app.loadConfigImpl();
        std::remove( path.c_str() );

        if( method == "parallel" || method == "typo" )
        {
            REQUIRE( result == -1 );
        }
        else
        {
            REQUIRE( result == 0 );
            REQUIRE( app.m_descrambleRequested.load() == ( method == "optimized"
                                                               ? ocam2KCtrl::DescrambleMethod::optimized
                                                               : ocam2KCtrl::DescrambleMethod::sdk ) );
        }
    }
}

/// Requests acknowledge completion only after a whole load, without serial traffic or a reconfiguration request.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl switches descramblers between complete frames", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::initDescramble;
    MagAOX::app::ocam2KCtrl::prepareDescramble;
    MagAOX::app::ocam2KCtrl::loadImageIntoStream;
    MagAOX::app::ocam2KCtrl::updateDescramble;
    MagAOX::app::ocam2KCtrl::newCallBack_m_indiP_descrambleMethod;
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_reconfig = false;
    REQUIRE( app.initDescramble() == 0 );
    prepareBin2Map( app );
    REQUIRE( app.m_descrambler.structured() );
    const auto         map = g_ocam2StubState.outputImage;
    std::vector<short> raw( OCAM2_PIXELS_RAW_BINNING2x2 );
    std::vector<short> output( map.size() );
    std::vector<short> expected( map.size() );

    for( size_t i = 0; i < raw.size(); ++i )
    {
        raw[i] = static_cast<short>( i % 317 );
    }

    for( size_t i = 0; i < map.size(); ++i )
    {
        expected[i] = raw[map[i]];
    }

    app.m_image_p = reinterpret_cast<u_char *>( raw.data() );
    g_ocam2StubState.outputImage.assign( map.size(), -777 );
    const auto sdkOutput = g_ocam2StubState.outputImage;
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE( output == sdkOutput );
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == "sdk" );

    for( const std::string method : { "optimized", "sdk", "optimized", "sdk" } )
    {
        const auto prior = app.m_descrambleApplied.load();
        REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, method ) ) == 0 );
        REQUIRE( app.m_descrambleApplied.load() == prior );
        REQUIRE( app.m_indiP_descrambleMethod.getState() == INDI_BUSY );
        REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
        REQUIRE( output == ( method == "sdk" ? sdkOutput : expected ) );
        app.updateDescramble();
        REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == method );
        REQUIRE( app.m_indiP_descrambleMethod.getState() == INDI_OK );
    }

    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE_FALSE( app.m_reconfig );
    REQUIRE( app.m_nextMode.empty() );

    std::thread requester(
        [&app]()
        {
            for( size_t i = 0; i < 200; ++i )
            {
                app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, i % 2 ? "sdk" : "optimized" ) );
            }
        } );
    bool completeFrames = true;

    for( size_t i = 0; i < 200; ++i )
    {
        app.loadImageIntoStream( output.data() );
        completeFrames = completeFrames && ( output == sdkOutput || output == expected );
    }

    requester.join();
    REQUIRE( completeFrames );
    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE_FALSE( app.m_reconfig );
}

/// Unknown, ambiguous and empty switch selections preserve the previous request.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl rejects malformed descramble requests", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::newCallBack_m_indiP_descrambleMethod;
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    REQUIRE( app.initDescramble() == 0 );

    for( const std::string kind : { "unknown", "both", "off", "wrong property" } )
    {
        CAPTURE( kind );
        auto request = methodRequest( app, kind == "unknown" ? "parallel" : "optimized" );

        if( kind == "both" )
        {
            request.add( pcf::IndiElement( "sdk", pcf::IndiElement::On ) );
        }
        else if( kind == "off" )
        {
            request["optimized"].setSwitchState( pcf::IndiElement::Off );
        }
        else if( kind == "wrong property" )
        {
            request.setName( "wrong" );
        }

        REQUIRE( app.newCallBack_m_indiP_descrambleMethod( request ) == -1 );
        REQUIRE( app.m_descrambleRequested.load() == ocam2KCtrl::DescrambleMethod::sdk );
    }
}

/// Unsupported modes, unavailable structures and lifecycle transitions retain SDK fallback and truthful status.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl invalidates optimized descramble across configuration epochs", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::prepareDescramble;
    MagAOX::app::ocam2KCtrl::releaseOcamSdk;
    MagAOX::app::ocam2KCtrl::updateDescramble;
    MagAOX::app::ocam2KCtrl::loadImageIntoStream;
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_reconfig = false;
    REQUIRE( app.initDescramble() == 0 );
    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "optimized" ) ) == 0 );

    for( const auto mode : { OCAM2_NORMAL, OCAM2_BINNING1x3, OCAM2_BINNING1x4, OCAM2_BINNING2x2 } )
    {
        prepareBin2Map( app );
        REQUIRE( app.m_descrambler.structured() );
        g_ocam2StubState.reportedMode = mode;
        g_ocam2StubState.outputImage.assign( OCAM2_PIXELS_IMAGE_BINNING2x2, 0 );
        app.prepareDescramble();
        REQUIRE_FALSE( app.m_descrambler.ready() );
        REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::none );

        std::array<short, 4> raw{};
        std::array<short, 4> output{};
        g_ocam2StubState.outputImage = { 17, 21, 35, 49 };
        app.m_image_p                = reinterpret_cast<u_char *>( raw.data() );
        REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
        REQUIRE( output == ( std::array<short, 4>{ 17, 21, 35, 49 } ) );
        app.updateDescramble();
        REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == "sdk" );
        REQUIRE( app.m_indiP_descrambleMethod.getState() == INDI_ALERT );
    }

    prepareBin2Map( app );
    app.m_ocam2Initialized = true;
    app.m_ocam2_id         = 0;
    app.releaseOcamSdk();
    REQUIRE( g_ocam2StubState.exitCalls == 1 );
    REQUIRE( g_ocam2StubState.lastExitId == 0 );
    REQUIRE_FALSE( app.m_descrambler.ready() );
    REQUIRE( app.m_descrambleAvailability.load() == ocam2KCtrl::DescrambleAvailability::unconfigured );
    REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::none );
    prepareBin2Map( app );
    REQUIRE( app.m_descrambler.structured() );
    app.m_descrambleApplied.store( ocam2KCtrl::DescrambleMethod::optimized );
    static_cast<MagAOXAppT &>( app ).m_powerState = 0;
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == "none" );
}

/// Digital binning retains the historical last-X-sample behavior for nonuniform input.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl descramble selection preserves digital binning", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::loadImageIntoStream;
    #endif
    // clang-format on
    resetStubState();
    ocam2KCtrl_test      app;
    std::array<short, 4> raw{};
    std::array<short, 4> output{};
    app.m_image_p     = reinterpret_cast<u_char *>( raw.data() );
    app.m_digitalBin  = true;
    app.m_digitalBinX = 3;
    app.m_digitalBinY = 1;
    app.m_width       = 2;
    app.m_height      = 2;
    app.m_digitalBinWork.resize( 6, 2 );
    g_ocam2StubState.outputImage = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    for( const auto requested : { ocam2KCtrl::DescrambleMethod::sdk, ocam2KCtrl::DescrambleMethod::optimized } )
    {
        app.m_descrambleRequested.store( requested );
        REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
        REQUIRE( output == ( std::array<short, 4>{ 3, 6, 9, 12 } ) );
        REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::sdk );
    }

    prepareBin2Map( app );
    REQUIRE( app.m_descrambler.structured() );
    const auto         map = g_ocam2StubState.outputImage;
    std::vector<short> bin2Raw( OCAM2_PIXELS_RAW_BINNING2x2 );
    std::vector<short> bin2Output( 60 * 120 );

    for( size_t i = 0; i < bin2Raw.size(); ++i )
    {
        bin2Raw[i] = static_cast<short>( i % 311 );
    }

    app.m_image_p     = reinterpret_cast<u_char *>( bin2Raw.data() );
    app.m_digitalBinX = 2;
    app.m_width       = 60;
    app.m_height      = 120;
    app.m_digitalBinWork.resize( 120, 120 );
    g_ocam2StubState.outputImage.assign( map.size(), -777 );
    g_ocam2StubState.lastRaw = nullptr;
    REQUIRE( app.loadImageIntoStream( bin2Output.data() ) == 0 );
    REQUIRE( g_ocam2StubState.lastRaw == nullptr );
    REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::optimized );

    for( size_t i = 0; i < bin2Output.size(); ++i )
    {
        REQUIRE( bin2Output[i] == bin2Raw[map[2 * i + 1]] );
    }
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
