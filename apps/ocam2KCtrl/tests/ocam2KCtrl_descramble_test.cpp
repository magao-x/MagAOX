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

/// Construct an INDI method request, leaving the other switches implicit.
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
    MagAOX::app::ocam2KCtrl::initDescramble;
    #endif
    // clang-format on
    for( const std::string method : { "", "sdk", "compact", "optimized", "parallel", "typo" } )
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
            const auto expected = method == "compact"     ? ocam2KCtrl::DescrambleMethod::compact
                                  : method == "optimized" ? ocam2KCtrl::DescrambleMethod::optimized
                                                          : ocam2KCtrl::DescrambleMethod::sdk;
            REQUIRE( app.m_descrambleRequested.load() == expected );
            REQUIRE( app.initDescramble() == 0 );

            for( const std::string choice : { "sdk", "compact", "optimized" } )
            {
                REQUIRE(
                    app.m_indiP_descrambleMethod[choice].getSwitchState() ==
                    ( choice == ( method.empty() ? "sdk" : method ) ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
            }
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

    for( const std::string method : { "compact", "optimized", "compact", "sdk", "optimized", "sdk" } )
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

        for( const std::string choice : { "sdk", "compact", "optimized" } )
        {
            REQUIRE( app.m_indiP_descrambleMethod[choice].getSwitchState() ==
                     ( choice == method ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        }
    }

    REQUIRE( g_edtStubState.serialCommands.empty() );
    REQUIRE_FALSE( app.m_reconfig );
    REQUIRE( app.m_nextMode.empty() );

    std::atomic<bool> begin{ false };
    std::atomic<bool> requestsAccepted{ true };
    std::thread       requester(
        [&app, &begin, &requestsAccepted]()
        {
            const std::array<std::string, 3> choices{ "sdk", "compact", "optimized" };

            while( !begin.load( std::memory_order_acquire ) )
            {
                std::this_thread::yield();
            }

            for( size_t i = 0; i < 300; ++i )
            {
                if( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, choices[i % choices.size()] ) ) != 0 )
                {
                    requestsAccepted.store( false );
                }
            }
        } );
    bool completeFrames = true;
    begin.store( true, std::memory_order_release );

    for( size_t i = 0; i < 300; ++i )
    {
        app.loadImageIntoStream( output.data() );
        const auto applied = app.m_descrambleApplied.load();
        completeFrames     = completeFrames && ( applied == ocam2KCtrl::DescrambleMethod::sdk
                                                     ? output == sdkOutput
                                                     : ( applied == ocam2KCtrl::DescrambleMethod::compact ||
                                                     applied == ocam2KCtrl::DescrambleMethod::optimized ) &&
                                                       output == expected );
    }

    requester.join();
    REQUIRE( requestsAccepted.load() );
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
    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "compact" ) ) == 0 );

    for( const std::string kind : { "unknown", "off", "wrong property", "wrong device", "empty" } )
    {
        CAPTURE( kind );
        auto request = methodRequest( app, kind == "unknown" ? "parallel" : "optimized" );

        if( kind == "off" )
        {
            request["optimized"].setSwitchState( pcf::IndiElement::Off );
        }
        else if( kind == "wrong property" )
        {
            request.setName( "wrong" );
        }
        else if( kind == "wrong device" )
        {
            request.setDevice( "wrong" );
        }
        else if( kind == "empty" )
        {
            request = pcf::IndiProperty( pcf::IndiProperty::Switch, app.configName(), "descramble_method" );
        }

        REQUIRE( app.newCallBack_m_indiP_descrambleMethod( request ) == -1 );
        REQUIRE( app.m_descrambleRequested.load() == ocam2KCtrl::DescrambleMethod::compact );
    }

    for( unsigned int mask : { 3u, 5u, 6u, 7u } )
    {
        CAPTURE( mask );
        pcf::IndiProperty                request( pcf::IndiProperty::Switch, app.configName(), "descramble_method" );
        const std::array<std::string, 3> choices{ "sdk", "compact", "optimized" };

        for( size_t index = 0; index < choices.size(); ++index )
        {
            request.add( pcf::IndiElement( choices[index],
                                           mask & ( 1u << index ) ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        }

        REQUIRE( app.newCallBack_m_indiP_descrambleMethod( request ) == -1 );
        REQUIRE( app.m_descrambleRequested.load() == ocam2KCtrl::DescrambleMethod::compact );
    }
}

/// Unsupported modes, unavailable structures and lifecycle transitions retain SDK fallback and truthful status.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl invalidates candidate descramblers across configuration epochs", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::prepareDescramble;
    MagAOX::app::ocam2KCtrl::releaseOcamSdk;
    MagAOX::app::ocam2KCtrl::updateDescramble;
    MagAOX::app::ocam2KCtrl::loadImageIntoStream;
    #endif
    // clang-format on
    for( const std::string method : { "compact", "optimized" } )
    {
        CAPTURE( method );
        resetStubState();
        outletHarness::Controller<ocam2KCtrl_test> app;
        app.driver();
        setPoweredOn( app );
        app.state( stateCodes::OPERATING );
        app.m_reconfig = false;
        REQUIRE( app.initDescramble() == 0 );
        REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, method ) ) == 0 );

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
            REQUIRE( app.m_indiP_descrambleStatus["availability"].get() ==
                     ( mode == OCAM2_BINNING2x2 ? "unavailable table" : "unsupported mode" ) );
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
        app.m_descrambleApplied.store( method == "compact" ? ocam2KCtrl::DescrambleMethod::compact
                                                           : ocam2KCtrl::DescrambleMethod::optimized );
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.updateDescramble();
        REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == "none" );
    }
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

    for( const auto requested : { ocam2KCtrl::DescrambleMethod::sdk,
                                  ocam2KCtrl::DescrambleMethod::compact,
                                  ocam2KCtrl::DescrambleMethod::optimized } )
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

    for( const auto requested : { ocam2KCtrl::DescrambleMethod::compact, ocam2KCtrl::DescrambleMethod::optimized } )
    {
        app.m_descrambleRequested.store( requested );
        g_ocam2StubState.lastRaw = nullptr;
        REQUIRE( app.loadImageIntoStream( bin2Output.data() ) == 0 );
        REQUIRE( g_ocam2StubState.lastRaw == nullptr );
        REQUIRE( app.m_descrambleApplied.load() == requested );

        for( size_t i = 0; i < bin2Output.size(); ++i )
        {
            REQUIRE( bin2Output[i] == bin2Raw[map[2 * i + 1]] );
        }
    }
}

} // namespace ocam2KCtrlTest
} // namespace libXWCTest
