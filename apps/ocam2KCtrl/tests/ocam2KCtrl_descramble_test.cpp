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

/// Every selectable runtime method, shared by selection and ambiguity checks.
constexpr std::array<const char *, 5> descrambleChoices{ "sdk", "compact", "mapped", "optimized", "parallel" };

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

/// Launch a local normal-priority worker without exercising instrument placement or RT configuration.
void startLocalParallel( ocam2KCtrl_test &app /**< [in,out] Controller owning the worker and prepared map. */ )
{
    app.m_staged           = true;
    app.m_descrambleCpuset = "offline-test";
    REQUIRE( app.m_descrambleParallel.prepare( app.m_descrambler ) == 0 );
    app.m_descrambleThread = std::thread(
        [&app]()
        {
            app.m_descrambleParallel.workerMain();
        } );
    REQUIRE( app.m_descrambleParallel.waitUntilStarted() );
    app.m_descrambleWorkerVerified.store( true );
    app.m_descrambleParallelAvailable.store( true );
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
    for( const std::string method : { "", "sdk", "compact", "mapped", "optimized", "parallel", "typo" } )
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

        if( method == "typo" )
        {
            REQUIRE( result == -1 );
        }
        else
        {
            REQUIRE( result == 0 );
            const auto expected = method == "compact"  ? ocam2KCtrl::DescrambleMethod::compact
                                  : method == "mapped" ? ocam2KCtrl::DescrambleMethod::mapped
                                  : ( method.empty() || method == "optimized" )
                                      ? ocam2KCtrl::DescrambleMethod::optimized
                                  : method == "parallel" ? ocam2KCtrl::DescrambleMethod::parallel
                                                         : ocam2KCtrl::DescrambleMethod::sdk;
            REQUIRE( app.m_descrambleRequested.load() == expected );
            REQUIRE( app.initDescramble() == 0 );

            for( const std::string choice : descrambleChoices )
            {
                REQUIRE( app.m_indiP_descrambleMethod[choice].getSwitchState() ==
                         ( choice == ( method.empty() ? "optimized" : method ) ? pcf::IndiElement::On
                                                                               : pcf::IndiElement::Off ) );
            }
        }
    }
}

/// Worker configuration validates the partition and resolved priority without starting a thread.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl validates parallel worker configuration", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::setupConfig;
    MagAOX::app::ocam2KCtrl::loadConfigImpl;
    #endif
    // clang-format on
    for( const std::string partition : { "halves", "paired28", "paired30", "unknown" } )
    {
        for( const std::string priority : { "-2", "-1", "0", "1", "99", "100" } )
        {
            CAPTURE( partition, priority );
            resetStubState();
            ocam2KCtrl_test app;
            app.setupConfig();
            const auto path = uniqueConfigPath( "parallel" );
            mx::app::writeConfigFile( path,
                                      { "descramble", "descramble", "descramble", "framegrabber" },
                                      { "cpuset", "threadPrio", "partition", "threadPrio" },
                                      { "operator-assigned", priority, partition, "47" } );
            app.config.readConfig( path );
            const int result = app.loadConfigImpl();
            std::remove( path.c_str() );
            const bool valid = partition != "unknown" && ( priority == "-1" || priority == "1" || priority == "99" );
            REQUIRE( result == ( valid ? 0 : -1 ) );
            REQUIRE_FALSE( app.m_descrambleThread.joinable() );

            if( valid )
            {
                REQUIRE( app.m_descrambleCpuset == "operator-assigned" );
                REQUIRE( app.m_descrambleThreadPrio == ( priority == "-1" ? 47 : std::stoi( priority ) ) );
                REQUIRE( app.m_descramblePartition ==
                         ( partition == "halves"     ? MagAOX::app::ocamDescrambleParallel::Partition::halves
                           : partition == "paired28" ? MagAOX::app::ocamDescrambleParallel::Partition::paired28
                                                     : MagAOX::app::ocamDescrambleParallel::Partition::paired30 ) );
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
    startLocalParallel( app );
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
    REQUIRE( output == expected );
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == "optimized" );

    for( const std::string method : { "parallel",
                                      "sdk",
                                      "parallel",
                                      "compact",
                                      "mapped",
                                      "parallel",
                                      "compact",
                                      "optimized",
                                      "mapped",
                                      "optimized",
                                      "sdk",
                                      "mapped",
                                      "sdk",
                                      "optimized",
                                      "sdk" } )
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

        for( const std::string choice : descrambleChoices )
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
            while( !begin.load( std::memory_order_acquire ) )
            {
                std::this_thread::yield();
            }

            for( size_t i = 0; i < 300; ++i )
            {
                if( app.newCallBack_m_indiP_descrambleMethod(
                        methodRequest( app, descrambleChoices[i % descrambleChoices.size()] ) ) != 0 )
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
                                                     applied == ocam2KCtrl::DescrambleMethod::mapped ||
                                                     applied == ocam2KCtrl::DescrambleMethod::optimized ||
                                                     applied == ocam2KCtrl::DescrambleMethod::parallel ) &&
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
    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "mapped" ) ) == 0 );

    for( const std::string kind : { "unknown", "off", "wrong property", "wrong device", "empty" } )
    {
        CAPTURE( kind );
        auto request = methodRequest( app, kind == "unknown" ? "unknown" : "optimized" );

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
        REQUIRE( app.m_descrambleRequested.load() == ocam2KCtrl::DescrambleMethod::mapped );
    }

    for( unsigned int mask = 1; mask < ( 1u << descrambleChoices.size() ); ++mask )
    {
        // A single selected element is valid; test every combination with multiple selected elements.
        if( ( mask & ( mask - 1 ) ) == 0 )
        {
            continue;
        }

        CAPTURE( mask );
        pcf::IndiProperty request( pcf::IndiProperty::Switch, app.configName(), "descramble_method" );

        for( size_t index = 0; index < descrambleChoices.size(); ++index )
        {
            request.add( pcf::IndiElement( descrambleChoices[index],
                                           mask & ( 1u << index ) ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
        }

        REQUIRE( app.newCallBack_m_indiP_descrambleMethod( request ) == -1 );
        REQUIRE( app.m_descrambleRequested.load() == ocam2KCtrl::DescrambleMethod::mapped );
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
    for( const std::string method : { "compact", "mapped", "optimized", "parallel" } )
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
        app.m_descrambleApplied.store( method == "compact"    ? ocam2KCtrl::DescrambleMethod::compact
                                       : method == "mapped"   ? ocam2KCtrl::DescrambleMethod::mapped
                                       : method == "parallel" ? ocam2KCtrl::DescrambleMethod::parallel
                                                              : ocam2KCtrl::DescrambleMethod::optimized );
        static_cast<MagAOXAppT &>( app ).m_powerState = 0;
        app.updateDescramble();
        REQUIRE( app.m_indiP_descrambleStatus["applied"].get<std::string>() == "none" );
    }
}

/// Parallel dispatch requires staged output and a live prepared worker, and drains before SDK or map reuse.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl parks parallel work and reports unavailable workers", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::startDescrambleWorker;
    MagAOX::app::ocam2KCtrl::stopDescrambleWorker;
    MagAOX::app::ocam2KCtrl::loadImageIntoStream;
    MagAOX::app::ocam2KCtrl::updateDescramble;
    MagAOX::app::ocam2KCtrl::releaseOcamSdk;
    #endif
    // clang-format on
    resetStubState();
    outletHarness::Controller<ocam2KCtrl_test> app;
    app.driver();
    setPoweredOn( app );
    app.state( stateCodes::OPERATING );
    app.m_reconfig = false;
    REQUIRE( app.initDescramble() == 0 );
    REQUIRE( app.startDescrambleWorker() == 0 );
    REQUIRE_FALSE( app.m_descrambleThread.joinable() );
    prepareBin2Map( app );
    const auto         map = g_ocam2StubState.outputImage;
    std::vector<short> raw( OCAM2_PIXELS_RAW_BINNING2x2, 123 );
    std::vector<short> output( map.size() );
    app.m_image_p = reinterpret_cast<u_char *>( raw.data() );
    g_ocam2StubState.outputImage.assign( map.size(), -777 );
    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "parallel" ) ) == 0 );
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE( output == g_ocam2StubState.outputImage );
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["parallel"].get() == "disabled" );
    REQUIRE( app.m_indiP_descrambleMethod.getState() == INDI_ALERT );

    app.m_descrambleCpuset = "offline-test";
    app.m_staged           = false;
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["parallel"].get() == "requires staging" );
    app.m_staged = true;
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["parallel"].get() == "unverified worker" );
    startLocalParallel( app );

    for( const auto partition : { MagAOX::app::ocamDescrambleParallel::Partition::halves,
                                  MagAOX::app::ocamDescrambleParallel::Partition::paired28,
                                  MagAOX::app::ocamDescrambleParallel::Partition::paired30 } )
    {
        app.m_descramblePartition = partition;
        REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
        REQUIRE( std::all_of( output.begin(),
                              output.end(),
                              []( short value )
                              {
                                  return value == 123;
                              } ) );
        REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::parallel );
        app.updateDescramble();
        REQUIRE( app.m_indiP_descrambleStatus["parallel"].get() == "running" );
    }

    app.m_staged = false;
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE( output == g_ocam2StubState.outputImage );
    REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::sdk );
    REQUIRE_FALSE( app.m_descrambleParallel.running() );
    app.m_staged = true;
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::parallel );

    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "sdk" ) ) == 0 );
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE_FALSE( app.m_descrambleParallel.running() );
    REQUIRE( output == g_ocam2StubState.outputImage );
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["parallel"].get() == "parked" );
    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "parallel" ) ) == 0 );
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    app.m_ocam2Initialized = true;
    app.m_ocam2_id         = 0;
    app.releaseOcamSdk();
    REQUIRE_FALSE( app.m_descrambleParallel.prepared() );
    REQUIRE_FALSE( app.m_descrambleParallel.running() );
    REQUIRE_FALSE( app.m_descrambleParallelAvailable.load() );
    REQUIRE( g_ocam2StubState.lastExitId == 0 );
    app.stopDescrambleWorker();
    REQUIRE_FALSE( app.m_descrambleThread.joinable() );
    REQUIRE( app.m_descrambleParallel.stopped() );
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE( output == g_ocam2StubState.outputImage );
    REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::sdk );
}

/// Definitive worker exit invalidates admission before SDK fallback and publishes an operator-visible alert.
/** \ingroup ocam2KCtrl_descramble_unit_test */
TEST_CASE( "ocam2KCtrl invalidates a stopped parallel worker", "[ocam2KCtrl][descramble]" )
{
    // clang-format off
    #ifdef OCAM2KCTRL_TEST_DOXYGEN_REF
    MagAOX::app::ocam2KCtrl::loadImageIntoStream;
    MagAOX::app::ocam2KCtrl::updateDescramble;
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
    startLocalParallel( app );
    std::vector<short> raw( OCAM2_PIXELS_RAW_BINNING2x2, 123 );
    std::vector<short> output( OCAM2_PIXELS_IMAGE_BINNING2x2 );
    app.m_image_p = reinterpret_cast<u_char *>( raw.data() );
    g_ocam2StubState.outputImage.assign( output.size(), -777 );
    REQUIRE( app.newCallBack_m_indiP_descrambleMethod( methodRequest( app, "parallel" ) ) == 0 );
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    app.m_descrambleParallel.requestStop();
    app.m_descrambleThread.join();
    REQUIRE( app.m_descrambleParallelAvailable.load() );
    REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
    REQUIRE( output == g_ocam2StubState.outputImage );
    REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::sdk );
    REQUIRE_FALSE( app.m_descrambleParallelAvailable.load() );
    REQUIRE_FALSE( app.m_descrambleWorkerVerified.load() );
    app.updateDescramble();
    REQUIRE( app.m_indiP_descrambleStatus["parallel"].get() == "unverified worker" );
    REQUIRE( app.m_indiP_descrambleMethod.getState() == INDI_ALERT );
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
                                  ocam2KCtrl::DescrambleMethod::mapped,
                                  ocam2KCtrl::DescrambleMethod::optimized,
                                  ocam2KCtrl::DescrambleMethod::parallel } )
    {
        app.m_descrambleRequested.store( requested );
        REQUIRE( app.loadImageIntoStream( output.data() ) == 0 );
        REQUIRE( output == ( std::array<short, 4>{ 3, 6, 9, 12 } ) );
        REQUIRE( app.m_descrambleApplied.load() == ocam2KCtrl::DescrambleMethod::sdk );
    }

    prepareBin2Map( app );
    REQUIRE( app.m_descrambler.structured() );
    const auto map = g_ocam2StubState.outputImage;
    startLocalParallel( app );
    setPoweredOn( app );
    app.m_reconfig = false;
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

    for( const auto requested : { ocam2KCtrl::DescrambleMethod::compact,
                                  ocam2KCtrl::DescrambleMethod::mapped,
                                  ocam2KCtrl::DescrambleMethod::optimized,
                                  ocam2KCtrl::DescrambleMethod::parallel } )
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
