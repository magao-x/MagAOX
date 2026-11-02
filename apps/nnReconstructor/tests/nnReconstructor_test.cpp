/** \file nnReconstructor_test.cpp
 * \brief Exercise the real app methods using deterministic CPU-only SDK doubles.
 * \defgroup nnReconstructor_unit_test nnReconstructor Unit Tests
 * \ingroup application_unit_test
 */
#define CATCH_CONFIG_MAIN
#include "../../../tests/testXWC.hpp"
#include "../nnReconstructor.hpp"
#include <future>

namespace MagAOX::app
{
/** \cond NNRECONSTRUCTOR_TEST_SUPPORT */
class nnReconstructor_test : public nnReconstructor
{
  public:
    std::array<float, 16> m_frame{}; ///< Small column-major WFS fixture.

    /// Reset SDK fault injection and configure distinguishable WFS pixels.
    nnReconstructor_test();

    /// Run the production startup and stream-attachment interfaces.
    void prepare( bool enabled /**< [in] Startup admission state. */                  = true,
                  bool discover /**< [in] Omit the optional configured model list. */ = false );

    /// Return the immutable startup catalog.
    std::vector<std::string> models() const;

    /// Exercise production validation after the configuration boundary supplies values.
    int validateConfig( const float  &normalization /**< [in] Image normalization. */,
                        const double &timeout /**< [in] Transition deadline in seconds. */ );

    /// Submit an actual enable callback request.
    int enable( bool value /**< [in] Requested enable state. */ );

    /// Submit an actual configured-model selection.
    int select( int model /**< [in] Configured model index. */ );

    /// Submit an unknown toggle state.
    int invalidEnable();

    /// Submit ambiguous or unconfigured choices.
    int invalidSelection( bool multiple /**< [in] Select two entries instead of an unknown entry. */ );

    /// Read the acknowledged enable state under its mutex.
    bool acknowledged();

    /// Read whether zero acknowledgement is pending.
    bool disabling();

    /// Read whether a model load is pending or running.
    bool loading();

    /// Read the successfully installed filename.
    std::string loaded();

    /// Read the accepted requested filename.
    std::string requested();

    /// Move the request deadline into the past without sleeping.
    void expire();

    /// Run production model dispatch and status updates.
    void control();

    /// Inject an external stream identity change.
    void replaceOutput();

    /// Inject incompatible output storage.
    void badOutput();

    /// Reopen the fake output through the production attachment method.
    void reattachOutput();

    /// Invoke processing without advancing the input counter.
    void rawFrame();

    /// Wait boundedly for the actual loader thread.
    bool waitLoad();

    /// Advance the input counter and process one frame.
    void frame();

    /// Invoke the production WFS attachment method.
    void reconnect();

    /// Reconnect with an incompatible datatype.
    void wrongSource();

    /// Read the protected pending-result flag.
    bool pending();

    /// Read the protected runtime fault.
    std::string fault();

    /// Expose the persistent host input address.
    void *input();

    /// Claim the real publisher snapshot.
    bool claim();

    /// Copy the claimed snapshot into test-owned storage.
    std::vector<float> copy();

    /// Invoke the production post-publication hook.
    void complete();

    /// Read the timestamp owned by the publisher.
    timespec timestamp();
};

inline nnReconstructor_test::nnReconstructor_test()
{
    engineDirs = ".build/catalog";
    std::filesystem::remove_all( engineDirs );
    std::filesystem::create_directories( engineDirs );
    nvinfer1::serializedTensors.clear();
    nnTest::allocation = nnTest::failAllocation = nnTest::copies = nnTest::failCopy = 0;
    nnTest::runtimeFailure = nnTest::engineFailure = nnTest::contextFailure = nnTest::executeFailure = false;
    nnTest::duringExecute                                                                            = {};
    nnTest::duringDeserialize                                                                        = {};
    nnTest::result                                                                                   = 1;
    nnTest::failClock                                                                                = false;
    nvinfer1::tensors   = { { "image", nvinfer1::TensorIOMode::kINPUT, { 4, { 1, 4, 6, 6 } } },
                            { "modes", nvinfer1::TensorIOMode::kOUTPUT, { 2, { 1, 3 } } } };
    m_pupPix            = 6;
    pixels_per_quadrant = 36;
    imageNorm           = 2;
    modalNorm           = 7; // Deliberately non-unit: the legacy value must remain unapplied.
    pup_offset1_x = pup_offset1_y = 0;
    pup_offset2_x = pup_offset2_y = 2;
    for( size_t i = 0; i < m_frame.size(); ++i )
        m_frame[i] = i;
    shmimMonitorT::m_imageStream.array.raw = m_frame.data();
}

inline void nnReconstructor_test::prepare( bool enabled, bool discover )
{
    m_startEnabled = enabled;
    engineName     = "first.trt";
    m_models =
        discover ? std::vector<std::string>{} : std::vector<std::string>{ "first.trt", "second.trt", "missing.trt" };
    std::ofstream( engineDirs + "/first.trt" ) << "first";
    std::ofstream( engineDirs + "/second.trt" ) << "second";
    if( !discover )
        std::ofstream( engineDirs + "/missing.trt" ) << "missing";
    if( appStartup() < 0 )
        throw std::runtime_error( "startup rejected" );
    allocate( dev::shmimT{} );
    configureAcquisition();
}

inline std::vector<std::string> nnReconstructor_test::models() const
{
    return m_models;
}

inline int nnReconstructor_test::enable( bool value )
{
    pcf::IndiProperty p( pcf::IndiProperty::Switch );
    p.setName( "enable" );
    p.add( pcf::IndiElement( "toggle", value ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    return newCallBack_m_indiP_enable( p );
}

inline int nnReconstructor_test::validateConfig( const float &normalization, const double &timeout )
{
    imageNorm           = normalization;
    m_transitionTimeout = timeout;
    engineName          = "first.trt";
    return loadConfigImpl( config );
}

inline int nnReconstructor_test::select( int model )
{
    pcf::IndiProperty p( pcf::IndiProperty::Switch );
    p.setName( "model" );
    p.add( pcf::IndiElement( modelId( model ), pcf::IndiElement::On ) );
    return newCallBack_m_indiP_model( p );
}

inline int nnReconstructor_test::invalidEnable()
{
    pcf::IndiProperty p( pcf::IndiProperty::Switch );
    p.setName( "enable" );
    p.add( pcf::IndiElement( "toggle", pcf::IndiElement::UnknownSwitchState ) );
    return newCallBack_m_indiP_enable( p );
}

inline int nnReconstructor_test::invalidSelection( bool multiple )
{
    pcf::IndiProperty p( pcf::IndiProperty::Switch );
    p.setName( "model" );
    p.add( pcf::IndiElement( modelId( 0 ), pcf::IndiElement::On ) );
    p.add( pcf::IndiElement( multiple ? modelId( 1 ) : "unconfigured", pcf::IndiElement::On ) );
    return newCallBack_m_indiP_model( p );
}

inline bool nnReconstructor_test::acknowledged()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_enableReadback;
}

inline bool nnReconstructor_test::disabling()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_disabling;
}

inline bool nnReconstructor_test::loading()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_loading;
}

inline std::string nnReconstructor_test::loaded()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_loadedModel;
}

inline std::string nnReconstructor_test::requested()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_requestedModel;
}

inline void nnReconstructor_test::expire()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    m_transitionStart -= std::chrono::seconds( 60 );
}

inline void nnReconstructor_test::control()
{
    updateControls();
}

inline void nnReconstructor_test::replaceOutput()
{
    std::remove( ".build/test.im.shm" );
    std::ofstream( ".build/test.im.shm" ) << "replacement";
    --frameGrabberT::m_inode; // Deterministic inode mismatch even on immediate inode reuse.
    m_outputCheck = {};
}

inline void nnReconstructor_test::badOutput()
{
    frameGrabberT::md.location = 0;
}

inline void nnReconstructor_test::reattachOutput()
{
    configureAcquisition();
}

inline void nnReconstructor_test::rawFrame()
{
    processImage( m_frame.data(), dev::shmimT{} );
}

inline bool nnReconstructor_test::waitLoad()
{
    updateControls();
    for( int i = 0; i < 1000; ++i )
    {
        if( !loading() )
        {
            updateControls();
            return true;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
    return false;
}

inline void nnReconstructor_test::frame()
{
    ++shmimMonitorT::m_imageStream.md->cnt0;
    processImage( m_frame.data(), dev::shmimT{} );
}

inline void nnReconstructor_test::reconnect()
{
    allocate( dev::shmimT{} );
}

inline void nnReconstructor_test::wrongSource()
{
    shmimMonitorT::md.datatype = 1;
    reconnect();
}

inline bool nnReconstructor_test::pending()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_updated;
}

inline std::string nnReconstructor_test::fault()
{
    std::lock_guard<std::mutex> l( m_resultMutex );
    return m_fault;
}

inline void *nnReconstructor_test::input()
{
    return pp_image;
}

inline bool nnReconstructor_test::claim()
{
    return acquireAndCheckValid() == 0;
}

inline std::vector<float> nnReconstructor_test::copy()
{
    std::vector<float> v( outputSize );
    loadImageIntoStream( v.data() );
    return v;
}

inline void nnReconstructor_test::complete()
{
    frameGrabberPostPublish( nullptr );
}

inline timespec nnReconstructor_test::timestamp()
{
    return frameGrabberT::m_currImageTimestamp;
}

/** \endcond */
} // namespace MagAOX::app
namespace libXWCTest::nnReconstructorTest
{
using MagAOX::app::nnReconstructor_test;

/** \brief A claimed result and its timestamp remain stable while the producer advances. */
TEST_CASE( "publication snapshot survives a concurrent inference", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::processImage;
    MagAOX::app::nnReconstructor::acquireAndCheckValid;
    MagAOX::app::nnReconstructor::loadImageIntoStream;
    MagAOX::app::nnReconstructor::frameGrabberPostPublish;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    app.frame();
    REQUIRE( app.claim() );
    const auto time = app.timestamp();
    nnTest::result  = 2;
    std::thread producer( [&] { app.frame(); } );
    producer.join();
    REQUIRE( app.copy() == std::vector<float>{ 1, 1, 1 } );
    REQUIRE( app.timestamp().tv_sec == time.tv_sec );
    REQUIRE( app.timestamp().tv_nsec == time.tv_nsec );
    app.complete();
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 2, 2, 2 } );
    app.complete();
}

/** \brief Reconnection invalidates pending results without reallocating inference buffers. */
TEST_CASE( "WFS reconnect reuses buffers and validates the attachment", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::allocate;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    auto pointer = app.input();
    for( int i = 0; i < 100; ++i )
    {
        app.frame();
        app.reconnect();
        REQUIRE_FALSE( app.pending() );
        REQUIRE( app.input() == pointer );
    }
    app.wrongSource();
    REQUIRE_FALSE( app.fault().empty() );
    app.frame();
    REQUIRE_FALSE( app.pending() );
}

/** \brief Every failed CUDA transfer or execution prevents a fresh result notification. */
TEST_CASE( "failed inference never queues a result", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::processImage;
    MagAOX::app::nnReconstructor::fail;
#endif
    // clang-format on

    for( int failure = 0; failure < 3; ++failure )
    {
        nnReconstructor_test app;
        app.prepare();
        if( failure < 2 )
            nnTest::failCopy = failure + 1;
        if( failure == 2 )
            nnTest::executeFailure = true;
        app.frame();
        REQUIRE_FALSE( app.pending() );
        REQUIRE_FALSE( app.fault().empty() );
    }
}

/** \brief NaN and infinity never become pending results, including with production optimization flags. */
TEST_CASE( "non-finite FP32 output is rejected", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::infer;
    MagAOX::app::nnReconstructor::processImage;
#endif
    // clang-format on
    // Inject IEEE storage bits: floating-point NaN/Inf expressions are not reliable under fast-math.
    for( uint32_t bits : { 0x7fc00000U, 0xffc00000U, 0x7f800000U, 0xff800000U } )
    {
        nnReconstructor_test app;
        app.prepare();
        nnTest::result = std::bit_cast<float>( bits );
        app.frame();
        REQUIRE_FALSE( app.pending() );
        REQUIRE( app.fault() == "non-finite modal output" );
    }
}

/** \brief Validate supplied configuration values without relying on floating-point optimizer assumptions. */
TEST_CASE( "configuration rejects non-finite normalization and deadlines", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::loadConfigImpl;
#endif
    // clang-format on
    for( uint32_t bits : { 0x7fc00000U, 0x7f800000U, 0xff800000U } )
    {
        nnReconstructor_test app;
        REQUIRE( app.validateConfig( std::bit_cast<float>( bits ), 30 ) < 0 );
    }
    for( uint64_t bits :
         { 0x7ff8000000000000ULL, 0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0ULL, 0xbff0000000000000ULL } )
    {
        nnReconstructor_test app;
        REQUIRE( app.validateConfig( 2, std::bit_cast<double>( bits ) ) < 0 );
    }
    // Preserve the existing acceptance of finite zero and negative normalization.
    for( float value : { 0.F, -2.F, 2.F, std::numeric_limits<float>::max() } )
    {
        nnReconstructor_test app;
        REQUIRE( app.validateConfig( value, 30 ) == 0 );
    }
}

/** \brief Missing files and partial startup allocations are rejected and fully released. */
TEST_CASE( "loading failures unwind every owned resource", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::load_engine;
    MagAOX::app::nnReconstructor::create_engine_context;
    MagAOX::app::nnReconstructor::appShutdown;
#endif
    // clang-format on

    for( int failure = 0; failure < 6; ++failure )
    {
        {
            nnReconstructor_test app;
            if( failure == 0 )
                REQUIRE_THROWS( app.load_engine( "no-such-engine.trt" ) );
            else
            {
                nnTest::runtimeFailure = failure == 1;
                nnTest::engineFailure  = failure == 2;
                nnTest::contextFailure = failure == 3;
                if( failure > 3 )
                    nnTest::failAllocation = failure - 3;
                REQUIRE_THROWS( app.prepare() );
            }
            app.appShutdown();
            app.appShutdown();
        }
        REQUIRE( nnTest::liveAllocations == 0 );
        REQUIRE( nnTest::runtimes == 0 );
        REQUIRE( nnTest::engines == 0 );
        REQUIRE( nnTest::contexts == 0 );
    }
}

/** \brief Tensor contracts reject bad rank, dtype, batch, layout and unresolved dimensions. */
TEST_CASE( "incompatible engine contracts fail before inference", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::validateEngine;
#endif
    // clang-format on

    for( int failure = 0; failure < 6; ++failure )
    {
        nnReconstructor_test app;
        auto                &t = nvinfer1::tensors[0];
        if( failure == 0 )
            t.dims.d[0] = 2;
        if( failure == 1 )
            t.dims.d[2] = -1;
        if( failure == 2 )
            t.type = nvinfer1::DataType::kHALF;
        if( failure == 3 )
            t.format = nvinfer1::TensorFormat::kOTHER;
        if( failure == 4 )
            t.dims.nbDims = 2;
        if( failure == 5 )
            t.location = nvinfer1::TensorLocation::kHOST;
        REQUIRE_THROWS( app.prepare() );
    }
}

/** \brief Preserve the existing crop order, padding and FP32 scaling. */
TEST_CASE( "preprocessing retains its exact tensor values", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::infer;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    app.frame();
    std::vector<float> expected( 144, 0 );
    for( int pupil = 0; pupil < 4; ++pupil )
        for( int x = 0; x < 2; ++x )
            for( int y = 0; y < 2; ++y )
                expected[pupil * 36 + ( x + 2 ) * 6 + y + 2] =
                    2 * ( ( pupil / 2 ) * 2 + y + ( ( pupil % 2 ) * 2 + x ) * 4 );
    REQUIRE( nnTest::uploaded == expected );
}

/** \brief A waiting publisher is woken by shutdown without publishing a stale result. */
TEST_CASE( "shutdown wakes an idle publisher", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::appShutdown;
    MagAOX::app::nnReconstructor::acquireAndCheckValid;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    auto waiter = std::async( std::launch::async, [&] { return app.claim(); } );
    app.appShutdown();
    REQUIRE( waiter.wait_for( std::chrono::milliseconds( 200 ) ) == std::future_status::ready );
    REQUIRE_FALSE( waiter.get() );
}
/** \brief Real producer and consumer threads must never publish mixed vectors under sustained overlap. */
TEST_CASE( "concurrent handoff stress", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::processImage;
    MagAOX::app::nnReconstructor::acquireAndCheckValid;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    std::atomic<bool> done{ false };
    std::atomic<int>  mixed{ 0 }, publications{ 0 };
    std::thread       publisher(
        [&]
        {
            while( !done || app.pending() )
            {
                if( !app.claim() )
                    continue;
                const auto values = app.copy();
                if( values[0] != values[1] || values[1] != values[2] )
                    ++mixed;
                ++publications;
                app.complete();
            }
        } );
    for( int i = 1; i < 4000; ++i )
    {
        nnTest::result = i;
        app.frame();
    }
    done = true;
    publisher.join();
    REQUIRE( mixed == 0 );
    REQUIRE( publications > 0 );
}

/** \brief Disabling discards pending work and acknowledges only after the zero publication hook. */
TEST_CASE( "disable publishes one zero after the in-flight publication", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_enable;
    MagAOX::app::nnReconstructor::frameGrabberPostPublish;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    app.frame();
    REQUIRE( app.claim() );
    app.frame();
    REQUIRE( app.pending() );
    REQUIRE( app.enable( false ) == 0 );
    REQUIRE_FALSE( app.pending() );
    REQUIRE( app.acknowledged() );
    REQUIRE( app.select( 1 ) == -1 );
    REQUIRE( app.enable( true ) == -1 );
    REQUIRE( app.copy() == std::vector<float>{ 1, 1, 1 } );
    app.complete();
    REQUIRE( app.acknowledged() );
    REQUIRE( app.disabling() );
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 0, 0, 0 } );
    REQUIRE( app.acknowledged() );
    app.complete();
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE_FALSE( app.disabling() );
    app.frame();
    REQUIRE_FALSE( app.pending() );
    REQUIRE( app.enable( false ) == 0 );
    REQUIRE_FALSE( app.disabling() );
}

/** \brief Disable cannot publish its zero or acknowledge while an admitted inference is still running. */
TEST_CASE( "disable waits for admitted inference and discards its late result", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::processImage;
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_enable;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    std::promise<void> entered, release;
    auto               gate = release.get_future().share();
    nnTest::duringExecute   = [&]
    {
        entered.set_value();
        gate.wait();
    };
    auto producer = std::async( std::launch::async, [&] { app.frame(); } );
    entered.get_future().wait();
    const int  disabled     = app.enable( false );
    auto       publisher    = std::async( std::launch::async, [&] { return app.claim(); } );
    const auto before       = publisher.wait_for( std::chrono::milliseconds( 20 ) );
    const bool acknowledged = app.acknowledged();
    release.set_value();
    producer.get();
    REQUIRE( disabled == 0 );
    REQUIRE( before == std::future_status::timeout );
    REQUIRE( acknowledged );
    REQUIRE( publisher.get() );
    REQUIRE( app.copy() == std::vector<float>{ 0, 0, 0 } );
    app.complete();
    REQUIRE_FALSE( app.pending() );
    REQUIRE_FALSE( app.acknowledged() );
}

/** \brief Start-disabled is passive; enabling waits across a newly observed source-frame boundary. */
TEST_CASE( "start-disabled and re-enable do not replay old results", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::appStartup;
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_enable;
    MagAOX::app::nnReconstructor::processImage;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare( false );
    app.frame();
    REQUIRE_FALSE( app.pending() );
    REQUIRE_FALSE( app.disabling() );
    REQUIRE( app.enable( true ) == 0 );
    app.frame();
    REQUIRE_FALSE( app.pending() );
    app.frame();
    REQUIRE( app.pending() );
    REQUIRE( app.claim() );
    app.complete();
}

/** \brief Failed selection keeps the old model installed and only an explicit successful reload clears faults. */
TEST_CASE( "model replacement and fault recovery never activate a model", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::replaceModel;
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_model;
    MagAOX::app::nnReconstructor::fail;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare( false );
    std::filesystem::remove( ".build/catalog/missing.trt" );
    REQUIRE( app.select( 2 ) == 0 );
    REQUIRE( app.waitLoad() );
    REQUIRE( app.loaded() == "first.trt" );
    REQUIRE( app.requested() == "missing.trt" );
    REQUIRE_FALSE( app.fault().empty() );
    REQUIRE( app.enable( true ) == -1 );
    REQUIRE( app.select( 1 ) == 0 );
    REQUIRE( app.waitLoad() );
    REQUIRE( app.loaded() == "second.trt" );
    REQUIRE( app.fault().empty() );
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE( app.enable( true ) == 0 );
    app.frame();
    app.frame();
    nnTest::executeFailure = true;
    app.frame();
    app.control();
    REQUIRE_FALSE( app.pending() );
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE( app.enable( true ) == -1 );
    nnTest::executeFailure = false;
    REQUIRE( app.select( 1 ) == 0 );
    REQUIRE( app.waitLoad() );
    REQUIRE( app.fault().empty() );
    REQUIRE_FALSE( app.acknowledged() );
}

/** \brief Timeout leaves an unfinished disable explicit and never grants a premature model switch. */
TEST_CASE( "missing zero completion times out without acknowledging disabled", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::updateControls;
    MagAOX::app::nnReconstructor::frameGrabberPostPublish;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    REQUIRE( app.enable( false ) == 0 );
    app.expire();
    app.control();
    REQUIRE_FALSE( app.fault().empty() );
    REQUIRE( app.acknowledged() );
    REQUIRE( app.disabling() );
    REQUIRE( app.select( 1 ) == -1 );
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 0, 0, 0 } );
    app.complete();
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE( app.enable( true ) == -1 );
    REQUIRE( app.select( 0 ) == 0 );
    REQUIRE( app.waitLoad() );
    REQUIRE( app.fault().empty() );
}

/** \brief A replacement with a different modal count must not replace the validated live resources. */
TEST_CASE( "replacement compatibility failure preserves the loaded model", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::validateEngine;
    MagAOX::app::nnReconstructor::replaceModel;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare( false );
    nvinfer1::tensors[1].dims.d[1] = 4;
    REQUIRE( app.select( 1 ) == 0 );
    REQUIRE( app.waitLoad() );
    REQUIRE( app.loaded() == "first.trt" );
    REQUIRE_FALSE( app.fault().empty() );
}

/** \brief Delayed deserialization cannot install its candidate after the transition deadline. */
TEST_CASE( "timed-out background load never silently replaces the engine", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::replaceModel;
    MagAOX::app::nnReconstructor::updateControls;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare( false );
    std::promise<void> entered, release;
    auto               gate   = release.get_future().share();
    nnTest::duringDeserialize = [&]
    {
        entered.set_value();
        gate.wait();
    };
    REQUIRE( app.select( 1 ) == 0 );
    app.control();
    entered.get_future().wait();
    app.expire();
    app.control();
    const bool stillLoading = app.loading();
    const int  enable       = app.enable( true );
    release.set_value();
    REQUIRE( app.waitLoad() );
    REQUIRE( stillLoading );
    REQUIRE( enable == -1 );
    REQUIRE( app.loaded() == "first.trt" );
    REQUIRE_FALSE( app.fault().empty() );
}

/** \brief Output changes stop admission and cannot complete a requested disable on an invalid mapping. */
TEST_CASE( "output identity and location failures do not acknowledge a zero", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::verifyOutput;
    MagAOX::app::nnReconstructor::acquireAndCheckValid;
#endif
    // clang-format on

    for( bool replaced : { false, true } )
    {
        nnReconstructor_test app;
        app.prepare();
        REQUIRE( app.enable( false ) == 0 );
        if( replaced )
            app.replaceOutput();
        else
            app.badOutput();
        REQUIRE_FALSE( app.claim() );
        REQUIRE( app.acknowledged() );
        REQUIRE( app.disabling() );
        REQUIRE_FALSE( app.fault().empty() );
    }
}

/** \brief A reconnect during disable does not discard the requested zero or replay a prior result. */
TEST_CASE( "reconnect preserves a pending disable zero", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::allocate;
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_enable;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    app.frame();
    REQUIRE( app.enable( false ) == 0 );
    app.reconnect();
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 0, 0, 0 } );
    app.complete();
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE( app.enable( true ) == 0 );
    app.frame();
    REQUIRE_FALSE( app.pending() );
    app.rawFrame();
    REQUIRE_FALSE( app.pending() );
    app.frame();
    REQUIRE( app.pending() );
}

/** \brief Tensor roles, rather than list position, select the device binding slots. */
TEST_CASE( "reordered engine tensor roles retain the same preprocessing", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::validateEngine;
    MagAOX::app::nnReconstructor::infer;
#endif
    // clang-format on

    nnReconstructor_test app;
    std::swap( nvinfer1::tensors[0], nvinfer1::tensors[1] );
    app.prepare();
    app.frame();
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 1, 1, 1 } );
    app.complete();
    REQUIRE( nnTest::uploaded.size() == 144 );
}

/** \brief Shutdown waits for the real loader to return and prevents its late candidate installation. */
TEST_CASE( "shutdown cancels a background replacement without freeing live resources", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::appShutdown;
    MagAOX::app::nnReconstructor::replaceModel;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare( false );
    std::promise<void> entered, release;
    auto               gate   = release.get_future().share();
    nnTest::duringDeserialize = [&]
    {
        entered.set_value();
        gate.wait();
    };
    REQUIRE( app.select( 1 ) == 0 );
    app.control();
    entered.get_future().wait();
    auto       shutdown = std::async( std::launch::async, [&] { app.appShutdown(); } );
    const auto before   = shutdown.wait_for( std::chrono::milliseconds( 20 ) );
    release.set_value();
    shutdown.get();
    REQUIRE( before == std::future_status::timeout );
    REQUIRE( app.loaded() == "first.trt" );
    REQUIRE( nnTest::liveAllocations == 0 );
    REQUIRE( nnTest::contexts == 0 );
    REQUIRE( nnTest::engines == 0 );
    REQUIRE( nnTest::runtimes == 0 );
}

/** \brief Invalid operator messages must not be interpreted as a disable or model request. */
TEST_CASE( "invalid control requests leave the admitted state unchanged", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_enable;
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_model;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    REQUIRE( app.invalidEnable() == -1 );
    REQUIRE_FALSE( app.disabling() );
    REQUIRE( app.acknowledged() );
    REQUIRE( app.invalidSelection( true ) == -1 );
    REQUIRE( app.invalidSelection( false ) == -1 );
    REQUIRE_FALSE( app.loading() );
    REQUIRE( app.requested() == "first.trt" );
}

/** \brief Removed I/O contracts fail at startup and replacement without losing the valid model. */
TEST_CASE( "auxiliary inputs and FP16 I/O are rejected", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::validateEngine;
    MagAOX::app::nnReconstructor::replaceModel;
    MagAOX::app::nnReconstructor::appShutdown;
#endif
    // clang-format on

    for( int failure = 0; failure < 3; ++failure )
        for( bool replacement : { false, true } )
        {
            {
                nnReconstructor_test app;
                if( replacement )
                    app.prepare( false );
                if( failure == 0 )
                    nvinfer1::tensors.push_back( { "extra", nvinfer1::TensorIOMode::kINPUT, { 2, { 1, 4 } } } );
                else
                    nvinfer1::tensors[failure - 1].type = nvinfer1::DataType::kHALF;
                if( replacement )
                {
                    REQUIRE( app.select( 1 ) == 0 );
                    REQUIRE( app.waitLoad() );
                    REQUIRE( app.loaded() == "first.trt" );
                    REQUIRE_FALSE( app.fault().empty() );
                    REQUIRE_FALSE( app.acknowledged() );
                    REQUIRE( app.enable( true ) == -1 );
                }
                else
                    REQUIRE_THROWS( app.prepare() );
                app.appShutdown();
                app.appShutdown();
            }
            REQUIRE( nnTest::liveAllocations == 0 );
            REQUIRE( nnTest::runtimes == 0 );
            REQUIRE( nnTest::engines == 0 );
            REQUIRE( nnTest::contexts == 0 );
        }
}

/** \brief An inference failure while disabling must release admission and still complete the requested zero. */
TEST_CASE( "inference failure cannot strand a disable transition", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::processImage;
    MagAOX::app::nnReconstructor::fail;
    MagAOX::app::nnReconstructor::frameGrabberPostPublish;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    std::promise<void> entered, release;
    auto               gate = release.get_future().share();
    nnTest::duringExecute   = [&]
    {
        entered.set_value();
        gate.wait();
    };
    nnTest::executeFailure = true;
    auto producer          = std::async( std::launch::async, [&] { app.frame(); } );
    entered.get_future().wait();
    const int result = app.enable( false );
    release.set_value();
    producer.get();
    REQUIRE( result == 0 );
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 0, 0, 0 } );
    app.complete();
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE_FALSE( app.fault().empty() );
    REQUIRE( app.enable( true ) == -1 );
}

/** \brief Shutdown cancels unclaimed disable work without manufacturing a completed zero acknowledgement. */
TEST_CASE( "shutdown does not falsely acknowledge a pending zero", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::appShutdown;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    REQUIRE( app.enable( false ) == 0 );
    app.appShutdown();
    REQUIRE( app.acknowledged() );
    REQUIRE( app.disabling() );
    REQUIRE_FALSE( app.claim() );
}

/** \brief A timestamp failure keeps disable incomplete and permits an explicit Off retry without spinning. */
TEST_CASE( "zero timestamp failure is recoverable without false acknowledgement", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::acquireAndCheckValid;
    MagAOX::app::nnReconstructor::newCallBack_m_indiP_enable;
#endif
    // clang-format on

    nnReconstructor_test app;
    app.prepare();
    REQUIRE( app.enable( false ) == 0 );
    nnTest::failClock = true;
    REQUIRE_FALSE( app.claim() );
    REQUIRE( app.acknowledged() );
    REQUIRE( app.disabling() );
    REQUIRE_FALSE( app.fault().empty() );
    nnTest::failClock = false;
    REQUIRE( app.enable( false ) == 0 );
    REQUIRE( app.claim() );
    REQUIRE( app.copy() == std::vector<float>{ 0, 0, 0 } );
    app.complete();
    REQUIRE_FALSE( app.acknowledged() );
    REQUIRE( app.enable( true ) == -1 );
}

/** \brief Startup scans only engine files, validates each contract and leaves rejected resources released. */
TEST_CASE( "startup discovers compatible engines without an explicit list", "[nnReconstructor]" )
{
    // clang-format off
#ifdef NNRECONSTRUCTOR_TEST_DOXYGEN_REF
    MagAOX::app::nnReconstructor::discoverModels;
    MagAOX::app::nnReconstructor::appStartup;
    MagAOX::app::nnReconstructor::replaceModel;
#endif
    // clang-format on
    {
        nnReconstructor_test app;
        auto                 valid      = nvinfer1::tensors;
        auto                 wrongCount = valid;
        wrongCount[1].dims.d[1]         = 4;
        auto half                       = valid;
        half[0].type                    = nvinfer1::DataType::kHALF;
        auto extra                      = valid;
        extra.push_back( { "extra", nvinfer1::TensorIOMode::kINPUT, { 2, { 1, 4 } } } );
        nvinfer1::serializedTensors = {
            { "broken", {} }, { "wrong", wrongCount }, { "half", half }, { "extra", extra } };
        for( const auto &[name, contents] : std::map<std::string, std::string>{ { "z.engine", "valid" },
                                                                                { "a.plan", "valid" },
                                                                                { "wrong.trt", "wrong" },
                                                                                { "half.trt", "half" },
                                                                                { "extra.trt", "extra" },
                                                                                { "broken.trt", "broken" },
                                                                                { "empty.trt", "" },
                                                                                { "notes.txt", "valid" },
                                                                                { "upload.trt.partial", "valid" } } )
            std::ofstream( ".build/catalog/" + name ) << contents;
        std::filesystem::create_directory( ".build/catalog/directory.trt" );
        app.prepare( false, true );
        REQUIRE( app.models() == std::vector<std::string>{ "first.trt", "a.plan", "second.trt", "z.engine" } );
        REQUIRE( nnTest::engines == 1 );
        REQUIRE( nnTest::contexts == 1 );
        REQUIRE( app.loaded() == "first.trt" );
        REQUIRE_FALSE( app.acknowledged() );
        std::ofstream( ".build/catalog/later.trt" ) << "valid";
        app.control();
        REQUIRE( app.models().size() == 4 ); // Startup only: no live catalog mutation.
        REQUIRE( app.select( 1 ) == 0 );
        REQUIRE( app.waitLoad() );
        REQUIRE( app.loaded() == "a.plan" );
        REQUIRE_FALSE( app.acknowledged() );
    }
    REQUIRE( nnTest::liveAllocations == 0 );
    REQUIRE( nnTest::runtimes == 0 );
    REQUIRE( nnTest::engines == 0 );
    REQUIRE( nnTest::contexts == 0 );
}
} // namespace libXWCTest::nnReconstructorTest
