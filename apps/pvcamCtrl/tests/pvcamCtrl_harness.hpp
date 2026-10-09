/** \file pvcamCtrl_harness.hpp
 * \brief Offline PVCAM library, sysfs, and dependency doubles shared by the pvcamCtrl unit tests.
 *
 * \ingroup pvcamCtrl_unit_test
 */

#ifndef pvcamCtrl_harness_hpp
#define pvcamCtrl_harness_hpp

#include "../../../tests/testXWC.hpp"
#include "../../../tests/outletAppTest.hpp"

#include <map>
#include <set>
#include <sys/resource.h>

#include <master.h>
#include <pvcam.h>

#include "../pvcamPcie.hpp"

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace pvcamHarness
{

/// One scripted PVCAM camera.
struct Camera
{
    /// PVCAM camera name.
    std::string m_name;

    /// Alphanumeric serial number.
    std::string m_serial;

    /// Whether the serial number parameter is available.
    bool m_hasSerial{ true };
};

/// Scripted PVCAM library state and fault injection.
struct Fake
{
    /// Cameras reported by enumeration.
    std::vector<Camera> m_cameras;

    /// Whether pl_pvcam_init has been called without a matching uninit.
    bool m_initialized{ false };

    /// Open handles, mapped to camera indices.
    std::map<int16, size_t> m_open;

    /// Code returned by pl_error_code.
    int16 m_errorCode{ 0 };

    /// Call counts by key.
    std::map<std::string, unsigned> m_calls;

    /// Fail the Nth call of a key.
    std::map<std::string, unsigned> m_failAt;

    /// Fail every call of a key.
    std::set<std::string> m_failAlways;

    /// Parameter values by (parameter, attribute).
    std::map<std::pair<uns32, int16>, long long> m_values;

    /// Readout-port enumeration names.
    std::vector<std::string> m_ports{ "Sensitivity", "Speed" };

    /// Successful pl_set_param calls as (parameter, value).
    std::vector<std::pair<uns32, long long>> m_sets;

    /// Bytes per frame reported by pl_exp_setup_cont.
    uns32 m_frameBytes{ 8 };

    /// Exposure time passed to the last pl_exp_setup_cont.
    uns32 m_setupExposure{ 0 };

    /// Frame returned by pl_exp_get_latest_frame.
    std::vector<uint8_t> m_frame = std::vector<uint8_t>( 64, 7 );

    /// Registered end-of-frame callback context.
    void *m_context{ nullptr };

    /// Calls of the acquisition functions.
    std::vector<std::string> m_acq;

    /// Fail the named libc wrapper on every call.
    std::set<std::string> m_libcFail;

    /// Fail the Nth call of the named libc wrapper.
    std::map<std::string, unsigned> m_libcFailAt;

    /// Calls of each libc wrapper.
    std::map<std::string, unsigned> m_libcCalls;

    /// Whether threadStart starts real threads.
    bool m_startThreads{ false };

    /// Results of the shutter double's appStartup, appLogic, appShutdown, onPowerOff, whilePowerOff, set.
    std::array<int, 6> m_shutter{};

    /// Shutter states requested through the shutter double.
    std::vector<int> m_shutterStates;

    /// Called after each app state change, e.g. to simulate a concurrent power change.
    std::function<void( int )> m_onState;
};

/// Current test's fake state.
inline Fake g_fake;

/// Stops idle placeholder threads.
inline std::atomic<bool> g_stopIdle{ false };

/// Key for a pl_get_param call.
inline std::string getKey( uns32 param /**< [in] parameter */, int16 attr /**< [in] attribute */ )
{
    return "get:" + std::to_string( param ) + ":" + std::to_string( attr );
}

/// Key for a pl_set_param call.
inline std::string setKey( uns32 param /**< [in] parameter */ )
{
    return "set:" + std::to_string( param );
}

/// Count a call and decide whether it fails.
inline bool fails( const std::string &key /**< [in] call key */ )
{
    unsigned n = ++g_fake.m_calls[key];
    if( g_fake.m_failAlways.count( key ) || ( g_fake.m_failAt.count( key ) && g_fake.m_failAt[key] == n ) )
    {
        g_fake.m_errorCode = 1;
        return true;
    }
    return false;
}

/// Size of a parameter's value.
inline size_t paramSize( uns32 param /**< [in] parameter */, int16 attr /**< [in] attribute */ )
{
    if( attr == ATTR_AVAIL )
        return sizeof( rs_bool );
    if( attr == ATTR_COUNT )
        return sizeof( uns32 );
    switch( ( param >> 24 ) & 0xFF )
    {
    case TYPE_INT16:
    case TYPE_UNS16:
        return 2;
    case TYPE_UNS64:
    case TYPE_INT64:
        return 8;
    default: // TYPE_ENUM, and PARAM_READOUT_TIME whose actual type is uns32
        return 4;
    }
}

/// Fail selected libc wrappers.
inline bool libcFails( const std::string &name /**< [in] wrapper name */ )
{
    unsigned n = ++g_fake.m_libcCalls[name];
    if( g_fake.m_libcFail.count( name ) || ( g_fake.m_libcFailAt.count( name ) && g_fake.m_libcFailAt[name] == n ) )
    {
        errno = EINVAL;
        return true;
    }
    return false;
}

/// Fault-injectable sem_init.
inline int semInit( sem_t *sem, int pshared, unsigned value )
{
    return libcFails( "sem_init" ) ? -1 : ::sem_init( sem, pshared, value );
}

/// Fault-injectable sem_trywait.
inline int semTrywait( sem_t *sem )
{
    return libcFails( "sem_trywait" ) ? -1 : ::sem_trywait( sem );
}

/// Fault-injectable sem_post.
inline int semPost( sem_t *sem )
{
    return libcFails( "sem_post" ) ? -1 : ::sem_post( sem );
}

/// Fault-injectable clock_gettime.
inline int clockGettime( clockid_t clk, timespec *ts )
{
    return libcFails( "clock_gettime" ) ? -1 : ::clock_gettime( clk, ts );
}

/// Write a hexadecimal sysfs attribute.
inline void attribute( const std::string &path /**< [in] file */, const std::string &text /**< [in] contents */ )
{
    std::ofstream( path ) << text << "\n";
}

/// Little-endian config-space image.
typedef std::array<uint8_t, 256> Config;

/// Store a little-endian value in a config image.
inline void put( Config  &c /**< [in/out] image */,
                 size_t   off /**< [in] offset */,
                 uint32_t v /**< [in] value */,
                 size_t   n /**< [in] bytes */ )
{
    for( size_t i = 0; i < n; ++i )
        c[off + i] = ( v >> ( 8 * i ) ) & 0xFF;
}

/// Write a config image to a device directory.
inline void writeConfig( const std::string &dev /**< [in] device dir */, const Config &c /**< [in] image */ )
{
    std::ofstream( dev + "/config", std::ios::binary ).write( reinterpret_cast<const char *>( c.data() ), c.size() );
}

/// Read a device's config image.
inline Config readConfig( const std::string &dev /**< [in] device dir */ )
{
    Config c{};
    std::ifstream( dev + "/config", std::ios::binary ).read( reinterpret_cast<char *>( c.data() ), c.size() );
    return c;
}

/// Condition of a fake camera function.
enum class Cam
{
    Healthy,
    Stale,
    Unresponsive
};

/// A fake sysfs PCI tree in a private directory.
struct Sysfs
{
    /// Root of the tree.
    std::string m_root;

    /// Create the tree below a fixture directory.
    explicit Sysfs( const std::string &dir /**< [in] fixture directory */ ) : m_root( dir + "/pci" )
    {
        std::filesystem::create_directories( m_root );
    }

    /// Create a downstream port.
    void port( const std::string &bdf /**< [in] port address */,
               bool               up      = true /**< [in] link active */,
               bool               capable = true /**< [in] reports link activity */,
               uint16_t           device  = 0x8733 /**< [in] device ID */ )
    {
        std::string dir = m_root + "/" + bdf;
        std::filesystem::create_directories( dir );
        attribute( dir + "/vendor", "0x10b5" );
        attribute( dir + "/device", "0x" + hex( device ) );
        attribute( dir + "/rescan", "" );
        Config c{};
        put( c, 0x06, 0x10, 2 );
        put( c, 0x34, 0x40, 1 );
        put( c, 0x3E, 0x0003, 2 );
        put( c, 0x40, 0x10, 1 );
        put( c, 0x4C, capable ? ( 1u << 20 ) : 0, 4 );
        put( c, 0x52, up ? 0x2000 : 0, 2 );
        writeConfig( dir, c );
    }

    /// Set a port's link state.
    void link( const std::string &bdf /**< [in] port */, bool up /**< [in] link active */ )
    {
        Config c = readConfig( m_root + "/" + bdf );
        put( c, 0x52, up ? 0x2000 : 0, 2 );
        writeConfig( m_root + "/" + bdf, c );
    }

    /// Create or update a camera function below a port.
    void camera( const std::string &port /**< [in] port */,
                 const std::string &bdf /**< [in] camera address */,
                 Cam                state = Cam::Healthy /**< [in] condition */ )
    {
        std::string dir = m_root + "/" + port + "/" + bdf;
        std::filesystem::create_directories( dir );
        attribute( dir + "/vendor", "0x1b6b" );
        attribute( dir + "/device", "0x0001" );
        attribute( dir + "/remove", "" );
        attribute( dir + "/resource", "0x00000001fc000000 0x00000001fcffffff 0x0000000000140204" );
        Config c{};
        c.fill( state == Cam::Unresponsive ? 0xFF : 0 );
        if( state != Cam::Unresponsive )
        {
            put( c, 0x00, 0x1b6b, 2 );
            put( c, 0x02, 0x0001, 2 );
            if( state == Cam::Healthy )
            {
                put( c, 0x04, 0x0006, 2 );
                put( c, 0x10, 0xfc00000c, 4 );
                put( c, 0x14, 0x1, 4 );
            }
        }
        writeConfig( dir, c );
    }

    /// Check whether a camera function exists.
    bool has( const std::string &port /**< [in] port */, const std::string &bdf /**< [in] camera */ )
    {
        return std::filesystem::exists( m_root + "/" + port + "/" + bdf );
    }

    /// Format a 16-bit ID.
    static std::string hex( uint16_t v /**< [in] value */ )
    {
        char b[8];
        snprintf( b, sizeof( b ), "%04x", v );
        return b;
    }
};

} // namespace pvcamHarness

extern "C"
{
    rs_bool pl_pvcam_init( void )
    {
        if( pvcamHarness::fails( "pl_pvcam_init" ) )
            return PV_FAIL;
        pvcamHarness::g_fake.m_initialized = true;
        return PV_OK;
    }

    rs_bool pl_pvcam_uninit( void )
    {
        auto &f = pvcamHarness::g_fake;
        if( pvcamHarness::fails( "pl_pvcam_uninit" ) )
            return PV_FAIL;
        if( !f.m_initialized )
        {
            f.m_errorCode = 157; // PL_ERR_LIBRARY_NOT_INITIALIZED
            return PV_FAIL;
        }
        f.m_initialized = false;
        f.m_open.clear();
        return PV_OK;
    }

    rs_bool pl_cam_close( int16 hcam )
    {
        if( pvcamHarness::fails( "pl_cam_close" ) )
            return PV_FAIL;
        pvcamHarness::g_fake.m_open.erase( hcam );
        return PV_OK;
    }

    rs_bool pl_cam_get_name( int16 cam_num, char *camera_name )
    {
        if( pvcamHarness::fails( "pl_cam_get_name" ) )
            return PV_FAIL;
        strncpy( camera_name, pvcamHarness::g_fake.m_cameras.at( cam_num ).m_name.c_str(), CAM_NAME_LEN - 1 );
        return PV_OK;
    }

    rs_bool pl_cam_get_total( int16 *totl_cams )
    {
        if( pvcamHarness::fails( "pl_cam_get_total" ) )
            return PV_FAIL;
        *totl_cams = pvcamHarness::g_fake.m_cameras.size();
        return PV_OK;
    }

    rs_bool pl_cam_open( char *camera_name, int16 *hcam, int16 )
    {
        auto &f = pvcamHarness::g_fake;
        if( pvcamHarness::fails( "pl_cam_open" ) )
            return PV_FAIL;
        for( size_t n = 0; n < f.m_cameras.size(); ++n )
        {
            if( f.m_cameras[n].m_name == camera_name )
            {
                *hcam           = 100 + n;
                f.m_open[*hcam] = n;
                return PV_OK;
            }
        }
        return PV_FAIL;
    }

    rs_bool pl_cam_register_callback_ex3( int16, int32, void *, void *context )
    {
        if( pvcamHarness::fails( "pl_cam_register_callback_ex3" ) )
            return PV_FAIL;
        pvcamHarness::g_fake.m_context = context;
        return PV_OK;
    }

    rs_bool pl_cam_deregister_callback( int16, int32 )
    {
        return pvcamHarness::fails( "pl_cam_deregister_callback" ) ? PV_FAIL : PV_OK;
    }

    int16 pl_error_code( void )
    {
        return pvcamHarness::g_fake.m_errorCode;
    }

    rs_bool pl_error_message( int16 err_code, char *msg )
    {
        snprintf( msg, ERROR_MSG_LEN, "fake error %d", err_code );
        return PV_OK;
    }

    rs_bool pl_get_param( int16 hcam, uns32 param_id, int16 param_attribute, void *param_value )
    {
        auto &f = pvcamHarness::g_fake;
        if( pvcamHarness::fails( pvcamHarness::getKey( param_id, param_attribute ) ) )
            return PV_FAIL;
        if( param_id == PARAM_HEAD_SER_NUM_ALPHA && param_attribute == ATTR_CURRENT )
        {
            strncpy( static_cast<char *>( param_value ),
                     f.m_cameras.at( f.m_open.at( hcam ) ).m_serial.c_str(),
                     MAX_ALPHA_SER_NUM_LEN - 1 );
            return PV_OK;
        }
        long long value = 0;
        if( param_id == PARAM_HEAD_SER_NUM_ALPHA )
            value = f.m_cameras.at( f.m_open.at( hcam ) ).m_hasSerial;
        else if( f.m_values.count( { param_id, param_attribute } ) )
            value = f.m_values[{ param_id, param_attribute }];
        memcpy( param_value, &value, pvcamHarness::paramSize( param_id, param_attribute ) );
        return PV_OK;
    }

    rs_bool pl_set_param( int16, uns32 param_id, void *param_value )
    {
        if( pvcamHarness::fails( pvcamHarness::setKey( param_id ) ) )
            return PV_FAIL;
        long long value = 0;
        memcpy( &value, param_value, pvcamHarness::paramSize( param_id, ATTR_CURRENT ) );
        pvcamHarness::g_fake.m_sets.push_back( { param_id, value } );
        return PV_OK;
    }

    rs_bool pl_enum_str_length( int16, uns32, uns32 index, uns32 *length )
    {
        if( pvcamHarness::fails( "pl_enum_str_length" ) )
            return PV_FAIL;
        *length = pvcamHarness::g_fake.m_ports.at( index ).size() + 1;
        return PV_OK;
    }

    rs_bool pl_get_enum_param( int16, uns32, uns32 index, int32 *value, char *desc, uns32 length )
    {
        if( pvcamHarness::fails( "pl_get_enum_param" ) )
            return PV_FAIL;
        *value = index;
        strncpy( desc, pvcamHarness::g_fake.m_ports.at( index ).c_str(), length );
        return PV_OK;
    }

    rs_bool pl_exp_setup_cont( int16, uns16, const rgn_type *, int16, uns32 exposure_time, uns32 *exp_bytes, int16 )
    {
        if( pvcamHarness::fails( "pl_exp_setup_cont" ) )
            return PV_FAIL;
        pvcamHarness::g_fake.m_setupExposure = exposure_time;
        *exp_bytes                           = pvcamHarness::g_fake.m_frameBytes;
        return PV_OK;
    }

    rs_bool pl_exp_start_cont( int16, void *, uns32 )
    {
        pvcamHarness::g_fake.m_acq.push_back( "start" );
        return pvcamHarness::fails( "pl_exp_start_cont" ) ? PV_FAIL : PV_OK;
    }

    rs_bool pl_exp_get_latest_frame( int16, void **frame )
    {
        if( pvcamHarness::fails( "pl_exp_get_latest_frame" ) )
            return PV_FAIL;
        *frame = pvcamHarness::g_fake.m_frame.data();
        return PV_OK;
    }

    rs_bool pl_exp_stop_cont( int16, int16 )
    {
        pvcamHarness::g_fake.m_acq.push_back( "stop" );
        return pvcamHarness::fails( "pl_exp_stop_cont" ) ? PV_FAIL : PV_OK;
    }
}

namespace MagAOX
{
namespace app
{

/// App base whose thread starts are suppressed unless a test enables them.
template <bool useINDI = true>
class pvcamTestApp : public outletTestApp<useINDI>
{
  public:
    using outletTestApp<useINDI>::outletTestApp;
    using outletTestApp<useINDI>::state;

    /// Change state, then run the test's state hook.
    void state( const stateCodes::stateCodeT &s, bool stateAlert = false )
    {
        MagAOXApp<useINDI>::state( s, stateAlert );
        if( pvcamHarness::g_fake.m_onState )
            pvcamHarness::g_fake.m_onState( s );
    }

    /// Start the real thread only when enabled; otherwise start an idle placeholder that keeps the handle valid.
    template <class thisPtr, class Function>
    int threadStart( std::thread       &thrd,
                     bool              &thrdInit,
                     pid_t             &tpid,
                     pcf::IndiProperty &thProp,
                     int                thrdPrio,
                     const std::string &cpuset,
                     const std::string &thrdName,
                     thisPtr           *thrdThis,
                     Function         &&thrdStart )
    {
        if( !pvcamHarness::g_fake.m_startThreads )
        {
            thrd = std::thread(
                []
                {
                    while( !pvcamHarness::g_stopIdle )
                        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
                } );
            return 0;
        }
        return MagAOXApp<useINDI>::threadStart(
            thrd, thrdInit, tpid, thProp, thrdPrio, cpuset, thrdName, thrdThis, thrdStart );
    }
};

/// Threadless PCIe helper recording operations and emulating kernel remove and rescan.
class pvcamTestPcie : public pvcamPcie
{
  public:
    /// Recorded operations, relative to the sysfs root.
    std::vector<std::string> m_ops;

    /// Camera created below the port by a rescan, or empty for none.
    std::string m_rescanCreates;

    /// Fail the operation whose record starts with this text.
    std::string m_failOp;

    /// Whether PVCAM was initialized or had a camera open at the last removal.
    bool m_openAtRemove{ false };

    /// Number of pl_pvcam_init calls before the last removal.
    unsigned m_initsAtRemove{ 0 };

    /// Expose the protected hold and settle times.
    using pvcamPcie::m_resetHoldMs;
    using pvcamPcie::m_settleMs;

    /// Record a write, then emulate the kernel's response.
    int writeAttribute( const std::string &path, const std::string &value ) override
    {
        if( record( "write " + rel( path ) ) )
            return fail( "injected" );
        int                   rv = pvcamPcie::writeAttribute( path, value );
        std::filesystem::path p( path );
        if( p.filename() == "remove" )
        {
            m_openAtRemove  = !pvcamHarness::g_fake.m_open.empty() || pvcamHarness::g_fake.m_initialized;
            m_initsAtRemove = pvcamHarness::g_fake.m_calls["pl_pvcam_init"];
            std::filesystem::remove_all( p.parent_path() );
        }
        else if( p.filename() == "rescan" && !m_rescanCreates.empty() )
            pvcamHarness::Sysfs( std::filesystem::path( m_sysfsPath ).parent_path() ).camera( m_port, m_rescanCreates );
        return rv;
    }

    /// Fail reads when selected.
    int readConfig( const std::string &device, size_t offset, uint8_t *data, size_t size ) override
    {
        if( !m_failOp.empty() && m_failOp == "read " + std::to_string( offset ) )
            return fail( "injected" );
        return pvcamPcie::readConfig( device, offset, data, size );
    }

    /// Record a config write.
    int writeConfig( const std::string &device, size_t offset, const uint8_t *data, size_t size ) override
    {
        char b[16];
        snprintf( b, sizeof( b ), "%02x%02x", data[1], data[0] );
        if( record( "config " + rel( device ) + " " + std::to_string( offset ) + " " + b ) )
            return fail( "injected" );
        return pvcamPcie::writeConfig( device, offset, data, size );
    }

    /// Record a pause without sleeping.
    void pause( unsigned ms ) override
    {
        m_ops.push_back( "pause " + std::to_string( ms ) );
    }

    /// Sleep through the production implementation.
    void realPause( unsigned ms /**< [in] time to sleep */ )
    {
        pvcamPcie::pause( ms );
    }

    /// Path relative to the sysfs root, when below it.
    std::string rel( const std::string &path /**< [in] path */ )
    {
        return path.starts_with( m_sysfsPath + "/" ) ? path.substr( m_sysfsPath.size() + 1 ) : path;
    }

    /// Record an operation, returning whether it is selected to fail.
    bool record( const std::string &op /**< [in] operation */ )
    {
        m_ops.push_back( op );
        return !m_failOp.empty() && op.starts_with( m_failOp );
    }
};

namespace dev
{
/// Threadless shutter with injectable results.
template <class derivedT>
class pvcamTestShutter : public dssShutter<derivedT>
{
  public:
    /// Injected startup result.
    int appStartup()
    {
        return pvcamHarness::g_fake.m_shutter[0];
    }

    /// Injected logic result.
    int appLogic()
    {
        return pvcamHarness::g_fake.m_shutter[1];
    }

    /// Injected shutdown result.
    int appShutdown()
    {
        return pvcamHarness::g_fake.m_shutter[2];
    }

    /// Injected power-off result.
    int onPowerOff()
    {
        return pvcamHarness::g_fake.m_shutter[3];
    }

    /// Injected while-power-off result.
    int whilePowerOff()
    {
        return pvcamHarness::g_fake.m_shutter[4];
    }

    /// Record a requested shutter state.
    int setShutterState( int sh )
    {
        pvcamHarness::g_fake.m_shutterStates.push_back( sh );
        return pvcamHarness::g_fake.m_shutter[5];
    }
};
} // namespace dev
} // namespace app
} // namespace MagAOX

#define sem_init pvcamHarness::semInit
#define sem_trywait pvcamHarness::semTrywait
#define sem_post pvcamHarness::semPost
#define clock_gettime pvcamHarness::clockGettime
#define MagAOXApp pvcamTestApp
#define telemeter outletTestTelemeter
#define dssShutter pvcamTestShutter
#define pvcamPcie pvcamTestPcie
#define protected public
#include "../pvcamCtrl.hpp"
#undef protected
#undef pvcamPcie
#undef dssShutter
#undef telemeter
#undef MagAOXApp
#undef clock_gettime
#undef sem_post
#undef sem_trywait
#undef sem_init

namespace pvcamHarness
{

/// The real app with an isolated directory, a fake sysfs, and a simulated power loop.
struct Fixture : outletHarness::Controller<MagAOX::app::pvcamCtrl>
{
    /// Fake PCI device tree.
    Sysfs m_sysfs{ m_directory.m_path };

    /// Expose the stdCamera, frameGrabber, and MagAOXApp state used by the tests.
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_expTime;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_expTimeSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_fps;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_fpsSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_readoutSpeedName;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_readoutSpeedNameSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_defaultReadoutSpeed;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_fanSpeedName;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_fanSpeedNameSet;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_fanSpeedValid;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_fanSpeedControlEnabled;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_defaultFanSpeed;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_nextROI;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_currentROI;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_ccdTemp;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_ccdTempSetpt;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_tempControlStatus;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_tempControlOnTarget;
    using MagAOX::app::dev::stdCamera<MagAOX::app::pvcamCtrl>::m_tempControlStatusStr;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::pvcamCtrl>::m_reconfig;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::pvcamCtrl>::m_fgThread;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::pvcamCtrl>::m_width;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::pvcamCtrl>::m_height;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::pvcamCtrl>::m_dataType;
    using MagAOX::app::dev::frameGrabber<MagAOX::app::pvcamCtrl>::m_currImageTimestamp;
    using MagAOX::app::MagAOXApp<true>::m_powerOnWait;

    /// Configure power management, paths, and one camera, and reset the fakes.
    Fixture()
    {
        g_fake                                                    = {};
        outletHarness::g_faults                                   = {};
        g_fake.m_cameras                                          = { { "pvcamPCIE_0", "A22J723005" } };
        g_fake.m_values[{ PARAM_FAN_SPEED_SETPOINT, ATTR_AVAIL }] = 1;
        m_sysPath                                                 = m_directory.m_path;
        m_serialNumber                                            = "A22J723005";
        m_powerOnWait                                             = 0;
        m_pcieLockPath                                            = m_sysPath + "/pvcamCtrl_pcie.lock";
        m_pcie.sysfsPath( m_sysfs.m_root );
    }

    /// Stop and join the placeholder framegrabber thread.
    ~Fixture()
    {
        g_stopIdle = true;
        if( m_fgThread.joinable() )
            m_fgThread.join();
        g_stopIdle = false;
    }

    /// Set observed and target power.
    void power( int observed /**< [in] observed state */, int target /**< [in] target state */ )
    {
        MagAOX::app::MagAOXApp<true>::m_powerState       = observed;
        MagAOX::app::MagAOXApp<true>::m_powerTargetState = target;
    }

    /// Start the app as MagAOXApp::execute does, with the given initial power.
    void start( int on /**< [in] initial observed and target power */ )
    {
        power( on, on );
        state( MagAOX::app::stateCodes::INITIALIZED );
        REQUIRE( appStartup() == 0 );
        if( on > 0 )
        {
            state( MagAOX::app::stateCodes::POWERON );
        }
        else
        {
            m_powerOnCounter = 0;
            state( MagAOX::app::stateCodes::POWEROFF );
            REQUIRE( onPowerOff() == 0 );
        }
    }

    /// Number of captured logs at error priority or worse.
    static size_t errors()
    {
        size_t n = 0;
        for( auto &l : outletHarness::g_faults.m_logs )
            n += l.m_priority <= flatlogs::logPrio::LOG_ERROR;
        return n;
    }

    /// Run one iteration of MagAOXApp's main-loop power handling and logic.
    int loop()
    {
        int &m_powerState = MagAOX::app::MagAOXApp<true>::m_powerState;
        if( state() == MagAOX::app::stateCodes::POWEROFF )
        {
            if( m_powerState == 1 )
            {
                m_powerOnCounter = 0;
                state( MagAOX::app::stateCodes::POWERON );
            }
        }
        else if( m_powerState == 0 )
        {
            state( MagAOX::app::stateCodes::POWEROFF );
            REQUIRE( onPowerOff() == 0 );
        }
        if( m_powerState > 0 )
            return appLogic();
        return whilePowerOff();
    }

    /// Enable hotplug on a port with a camera below it.
    void pcie( const std::string &port = "0000:42:09.0" /**< [in] port */ )
    {
        REQUIRE( m_pcie.port( port ) == 0 );
        m_pcie.m_resetHoldMs = 1;
        m_pcie.m_settleMs    = 2;
    }

    /// Whether any captured log contains text.
    static bool logged( const std::string &text /**< [in] text */ )
    {
        for( auto &l : outletHarness::g_faults.m_logs )
            if( l.m_message.find( text ) != std::string::npos )
                return true;
        return false;
    }

    /// Number of captured logs containing text.
    static size_t count( const std::string &text /**< [in] text */ )
    {
        size_t n = 0;
        for( auto &l : outletHarness::g_faults.m_logs )
            n += l.m_message.find( text ) != std::string::npos;
        return n;
    }
};

} // namespace pvcamHarness
/// \endcond

#endif // pvcamCtrl_harness_hpp
