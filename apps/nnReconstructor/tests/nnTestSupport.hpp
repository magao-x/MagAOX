/** \file nnTestSupport.hpp
 * \brief CPU-only SDK and app-boundary doubles for the real nnReconstructor methods.
 * \ingroup nnReconstructor_unit_test
 *
 * Names and tiny inline accessors intentionally mirror external APIs. These doubles
 * inject SDK failures and provide property/metadata storage; they do not simulate
 * the application's control state machine or execute the shared base-class workers.
 */
#ifndef nnReconstructor_nnTestSupport_hpp
#define nnReconstructor_nnTestSupport_hpp

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/stat.h>

/** \cond NNRECONSTRUCTOR_TEST_SUPPORT */
namespace nnTest
{
/// Allocation/copy ordinals inject one selected failure; live allocations verify cleanup.
inline int allocation = 0, failAllocation = 0, copies = 0, failCopy = 0, liveAllocations = 0;
/// Count live SDK objects to check partial loading and replacement cleanup.
inline int runtimes = 0, engines = 0, contexts = 0;
/// Independently inject failures at SDK construction and execution boundaries.
inline bool runtimeFailure = false, engineFailure = false, contextFailure = false, executeFailure = false;
/// Test latches force overlap while the real app has admitted inference or loading.
inline std::function<void()> duringExecute, duringDeserialize;
/// Record input tensor values independently of the production preprocessing implementation.
inline std::vector<float> uploaded;
/// Constant fake network output used to detect mixed publisher snapshots.
inline float result = 1;
/// Inject failure of the callback/zero timestamp call.
inline bool failClock = false;
} // namespace nnTest
/// Return the real time unless timestamp failure is injected.
inline int nnTestClock( clockid_t clock /**< [in] Requested clock. */, timespec *value /**< [out] Timestamp. */ )
{
    if( nnTest::failClock )
        return -1;
    return ::clock_gettime( clock, value );
}
#define clock_gettime nnTestClock
using cudaError_t         = int;
constexpr int cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
/// Provide a deterministic message for any injected CUDA error.
inline const char *cudaGetErrorString( int status /**< [in] Ignored injected status. */ )
{
    return "injected CUDA error";
}
/// Allocate CPU storage or fail the selected allocation ordinal.
inline int cudaMalloc( void **pointer /**< [out] Fake device allocation. */,
                       size_t bytes /**< [in] Allocation size. */ )
{
    if( ++nnTest::allocation == nnTest::failAllocation )
        return 1;
    *pointer = std::malloc( bytes );
    ++nnTest::liveAllocations;
    return *pointer ? 0 : 1;
}
/// Free a fake device allocation and update ownership accounting.
inline int cudaFree( void *pointer /**< [in] Owned CPU allocation. */ )
{
    if( pointer )
    {
        --nnTest::liveAllocations;
        std::free( pointer );
    }
    return 0;
}
/// Copy exactly the requested bytes or inject failure before touching the destination.
inline int cudaMemcpy( void       *dest /**< [out] Destination. */,
                       const void *src /**< [in] Source. */,
                       size_t      size /**< [in] Byte count. */,
                       int         kind /**< [in] Unused transfer direction. */ )
{
    if( ++nnTest::copies == nnTest::failCopy )
        return 1;
    std::memcpy( dest, src, size );
    return 0;
}
namespace nvinfer1
{
enum class DataType
{
    kFLOAT,
    kHALF
};
enum class TensorFormat
{
    kLINEAR,
    kOTHER
};
enum class TensorLocation
{
    kDEVICE,
    kHOST
};
enum class TensorIOMode
{
    kINPUT,
    kOUTPUT
};
/// Static dimensions returned by the engine boundary double.
struct Dims
{
    int                nbDims;
    std::array<int, 8> d;
};
/// Tensor metadata used by the real engine-contract validator.
struct Tensor
{
    std::string    name;
    TensorIOMode   role;
    Dims           dims;
    DataType       type     = DataType::kFLOAT;
    TensorFormat   format   = TensorFormat::kLINEAR;
    TensorLocation location = TensorLocation::kDEVICE;
};
/// Metadata copied into each fake engine at deserialization.
inline std::vector<Tensor> tensors;
/// Optional per-file contracts; an empty contract injects deserialization failure.
inline std::map<std::string, std::vector<Tensor>> serializedTensors;
/// Minimal TensorRT logging interface.
struct ILogger
{
    enum class Severity
    {
        kERROR,
        kWARNING,
        kINFO
    };
    /// Match the SDK logger callback.
    virtual void log( Severity severity /**< [in] Severity. */, const char *message /**< [in] Text. */ ) noexcept = 0;
    /// Permit destruction through the logging interface.
    virtual ~ILogger() = default;
};
/// Fake SDK execution; production admission and handoff logic remain in the app.
struct IExecutionContext
{
    std::vector<Tensor> m_tensors; ///< Copy of the creating engine contract.
    /// Count ownership and retain the input/output contract.
    explicit IExecutionContext( std::vector<Tensor> values /**< [in] Tensor metadata. */ )
        : m_tensors( std::move( values ) )
    {
        ++nnTest::contexts;
    }
    /// Account for released execution-context ownership.
    ~IExecutionContext()
    {
        --nnTest::contexts;
    }
    /// Capture inputs and write a constant output, with optional synchronization/failure injection.
    bool executeV2( void *const *buffers /**< [in/out] Actual production binding array. */ );
};
inline bool IExecutionContext::executeV2( void *const *buffers )
{
    if( nnTest::duringExecute )
        nnTest::duringExecute();
    if( nnTest::executeFailure )
        return false;
    for( size_t i = 0; i < m_tensors.size(); ++i )
    {
        auto  &t = m_tensors[i];
        size_t n = 1;
        for( int d = 0; d < t.dims.nbDims; ++d )
            n *= t.dims.d[d];
        if( t.role == TensorIOMode::kOUTPUT )
            for( size_t j = 0; j < n; ++j )
                static_cast<float *>( buffers[i] )[j] = nnTest::result;
        else
        {
            nnTest::uploaded.resize( n );
            for( size_t j = 0; j < n; ++j )
                nnTest::uploaded[j] = static_cast<float *>( buffers[i] )[j];
        }
    }
    return true;
}

/// Fake engine exposing the metadata queried by production validation.
struct ICudaEngine
{
    std::vector<Tensor> m_tensors = tensors; ///< Snapshot independent of subsequent test setup.
    /// Count engine ownership.
    ICudaEngine()
    {
        ++nnTest::engines;
    }
    /// Account for released engine ownership.
    ~ICudaEngine()
    {
        --nnTest::engines;
    }
    /// Return the configured tensor count.
    int getNbIOTensors() const
    {
        return m_tensors.size();
    }
    /// Return the configured enumeration order.
    const char *getIOTensorName( int i /**< [in] Tensor index. */ ) const
    {
        return m_tensors.at( i ).name.c_str();
    }
    /// Look up a tensor by its engine name.
    const Tensor &get( const char *name /**< [in] Tensor name. */ ) const;

    /// Return the configured tensor datatype.
    DataType getTensorDataType( const char *name /**< [in] Tensor name. */ ) const
    {
        return get( name ).type;
    }
    /// Return the configured tensor layout.
    TensorFormat getTensorFormat( const char *name /**< [in] Tensor name. */ ) const
    {
        return get( name ).format;
    }
    /// Return the configured tensor memory location.
    TensorLocation getTensorLocation( const char *name /**< [in] Tensor name. */ ) const
    {
        return get( name ).location;
    }
    /// Return the configured tensor input/output role.
    TensorIOMode getTensorIOMode( const char *name /**< [in] Tensor name. */ ) const
    {
        return get( name ).role;
    }
    /// Return the configured tensor dimensions.
    Dims getTensorShape( const char *name /**< [in] Tensor name. */ ) const
    {
        return get( name ).dims;
    }
    /// Create a counted execution context or inject failure.
    IExecutionContext *createExecutionContext()
    {
        return nnTest::contextFailure ? nullptr : new IExecutionContext( m_tensors );
    }
};
inline const Tensor &ICudaEngine::get( const char *name ) const
{
    return *std::find_if( m_tensors.begin(), m_tensors.end(), [name]( auto &t ) { return t.name == name; } );
}

/// Fake TensorRT runtime with a controllable deserialization boundary.
struct IRuntime
{
    /// Count runtime ownership.
    IRuntime()
    {
        ++nnTest::runtimes;
    }
    /// Account for released runtime ownership.
    ~IRuntime()
    {
        --nnTest::runtimes;
    }
    /// Pause/fail at the SDK boundary; serialized file reading is still production code.
    ICudaEngine *deserializeCudaEngine( const void *data /**< [in] Unparsed serialized bytes. */,
                                        size_t      size /**< [in] Byte count. */ )
    {
        if( nnTest::duringDeserialize )
            nnTest::duringDeserialize();
        const auto found = serializedTensors.find( std::string( static_cast<const char *>( data ), size ) );
        if( nnTest::engineFailure || ( found != serializedTensors.end() && found->second.empty() ) )
            return nullptr;
        auto *engine = new ICudaEngine;
        if( found != serializedTensors.end() )
            engine->m_tensors = found->second;
        return engine;
    }
};
/// Create a counted runtime or inject failure.
inline IRuntime *createInferRuntime( ILogger &logger /**< [in] Unused SDK logger. */ )
{
    return nnTest::runtimeFailure ? nullptr : new IRuntime;
}
} // namespace nvinfer1
namespace mx::app
{
/// Configuration boundary double; tests supply validated fixture values directly.
struct appConfigurator
{
    template <class... T>
    void add( T &&... )
    {
    }
    template <class T>
    void operator()( T &, const char * )
    {
    }
};
} // namespace mx::app
namespace pcf
{
/// Value storage used by actual production INDI callbacks.
struct IndiElement
{
    enum SwitchStateType
    {
        UnknownSwitchState,
        Off,
        On
    };
    std::string     name, text;
    float           value = 0;
    SwitchStateType sw    = Off;
    /// Match text request construction at the INDI boundary.
    IndiElement( std::string n /**< [in] Element name. */, const std::string &value /**< [in] Text value. */ )
        : name( std::move( n ) ), text( value )
    {
    }
    /// Match numeric request construction at the INDI boundary.
    IndiElement( std::string n /**< [in] Element name. */, double value /**< [in] Numeric value. */ )
        : name( std::move( n ) ), value( value )
    {
    }
    explicit IndiElement( std::string n, SwitchStateType s = Off ) : name( std::move( n ) ), sw( s )
    {
    }
    template <class T>
    T get() const
    {
        if constexpr( std::is_same_v<T, std::string> )
            return text;
        else
            return value;
    }
    SwitchStateType getSwitchState() const
    {
        return sw;
    }
    void setSwitchState( SwitchStateType s )
    {
        sw = s;
    }
    const std::string &getName() const
    {
        return name;
    }
    IndiElement &operator=( float v )
    {
        value = v;
        return *this;
    }
    IndiElement &operator=( const std::string &v )
    {
        text = v;
        return *this;
    }
    IndiElement &operator=( SwitchStateType v )
    {
        sw = v;
        return *this;
    }
};
/// Property storage; no network or real INDI serialization is tested.
struct IndiProperty
{
    enum Type
    {
        Text,
        Switch,
        Number
    };
    enum PropertyStateType
    {
        Idle,
        Ok,
        Busy,
        Alert
    };
    enum Perm
    {
        ReadOnly,
        ReadWrite
    };
    enum Rule
    {
        OneOfMany,
        AtMostOne
    };
    std::string                        name, device;
    PropertyStateType                  status = Idle;
    std::map<std::string, IndiElement> elements;
    explicit IndiProperty( Type = Text )
    {
    }
    void add( IndiElement e )
    {
        elements.emplace( e.name, std::move( e ) );
    }
    bool find( const std::string &n ) const
    {
        return elements.contains( n );
    }
    IndiElement &operator[]( const std::string &n )
    {
        return elements.at( n );
    }
    const IndiElement &operator[]( const std::string &n ) const
    {
        return elements.at( n );
    }
    const auto &getElements() const
    {
        return elements;
    }
    void setName( const std::string &n )
    {
        name = n;
    }
    void setDevice( const std::string &n )
    {
        device = n;
    }
    const std::string &getName() const
    {
        return name;
    }
    const std::string &getDevice() const
    {
        return device;
    }
    /// Return the observed vector state.
    PropertyStateType getState() const
    {
        return status;
    }

    void setState( PropertyStateType v )
    {
        status = v;
    }
    void setPerm( Perm )
    {
    }
    void setRule( Rule )
    {
    }
};
} // namespace pcf
constexpr auto INDI_IDLE = pcf::IndiProperty::Idle, INDI_OK = pcf::IndiProperty::Ok;
constexpr auto INDI_BUSY = pcf::IndiProperty::Busy, INDI_ALERT = pcf::IndiProperty::Alert;
constexpr int  _DATATYPE_FLOAT = 9;
/// ImageStreamIO metadata subset used by production attachment checks.
struct Metadata
{
    int      datatype = _DATATYPE_FLOAT, location = -1, naxis = 2, sem = 10;
    uint32_t size[3] = { 4, 4, 1 };
    uint64_t cnt0 = 0, cnt1 = 0;
    int      write = 0;
    timespec atime{}, writetime{};
};
/// ImageStreamIO handle subset with CPU-backed test pixels.
struct IMAGE
{
    Metadata *md = nullptr;
    struct
    {
        void *raw = nullptr;
    } array;
};
/// Keep fake stream files within the app-local test build directory.
inline void ImageStreamIO_filename( char       *dest /**< [out] Filename buffer. */,
                                    size_t      size /**< [in] Buffer capacity. */,
                                    const char *name /**< [in] Stream name. */ )
{
    std::snprintf( dest, size, ".build/%s.im.shm", name );
}
namespace MagAOX::app
{
struct software_error
{
};
struct software_critical
{
};
struct text_log
{
};
struct telem_fgtimings
{
};
namespace logPrio
{
constexpr int LOG_INFO = 0, LOG_WARNING = 1;
}
namespace argType
{
constexpr int Required = 0;
}
namespace stateCodes
{
constexpr int READY = 0, OPERATING = 1;
}
/// Small contiguous image double; preprocessing indexing is separately checked against an oracle.
template <class T>
struct eigenImage
{
    std::vector<T> values;
    void           resize( size_t n, size_t )
    {
        values.resize( n );
    }
    T *data()
    {
        return values.data();
    }
    size_t rows() const
    {
        return values.size();
    }
    void setZero()
    {
        std::fill( values.begin(), values.end(), T{} );
    }
};
struct logMessage
{
    logMessage( const std::string & )
    {
    }
};
/// Stub registration/logging boundary; production app state and callbacks are compiled unchanged.
template <bool>
struct MagAOXApp
{
    /// Socket-free discovery boundary; no instrument connection is opened.
    struct Driver
    {
        /// Ignore read-only discovery requests in the CPU fixture.
        void sendGetProperties( const pcf::IndiProperty &property /**< [in] Requested vector. */ )
        {
            (void)property;
        }
    };
    Driver                        *m_indiDriver{ nullptr }; ///< No live driver is owned by tests.
    std::vector<pcf::IndiProperty> m_sent;                  ///< Recorded external control requests.
    /// Record the production modulation command without network access.
    int sendNewProperty( const pcf::IndiProperty &property /**< [in] Command payload. */ )
    {
        m_sent.push_back( property );
        return 0;
    }
    std::atomic<bool>        m_shutdown{ false };
    std::mutex               m_indiMutex;
    mx::app::appConfigurator config;
    MagAOXApp( int, int )
    {
    }
    virtual ~MagAOXApp() = default;
    std::string configName()
    {
        return "nnReconstructor";
    }
    void state( int )
    {
    }
    template <class T, int R = 0>
    int log( logMessage, int = 0 )
    {
        return R;
    }
    void createROIndiNumber( pcf::IndiProperty &, const char * )
    {
    }
    void createROIndiText( pcf::IndiProperty &p, const std::string &name, const std::string &element )
    {
        p.setName( name );
        p.add( pcf::IndiElement( element ) );
    }
    int createStandardIndiSelectionSw( pcf::IndiProperty              &p,
                                       const std::string              &name,
                                       const std::vector<std::string> &ids,
                                       const std::vector<std::string> &,
                                       const std::string & = "",
                                       const std::string & = "" )
    {
        p.setName( name );
        for( auto &id : ids )
            p.add( pcf::IndiElement( id ) );
        return 0;
    }
    void createStandardIndiToggleSw( pcf::IndiProperty &p, const std::string &name )
    {
        p.setName( name );
        p.add( pcf::IndiElement( "toggle" ) );
    }
    template <class... T>
    int registerIndiPropertyNew( T &&... )
    {
        return 0;
    }
    template <class T>
    int registerIndiPropertyReadOnly( T & )
    {
        return 0;
    }
    template <class T>
    void
    updateIfChanged( pcf::IndiProperty &p, const char *name, T v, pcf::IndiProperty::PropertyStateType s = INDI_OK )
    {
        p[name]  = v;
        p.status = s;
    }
    void updateSwitchIfChanged( pcf::IndiProperty                   &p,
                                const std::string                   &name,
                                pcf::IndiElement::SwitchStateType    v,
                                pcf::IndiProperty::PropertyStateType s )
    {
        p[name]  = v;
        p.status = s;
    }
};
namespace dev
{
struct shmimT
{
};
/// Expose WFS metadata without starting a base-class thread.
template <class T>
class shmimMonitor
{
  public:
    int      m_width = 4, m_height = 4;
    Metadata md;
    IMAGE    m_imageStream{ &md, { nullptr } };
    bool     m_getExistingFirst = false;
    /// Fixture attachment state; real semaphore ownership is tested separately.
    bool m_idle = true;
    /// Apply a state request to the fixture monitor.
    void monitorState( int value /**< [in] Requested FSM state. */ )
    {
        m_idle = value != stateCodes::OPERATING;
    }
    /// Report fixture detachment.
    bool monitorIdle() const
    {
        return m_idle;
    }
    int appShutdown()
    {
        return 0;
    }
};
/// Expose an external destination; tests explicitly invoke the production publication stages.
template <class T>
class frameGrabber
{
  public:
    uint32_t             m_width = 3, m_height = 1;
    int                  m_dataType  = _DATATYPE_FLOAT;
    size_t               m_typeSize  = 4;
    bool                 m_ownShmim  = true;
    std::string          m_shmimName = "test";
    timespec             m_currImageTimestamp{};
    Metadata             md;
    std::array<float, 3> data{};
    IMAGE                stream{ &md, { data.data() } };
    IMAGE               *m_imageStream    = &stream;
    size_t               m_circBuffLength = 1;
    ino_t                m_inode          = 0;
    int                  openShmim();

    int appShutdown()
    {
        return 0;
    }
    int recordFGTimings( bool )
    {
        return 0;
    }
};
template <class T>
inline int frameGrabber<T>::openShmim()
{
    md.size[0] = 3;
    md.size[1] = 1;
    std::ofstream( ".build/test.im.shm" ) << "stream";
    struct stat info;
    ::stat( ".build/test.im.shm", &info );
    m_inode = info.st_ino;
    return 0;
}

/// No-op telemetry boundary for CPU-only app tests.
template <class T>
class telemeter
{
  public:
    int appShutdown()
    {
        return 0;
    }
    template <class U>
    int checkRecordTimes( U )
    {
        return 0;
    }
};
} // namespace dev
} // namespace MagAOX::app
namespace Eigen
{
/// Column-major float view matching the existing Eigen map indexing.
template <class T>
struct Map
{
    float *m_data;
    size_t m_rows;
    Map( float *data, size_t rows, size_t ) : m_data( data ), m_rows( rows )
    {
    }
    float operator()( size_t row, size_t col ) const
    {
        return m_data[row + col * m_rows];
    }
};
} // namespace Eigen
#define MAGAOX_CURRENT_SHA1 0
#define MAGAOX_REPO_MODIFIED 0
#define SHMIMMONITOR_SETUP_CONFIG( x )
#define FRAMEGRABBER_SETUP_CONFIG( x )
#define TELEMETER_SETUP_CONFIG( x )
#define SHMIMMONITOR_LOAD_CONFIG( x )
#define FRAMEGRABBER_LOAD_CONFIG( x )
#define TELEMETER_LOAD_CONFIG( x )
#define SHMIMMONITOR_APP_STARTUP
#define FRAMEGRABBER_APP_STARTUP
#define TELEMETER_APP_STARTUP
#define SHMIMMONITOR_APP_LOGIC
#define FRAMEGRABBER_APP_LOGIC
#define TELEMETER_APP_LOGIC
#define SHMIMMONITOR_UPDATE_INDI
#define FRAMEGRABBER_UPDATE_INDI
#define REG_INDI_SETPROP( prop, dev, name )                                                                            \
    ( prop ).setDevice( dev );                                                                                         \
    ( prop ).setName( name )
#define INDI_SETCALLBACK_DECL( cls, prop ) int setCallBack_##prop( const pcf::IndiProperty & )
#define INDI_SETCALLBACK_DEFN( cls, prop ) int cls::setCallBack_##prop
#define INDI_NEWCALLBACK_DECL( cls, prop ) int newCallBack_##prop( const pcf::IndiProperty & )
#define INDI_NEWCALLBACK_DEFN( cls, prop ) int cls::newCallBack_##prop
#define INDI_NEWCALLBACK( prop ) nullptr
#define INDI_VALIDATE_CALLBACK_PROPS( prop, recv )                                                                     \
    if( ( recv ).getName() != ( prop ).getName() )                                                                     \
    return -1
/** \endcond */
#include "cacaoWriter_test.hpp"
#endif
