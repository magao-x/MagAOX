/** \file accelFFCtrlStream.hpp
 * \brief Existing-only ImageStreamIO ADC attachment; payload is never modified.
 * \ingroup accelFFCtrl_files
 */
#ifndef accelFFCtrlStream_hpp
#define accelFFCtrlStream_hpp

#include <atomic>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>
#include <sys/stat.h>

#include <ImageStreamIO/ImageStreamIO.h>

#include "accelFFCtrlInput.hpp"

namespace MagAOX
{
namespace app
{
namespace accelFFCtrlDetail
{

/// Worker-owned mapping and reader semaphore; no create, destroy, update or payload write API.
class adcStream
{
    IMAGE    m_image{};         ///< Existing native mapping, retained until its worker closes it.
    bool     m_open{ false };   ///< Whether close() owns a mapping.
    int      m_semaphore{ -1 }; ///< Claimed consumer slot; only this owner waits/releases it.
    ino_t    m_inode{ 0 };      ///< Backing-file identity published to supervision after a successful attachment.
    dev_t    m_device{ 0 };     ///< Device disambiguates inode numbers across mounted namespaces.
    uint32_t m_depth{ 0 };      ///< Attached allocation bound, independent of later mutable metadata.

  public:
    /// Construct without accessing any stream.
    adcStream() = default;

    /// A native mapping and reader slot have exactly one owner.
    adcStream( const adcStream &other /**< [in] prohibited second owner */ ) = delete;

    /// A native mapping and reader slot cannot be copied between owners.
    adcStream &operator=( const adcStream &other /**< [in] prohibited second owner */ ) = delete;

    /// Release only the reader slot and mapping.
    ~adcStream();

    /// Attach a validated ADC-named existing stream and claim a reader semaphore; never creates an image.
    bool open( const std::string &name /**< [in] non-DM input name */ );

    /// Return a bracketed metadata snapshot; native publication conventions are required.
    inputHeader header() const;

    /// Copy one complete plane's codes/counter/timestamps into local storage, within attached bounds.
    inputFrame frame( size_t index /**< [in] temporal plane */ ) const;

    /// Wait on a monotonic deadline, or use bounded relative polling; drain at most 256 additional posts after a wake.
    bool wait( int64_t deadline /**< [in] absolute monotonic deadline [ns] */ );

    /// Return the attached backing inode for supervision, without filesystem access.
    ino_t inode() const;

    /// Return the attached backing device.
    dev_t device() const;

    /// Release the reader slot and mapping after the owner stops using it.
    void close();
};

inline adcStream::~adcStream()
{
    close();
}

inline bool adcStream::open( const std::string &name )
{
    close();
    if( !inputNameValid( name ) )
        return false;
    char path[1024];
    if( ImageStreamIO_filename( path, sizeof( path ), name.c_str() ) != 0 )
        return false;
    struct stat before, after;
    if( stat( path, &before ) != 0 || ImageStreamIO_openIm( &m_image, name.c_str() ) != 0 )
        return false;
    m_open = true;
    if( stat( path, &after ) != 0 || before.st_dev != after.st_dev || before.st_ino != after.st_ino || !m_image.md ||
        !m_image.semptr || !m_image.semReadPID || m_image.md->sem == 0 )
    {
        close();
        return false;
    }
    m_depth     = m_image.md->naxis == 3 ? m_image.md->size[2] : 1;
    m_semaphore = ImageStreamIO_getsemwaitindex( &m_image, 5 );
    if( m_semaphore < 0 || m_semaphore >= m_image.md->sem || !m_image.semptr[m_semaphore] ||
        m_image.semptr[m_semaphore] == SEM_FAILED )
    {
        close();
        return false;
    }
    m_inode  = before.st_ino;
    m_device = before.st_dev;
    ImageStreamIO_semflush( &m_image, m_semaphore );
    return true;
}

inline inputHeader adcStream::header() const
{
    inputHeader value;
    if( !m_open || !m_image.md )
        return value;
    const volatile auto *md = m_image.md;
    value.m_write           = md->write != 0;
    std::atomic_thread_fence( std::memory_order_seq_cst );
    value.m_axes        = md->naxis;
    value.m_width       = md->size[0];
    value.m_height      = value.m_axes > 1 ? md->size[1] : 0;
    value.m_depth       = value.m_axes == 3 ? md->size[2] : 1;
    value.m_counter     = md->cnt0;
    value.m_index       = value.m_axes == 3 ? md->cnt1 : 0;
    value.m_uint16      = md->datatype == _DATATYPE_UINT16;
    value.m_cpu         = md->location == -1;
    const bool temporal = ( md->imagetype & 0xF000F ) == ( CIRCULAR_BUFFER | ZAXIS_TEMPORAL );
    value.m_valid =
        m_image.array.raw && value.m_depth == m_depth &&
        ( m_depth == 1 || ( temporal && m_image.cntarray && m_image.atimearray && m_image.writetimearray ) );
    std::atomic_thread_fence( std::memory_order_seq_cst );
    value.m_write = value.m_write || md->write != 0;
    return value;
}

inline inputFrame adcStream::frame( size_t index ) const
{
    inputFrame value;
    if( !m_open || index >= m_depth || !m_image.array.raw ||
        ( m_depth > 1 && ( !m_image.cntarray || !m_image.atimearray || !m_image.writetimearray ) ) )
        return value;
    std::atomic_thread_fence( std::memory_order_seq_cst );
    std::memcpy( value.m_adc.data(),
                 static_cast<const char *>( m_image.array.raw ) + index * 2 * sizeof( uint16_t ),
                 2 * sizeof( uint16_t ) );
    if( m_depth > 1 )
    {
        value.m_counter  = m_image.cntarray[index];
        value.m_acquired = m_image.atimearray[index];
        value.m_written  = m_image.writetimearray[index];
    }
    else
    {
        value.m_counter  = m_image.md->cnt0;
        value.m_acquired = m_image.md->atime;
        value.m_written  = m_image.md->writetime;
    }
    std::atomic_thread_fence( std::memory_order_seq_cst );
    return value;
}

inline bool adcStream::wait( int64_t deadline )
{
    if( !m_open || m_semaphore < 0 )
        return false;
    int result;
#if defined( __linux__ ) && defined( __GLIBC__ )
    #if __GLIBC_PREREQ( 2, 30 )
    const timespec target{ deadline / 1000000000, deadline % 1000000000 };
    result = sem_clockwait( m_image.semptr[m_semaphore], CLOCK_MONOTONIC, &target );
    #else
    result = sem_trywait( m_image.semptr[m_semaphore] );
    #endif
#else
    result = sem_trywait( m_image.semptr[m_semaphore] );
#endif
    if( result == 0 )
    {
        for( unsigned n = 0; n < 256; ++n )
            if( sem_trywait( m_image.semptr[m_semaphore] ) != 0 )
                break;
        return true;
    }
    if( errno == EAGAIN )
    {
        timespec now;
        if( clock_gettime( CLOCK_MONOTONIC, &now ) != 0 )
            return false;
        const int64_t remaining = deadline - ( int64_t( now.tv_sec ) * 1000000000 + now.tv_nsec );
        if( remaining > 0 )
            std::this_thread::sleep_for( std::chrono::nanoseconds( std::min<int64_t>( remaining, 20000 ) ) );
        return true;
    }
    return errno == ETIMEDOUT || errno == EINTR;
}

inline ino_t adcStream::inode() const
{
    return m_inode;
}
inline dev_t adcStream::device() const
{
    return m_device;
}

inline void adcStream::close()
{
    if( m_open )
    {
        if( m_semaphore >= 0 && m_image.semReadPID && m_image.md && m_semaphore < m_image.md->sem &&
            m_image.semReadPID[m_semaphore] == getpid() )
            m_image.semReadPID[m_semaphore] = 0;
        ImageStreamIO_closeIm( &m_image );
    }
    m_open      = false;
    m_semaphore = -1;
    m_inode = m_device = 0;
    m_depth            = 0;
    m_image            = {};
}

} // namespace accelFFCtrlDetail
} // namespace app
} // namespace MagAOX
#endif
