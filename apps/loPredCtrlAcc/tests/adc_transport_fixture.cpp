/** \file adc_transport_fixture.cpp
 * \brief Fake existing-only ImageStreamIO API with private ordinary files, not native shared memory.
 */
#include "adc_transport_fixture.hpp"
#include <array>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <unistd.h>

/// \cond ADC_TRANSPORT_FIXTURE
namespace
{
std::mutex                   s_mutex;  ///< Protect fixture registration against the real reader thread.
std::map<std::string, IMAGE> s_images; ///< Registered immutable borrowed images, never actual streams.

/// Return a process-specific temporary directory used solely by the API double.
std::filesystem::path fixture_directory()
{
    return std::filesystem::temp_directory_path() / ( "ddspc-adc-api-" + std::to_string( getpid() ) );
}
} // namespace

int ImageStreamIO_filename( char *path, std::size_t size, const char *name )
{
    const auto filename = ( fixture_directory() / name ).string();
    return std::snprintf( path, size, "%s", filename.c_str() ) < int( size ) ? 0 : -1;
}

int ImageStreamIO_openIm( IMAGE *image, const char *name )
{
    std::lock_guard<std::mutex> guard( s_mutex ); // mutex scope
    const auto                  found = s_images.find( name );
    if( found == s_images.end() )
    {
        return -1;
    }
    *image = found->second;
    return 0;
}

int ImageStreamIO_getsemwaitindex( IMAGE *image, int )
{
    image->semReadPID[0] = getpid();
    return 0;
}

void ImageStreamIO_semflush( IMAGE *, int )
{
}

void ImageStreamIO_closeIm( IMAGE *image )
{
    *image = {};
}

int ddspc_adc_sem_trywait( sem_t * )
{
    errno = EAGAIN;
    return -1;
}

namespace libXWCTest::loPredCtrlAccTest
{
struct AdcTransportFixture::Impl
{
    std::string                m_name;        ///< Registry key and private ordinary-file basename.
    IMAGE_METADATA             m_metadata;    ///< Immutable static publication, initialized before reader startup.
    std::vector<std::uint16_t> m_raw;         ///< Known positive/negative acceleration pair.
    sem_t                      m_semaphore{}; ///< Opaque address only; never passed to a native semaphore function.
    sem_t                     *m_pointer{ &m_semaphore }; ///< Stable fake notification pointer.
    pid_t                      m_reader{ 0 };             ///< Bookkeeping written only by the real reader thread.
    std::vector<std::uint64_t> m_counters;                ///< Immutable per-plane publication identities.
    std::vector<timespec>      m_acquired, m_written;     ///< Immutable per-plane timestamps.
};

AdcTransportFixture::AdcTransportFixture( const std::string &name, std::uint32_t depth )
    : m_impl( std::make_unique<Impl>() )
{
    if( depth < 1 || depth > 16 || name.empty() ||
        name.find_first_not_of( "abcdefghijklmnopqrstuvwxyz0123456789_-" ) != std::string::npos )
    {
        throw std::invalid_argument( "Unsafe private fixture name" );
    }
    m_impl->m_name = name;
    clock_gettime( CLOCK_REALTIME, &m_impl->m_metadata.atime );
    m_impl->m_metadata.writetime = m_impl->m_metadata.atime;
    m_impl->m_metadata.cnt0      = 1;
    m_impl->m_metadata.size[2]   = depth;
    m_impl->m_raw.resize( 2 * depth );
    m_impl->m_raw[0] = 2049;
    m_impl->m_raw[1] = 2047;
    m_impl->m_counters.resize( depth );
    m_impl->m_counters[0] = 1;
    m_impl->m_acquired.resize( depth );
    m_impl->m_written.resize( depth );
    m_impl->m_acquired[0] = m_impl->m_metadata.atime;
    m_impl->m_written[0]  = m_impl->m_metadata.writetime;
    if( depth > 1 )
    {
        ++m_impl->m_metadata.atime.tv_sec; // Multi-plane readers must use the original per-plane time.
    }
    IMAGE image;
    image.md             = &m_impl->m_metadata;
    image.array.raw      = m_impl->m_raw.data();
    image.atimearray     = m_impl->m_acquired.data();
    image.writetimearray = m_impl->m_written.data();
    image.cntarray       = m_impl->m_counters.data();
    image.semptr         = &m_impl->m_pointer;
    image.semReadPID     = &m_impl->m_reader;
    std::filesystem::create_directory( fixture_directory() );
    { //mutex scope
        std::lock_guard<std::mutex> guard( s_mutex );
        s_images.emplace( name, image );
    }
    reconnect();
}

AdcTransportFixture::AdcTransportFixture( const std::string &name, const IMAGE &image )
    : m_impl( std::make_unique<Impl>() )
{
    if( name.empty() || name.find_first_not_of( "abcdefghijklmnopqrstuvwxyz0123456789_-" ) != std::string::npos )
        throw std::invalid_argument( "Unsafe private observation fixture name" );
    m_impl->m_name = name;
    std::filesystem::create_directory( fixture_directory() );
    { //mutex scope
        std::lock_guard<std::mutex> guard( s_mutex );
        if( !s_images.emplace( name, image ).second )
            throw std::invalid_argument( "Duplicate private fixture" );
    }
    reconnect();
}

AdcTransportFixture::~AdcTransportFixture()
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( s_mutex );
        s_images.erase( m_impl->m_name );
    }
    disconnect();
    std::error_code ignored;
    std::filesystem::remove( fixture_directory(), ignored );
}

void AdcTransportFixture::disconnect()
{
    std::filesystem::remove( fixture_directory() / m_impl->m_name );
}

void AdcTransportFixture::reconnect()
{
    std::ofstream file( fixture_directory() / m_impl->m_name );
    if( !file )
    {
        throw std::runtime_error( "Cannot create private ADC API fixture" );
    }
}

void AdcTransportFixture::replace()
{
    const auto path        = fixture_directory() / m_impl->m_name;
    const auto replacement = path.string() + ".replacement";
    {
        std::ofstream file( replacement );
        if( !file )
        {
            throw std::runtime_error( "Cannot create replacement ADC API fixture" );
        }
    }
    std::filesystem::rename( replacement, path );
}

pid_t AdcTransportFixture::reader() const
{
    return m_impl->m_reader;
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
