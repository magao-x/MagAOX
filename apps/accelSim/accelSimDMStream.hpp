/** \file accelSimDMStream.hpp
 * \brief Existing-image attachment and native active DM publication.
 * \ingroup accelSim_files
 */
#ifndef accelSimDMStream_hpp
#define accelSimDMStream_hpp

#include <atomic>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <ImageStreamIO/ImageStreamIO.h>

namespace MagAOX
{
namespace app
{
namespace accelSimDetail
{

/// Existing-only channel attachment; acquisition owns writes, supervision only probes identity.
class dmStream
{
    IMAGE             m_image{};        ///< Mapping remains alive until acquisition has joined.
    bool              m_open{ false };  ///< Whether this object owns a mapping, not exclusive producer ownership.
    std::string       m_filename;       ///< Resolved backing path, allocated outside acquisition.
    ino_t             m_inode{ 0 };     ///< Original inode; replacement requires restart and operator handover.
    std::atomic<bool> m_valid{ false }; ///< Supervision can invalidate but never reattach an active mapping.
    timespec          m_acquired{};     ///< Source ADC acquisition time, set by the sole publisher.

  public:
    /// Construct without accessing a stream.
    dmStream() = default;

    /// Close only; callers must explicitly zero after quiescing acquisition.
    ~dmStream();

    /// Attach an existing 2-D 50-by-50 float image; never creates or replaces it.
    bool open( const std::string &name /**< [in] explicitly owned channel name */ );

    /// Validate mapped geometry without filesystem I/O or waiting on a consumer.
    bool valid() const;

    /// Detect disappeared/replaced backing files outside the acquisition path.
    bool probe();

    /// Retain the corresponding ADC acquisition timestamp without changing publication counters.
    void acquisitionTime( const timespec &value /**< [in] ADC sample time */ );

    /// Publish a complete local image via ImageStreamIO's native active update operation.
    bool operator()( const std::vector<float> &command /**< [in] complete surface-micron contribution */ );

    /// Release the mapping after the last writer has stopped.
    void close();
};

inline dmStream::~dmStream()
{
    close();
}

inline bool dmStream::open( const std::string &name )
{
    close();
    char filename[1024];
    ImageStreamIO_filename( filename, sizeof( filename ), name.c_str() );
    struct stat before, after;
    if( stat( filename, &before ) != 0 || ImageStreamIO_openIm( &m_image, name.c_str() ) != 0 )
        return false;
    m_open     = true;
    m_filename = filename;
    m_inode    = before.st_ino;
    m_valid    = stat( filename, &after ) == 0 && before.st_ino == after.st_ino;
    if( !valid() )
    {
        close();
        return false;
    }
    return true;
}

inline bool dmStream::valid() const
{
    return m_valid.load() && m_open && m_image.md && m_image.array.raw && m_image.md->naxis == 2 &&
           m_image.md->size[0] == 50 && m_image.md->size[1] == 50 && m_image.md->datatype == _DATATYPE_FLOAT &&
           m_image.md->sem > 0;
}

inline bool dmStream::probe()
{
    struct stat info;
    if( !m_open || stat( m_filename.c_str(), &info ) != 0 || info.st_ino != m_inode )
        m_valid = false;
    return valid();
}

inline bool dmStream::operator()( const std::vector<float> &command )
{
    if( !valid() || command.size() != 2500 || m_image.md->write )
        return false; // Never wait on another writer; ownership is an operator precondition.
    m_image.md->write = 1;
    std::memcpy( m_image.array.raw, command.data(), 2500 * sizeof( float ) );
    m_image.md->atime = m_acquired;
    // Same active primitive used by mx::improc::milkImage::post; no invented counter/semaphore sequence.
    const bool success = ImageStreamIO_UpdateIm( &m_image ) == 0;
    if( !success )
        m_image.md->write = 0;
    return success;
}

inline void dmStream::acquisitionTime( const timespec &value )
{
    m_acquired = value;
}

inline void dmStream::close()
{
    m_valid = false;
    if( m_open )
        ImageStreamIO_closeIm( &m_image );
    m_open = false;
}

} // namespace accelSimDetail
} // namespace app
} // namespace MagAOX
#endif
