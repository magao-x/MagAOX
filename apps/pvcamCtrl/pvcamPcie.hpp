/** \file pvcamPcie.hpp
 * \brief PCIe re-enumeration of a single PVCAM camera behind its own switch downstream port.
 *
 * \ingroup pvcamCtrl_files
 */

#ifndef pvcamPcie_hpp
#define pvcamPcie_hpp

#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace MagAOX
{
namespace app
{

/// Condition of the camera function found below a downstream port.
enum class pcieCamera
{
    error,        ///< The port or camera could not be read.
    none,         ///< No camera function is enumerated below the port.
    unresponsive, ///< The camera is enumerated but config reads return all ones (powered off or link down).
    stale,        ///< The camera responds but lost its kernel-assigned configuration (power-cycled since enumeration).
    healthy       ///< The camera responds with memory decoding enabled at its kernel-assigned BAR0 address.
};

/// PCIe hotplug of one PVCAM camera behind its own switch downstream port.
/** Performs the steps of the vendor script `/opt/pvcam/drivers/in-kernel/pcie/hotplug_pcie.sh` (as installed
 * 2026-10) for exactly one downstream port: a secondary bus reset of that port, removal of the camera function
 * below it (if any), and a rescan of that port.  The vendor script instead resets every downstream port and
 * removes every camera, which would disrupt other cameras on the same card.
 *
 * All access is through sysfs below \ref m_sysfsPath.  Config-space reads beyond 64 bytes and all writes require
 * root, so callers must elevate privileges.  This class does not log; on failure \ref error() describes the cause.
 *
 * \ingroup pvcamCtrl
 */
class pvcamPcie
{
  public:
    /// PCI vendor ID of Teledyne Photometrics PCIe cameras.
    static constexpr uint16_t c_cameraVendor = 0x1b6b;

    /// PCI device ID of Teledyne Photometrics PCIe cameras.
    static constexpr uint16_t c_cameraDevice = 0x0001;

    /// Vendor and device IDs of supported switch cards (OSS, IOI, and two Dolphin cards), from the vendor script.
    static constexpr std::array<std::array<uint16_t, 2>, 4> c_cardIds{
        { { 0x10b5, 0x8609 }, { 0x10b5, 0x8718 }, { 0x10b5, 0x8733 }, { 0x10b5, 0x8747 } } };

  protected:
    /** \name Configuration - Data
     *@{
     */

    std::string m_sysfsPath{ "/sys/bus/pci/devices" }; ///< Root of the PCI device tree.

    std::string m_port; ///< PCI address (BDF) of the downstream port above the camera.  Empty disables hotplug.

    unsigned m_resetHoldMs{ 10 }; ///< Time the secondary bus reset is held, in ms, as in the vendor script.

    unsigned m_settleMs{ 500 }; ///< Settle time after each reset, removal, and rescan step, in ms.

    ///@}

    std::string m_error; ///< Description of the most recent failure.

  public:
    /// Destructor.
    virtual ~pvcamPcie() = default;

    /** \name Configuration
     *@{
     */

    /// Set the downstream port.
    /**
     * \returns 0 on success
     * \returns -1 if the address is not empty and is not a PCI address of the form `0000:42:09.0`
     */
    int port( const std::string &bdf /**< [in] PCI address of the port, or empty to disable hotplug */ );

    /// Get the downstream port.
    const std::string &port() const;

    /// Set the root of the PCI device tree.
    void sysfsPath( const std::string &path /**< [in] the new root, normally `/sys/bus/pci/devices` */ );

    /// Get the root of the PCI device tree.
    const std::string &sysfsPath() const;

    /// Check whether a port is configured.
    bool enabled() const;

    ///@}

    /// Get the description of the most recent failure.
    const std::string &error() const;

    /// Check whether a string is a PCI address of the form `0000:42:09.0`.
    static bool validBdf( const std::string &bdf /**< [in] the string to check */ );

    /// Check that the port exists and is a downstream port of a supported switch card.
    /**
     * \returns 0 if the port is valid
     * \returns -1 otherwise
     */
    int validatePort();

    /// Find the camera function enumerated below the port.
    /**
     * \returns 0 if found
     * \returns 1 if no camera is enumerated below the port
     * \returns -1 on error
     */
    int cameraDevice( std::string &bdf /**< [out] PCI address of the camera, if found */ );

    /// Determine the condition of the camera below the port.
    pcieCamera cameraState();

    /// Read the port's PCIe Data Link Layer Link Active status.
    /**
     * \returns 1 if the link is active
     * \returns 0 if the link is down
     * \returns -1 if the port does not report link activity or on error
     */
    int linkActive();

    /// Check whether the port shows that its camera is off.
    /** Used to verify that the configured port belongs to this camera: while this camera's power is off its link
     * must be down, or its camera must be absent or unresponsive.
     *
     * \returns 1 if the port is down
     * \returns 0 if the port's camera is still up
     * \returns -1 on error
     */
    int portDown();

    /// Perform a secondary bus reset of the port.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int secondaryBusReset();

    /// Re-enumerate the camera below the port.
    /** Resets the port, removes the camera function if present, and rescans the port.  The caller must ensure the
     * camera's device node is not open in any process.
     *
     * \returns 0 if a camera is enumerated afterwards
     * \returns 1 if no camera is enumerated afterwards
     * \returns -1 on error
     */
    int hotplug();

  protected:
    /// Get the sysfs directory of the port.
    std::string portPath() const;

    /// Record a failure.
    /**
     * \returns -1
     */
    int fail( const std::string &message /**< [in] description of the failure */ );

    /// Read a hexadecimal sysfs attribute such as `vendor`.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int readHex( const std::string &path, /**< [in] the attribute file */
                 uint64_t          &value /**< [out] the value read */ );

    /// Read a little-endian value from a device's config space.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    template <typename T>
    int readConfigValue( const std::string &device, /**< [in] sysfs directory of the device */
                         size_t             offset, /**< [in] byte offset in config space */
                         T                 &value /**< [out] the value read */ );

    /// Write a sysfs attribute.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int writeAttribute( const std::string &path, /**< [in] the attribute file */
                                const std::string &value /**< [in] the value to write */ );

    /// Read bytes from a device's config space.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int readConfig( const std::string &device, /**< [in] sysfs directory of the device */
                            size_t             offset, /**< [in] byte offset in config space */
                            uint8_t           *data,   /**< [out] the bytes read */
                            size_t             size /**< [in] number of bytes to read */ );

    /// Write bytes to a device's config space.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int writeConfig( const std::string &device, /**< [in] sysfs directory of the device */
                             size_t             offset, /**< [in] byte offset in config space */
                             const uint8_t     *data,   /**< [in] the bytes to write */
                             size_t             size /**< [in] number of bytes to write */ );

    /// Sleep between hotplug steps.
    virtual void pause( unsigned ms /**< [in] the time to sleep in ms */ );
};

/// Exclusive, non-blocking advisory lock shared by all pvcamCtrl instances on a host.
/** Serializes camera enumeration in one instance against PCIe hotplug in another, since enumeration opens every
 * camera and removing an open camera crashes the kernel.  The lock is released on destruction.
 *
 * \ingroup pvcamCtrl
 */
class pvcamPcieLock
{
  protected:
    int m_fd{ -1 }; ///< Descriptor of the locked file, or -1 if not locked.

    int m_errno{ 0 }; ///< The errno from a failed open or lock.

  public:
    /// Open the lock file, creating it if needed, and try to lock it.
    explicit pvcamPcieLock( const std::string &path /**< [in] the lock file */ );

    /// Release the lock.
    ~pvcamPcieLock();

    /// Check whether the lock is held.
    bool locked() const;

    /// Check whether locking failed because another process holds the lock.
    bool busy() const;

    /// Get the errno from a failed open or lock.
    int error() const;
};

inline int pvcamPcie::port( const std::string &bdf )
{
    if( !bdf.empty() && !validBdf( bdf ) )
    {
        return fail( "invalid PCI address: '" + bdf + "'" );
    }

    m_port = bdf;

    return 0;
}

inline const std::string &pvcamPcie::port() const
{
    return m_port;
}

inline void pvcamPcie::sysfsPath( const std::string &path )
{
    m_sysfsPath = path;
}

inline const std::string &pvcamPcie::sysfsPath() const
{
    return m_sysfsPath;
}

inline bool pvcamPcie::enabled() const
{
    return !m_port.empty();
}

inline const std::string &pvcamPcie::error() const
{
    return m_error;
}

inline bool pvcamPcie::validBdf( const std::string &bdf )
{
    static constexpr char pattern[] = "hhhh:hh:hh.d";

    if( bdf.size() != sizeof( pattern ) - 1 )
    {
        return false;
    }

    for( size_t n = 0; n < bdf.size(); ++n )
    {
        char c  = bdf[n];
        bool ok = pattern[n] == 'h' ? ( isdigit( c ) || ( c >= 'a' && c <= 'f' ) )
                  : pattern[n] == 'd' ? ( c >= '0' && c <= '7' )
                                      : c == pattern[n];
        if( !ok )
        {
            return false;
        }
    }

    return true;
}

inline int pvcamPcie::validatePort()
{
    uint64_t vendor, device;

    if( readHex( portPath() + "/vendor", vendor ) < 0 || readHex( portPath() + "/device", device ) < 0 )
    {
        return fail( "PCIe port " + m_port + " not found: " + m_error );
    }

    for( auto &id : c_cardIds )
    {
        if( vendor == id[0] && device == id[1] )
        {
            return 0;
        }
    }

    char ids[16];
    snprintf( ids, sizeof( ids ), "%04x:%04x", static_cast<unsigned>( vendor ), static_cast<unsigned>( device ) );

    return fail( "PCIe port " + m_port + " has ID " + ids + ", which is not a supported PVCAM switch card" );
}

inline int pvcamPcie::cameraDevice( std::string &bdf )
{
    std::error_code ec;
    for( std::filesystem::directory_iterator it( portPath(), ec ), end; !ec && it != end; it.increment( ec ) )
    {
        std::string name = it->path().filename();
        uint64_t    vendor, device;

        if( validBdf( name ) && readHex( it->path().string() + "/vendor", vendor ) == 0 &&
            readHex( it->path().string() + "/device", device ) == 0 && vendor == c_cameraVendor &&
            device == c_cameraDevice )
        {
            bdf = name;
            return 0;
        }
    }

    if( ec )
    {
        return fail( "listing PCIe port " + m_port + ": " + ec.message() );
    }

    return 1;
}

inline pcieCamera pvcamPcie::cameraState()
{
    std::string camera;
    int         rv = cameraDevice( camera );

    if( rv != 0 )
    {
        return rv > 0 ? pcieCamera::none : pcieCamera::error;
    }

    std::string device = portPath() + "/" + camera;
    uint16_t    vendor, command;
    uint32_t    bar0, bar1 = 0;

    if( readConfigValue( device, 0x00, vendor ) < 0 )
    {
        return pcieCamera::error;
    }

    if( vendor == 0xffff )
    {
        return pcieCamera::unresponsive;
    }

    // A 64-bit memory BAR0 (type bits 2:1 == 2) continues in BAR1.
    if( readConfigValue( device, 0x04, command ) < 0 || readConfigValue( device, 0x10, bar0 ) < 0 ||
        ( ( bar0 & 0x6 ) == 0x4 && readConfigValue( device, 0x14, bar1 ) < 0 ) )
    {
        return pcieCamera::error;
    }

    uint64_t assigned;
    if( readHex( device + "/resource", assigned ) < 0 )
    {
        return pcieCamera::error;
    }

    uint64_t address = ( static_cast<uint64_t>( bar1 ) << 32 ) | ( bar0 & ~0xFu );

    // A power cycle clears Memory Space Enable (command bit 1) and the BARs assigned by the kernel.
    if( !( command & 0x2 ) || address != assigned )
    {
        return pcieCamera::stale;
    }

    return pcieCamera::healthy;
}

inline int pvcamPcie::linkActive()
{
    uint16_t status;
    uint8_t  ptr;

    if( readConfigValue( portPath(), 0x06, status ) < 0 || readConfigValue( portPath(), 0x34, ptr ) < 0 )
    {
        return -1;
    }

    // Walk the capability list (status bit 4) for the PCI Express capability (ID 0x10).
    for( int n = 0; ( status & 0x10 ) && ptr >= 0x40 && n < 48; ++n )
    {
        uint8_t id, next;
        if( readConfigValue( portPath(), ptr, id ) < 0 || readConfigValue( portPath(), ptr + 1, next ) < 0 )
        {
            return -1;
        }

        if( id == 0x10 )
        {
            uint32_t linkCap;
            uint16_t linkStatus;
            if( readConfigValue( portPath(), ptr + 0x0C, linkCap ) < 0 ||
                readConfigValue( portPath(), ptr + 0x12, linkStatus ) < 0 )
            {
                return -1;
            }

            // Link Capabilities bit 20: Data Link Layer Link Active Reporting Capable.
            if( !( linkCap & ( 1u << 20 ) ) )
            {
                return fail( "PCIe port " + m_port + " does not report link activity" );
            }

            // Link Status bit 13: Data Link Layer Link Active.
            return ( linkStatus & 0x2000 ) ? 1 : 0;
        }

        ptr = next & 0xFC;
    }

    return fail( "PCIe port " + m_port + " has no PCI Express capability" );
}

inline int pvcamPcie::portDown()
{
    int link = linkActive();
    if( link >= 0 )
    {
        return link == 0 ? 1 : 0;
    }

    switch( cameraState() )
    {
    case pcieCamera::error:
        return -1;
    case pcieCamera::none:
    case pcieCamera::unresponsive:
        return 1;
    default:
        return 0;
    }
}

inline int pvcamPcie::secondaryBusReset()
{
    // Bridge Control register (type 1 header offset 0x3E), bit 6: Secondary Bus Reset.
    uint16_t control;
    if( readConfigValue( portPath(), 0x3E, control ) < 0 )
    {
        return -1;
    }

    uint8_t asserted[2] = { static_cast<uint8_t>( control | 0x40 ), static_cast<uint8_t>( control >> 8 ) };
    uint8_t restored[2] = { static_cast<uint8_t>( control ), static_cast<uint8_t>( control >> 8 ) };

    if( writeConfig( portPath(), 0x3E, asserted, 2 ) < 0 )
    {
        return -1;
    }

    pause( m_resetHoldMs );

    if( writeConfig( portPath(), 0x3E, restored, 2 ) < 0 )
    {
        return -1;
    }

    pause( m_settleMs );

    return 0;
}

inline int pvcamPcie::hotplug()
{
    if( secondaryBusReset() < 0 )
    {
        return -1;
    }

    std::string camera;
    int         rv = cameraDevice( camera );
    if( rv < 0 || ( rv == 0 && writeAttribute( portPath() + "/" + camera + "/remove", "1" ) < 0 ) )
    {
        return -1;
    }

    pause( m_settleMs );

    if( writeAttribute( portPath() + "/rescan", "1" ) < 0 )
    {
        return -1;
    }

    pause( m_settleMs );

    return cameraDevice( camera );
}

inline std::string pvcamPcie::portPath() const
{
    return m_sysfsPath + "/" + m_port;
}

inline int pvcamPcie::fail( const std::string &message )
{
    m_error = message;
    return -1;
}

inline int pvcamPcie::readHex( const std::string &path, uint64_t &value )
{
    std::ifstream fin( path );
    std::string   text;

    if( !( fin >> text ) )
    {
        return fail( "reading " + path );
    }

    try
    {
        size_t used;
        value = std::stoull( text, &used, 16 );
        if( used == text.size() )
        {
            return 0;
        }
    }
    catch( const std::exception & )
    {
    }

    return fail( "invalid value '" + text + "' in " + path );
}

template <typename T>
int pvcamPcie::readConfigValue( const std::string &device, size_t offset, T &value )
{
    uint8_t bytes[sizeof( T )];
    if( readConfig( device, offset, bytes, sizeof( T ) ) < 0 )
    {
        return -1;
    }

    value = 0;
    for( size_t n = 0; n < sizeof( T ); ++n )
    {
        value |= static_cast<T>( static_cast<T>( bytes[n] ) << ( 8 * n ) );
    }

    return 0;
}

inline int pvcamPcie::writeAttribute( const std::string &path, const std::string &value )
{
    int fd = open( path.c_str(), O_WRONLY | O_CLOEXEC );
    if( fd < 0 )
    {
        return fail( "opening " + path + ": " + strerror( errno ) );
    }

    ssize_t written = write( fd, value.data(), value.size() );
    int     err     = errno;
    close( fd );

    if( written != static_cast<ssize_t>( value.size() ) )
    {
        return fail( "writing " + path + ": " + ( written < 0 ? strerror( err ) : "short write" ) );
    }

    return 0;
}

inline int pvcamPcie::readConfig( const std::string &device, size_t offset, uint8_t *data, size_t size )
{
    std::string path = device + "/config";
    int         fd   = open( path.c_str(), O_RDONLY | O_CLOEXEC );
    if( fd < 0 )
    {
        return fail( "opening " + path + ": " + strerror( errno ) );
    }

    ssize_t got = pread( fd, data, size, offset );
    int     err = errno;
    close( fd );

    if( got != static_cast<ssize_t>( size ) )
    {
        return fail( "reading " + path + ": " + ( got < 0 ? strerror( err ) : "short read" ) );
    }

    return 0;
}

inline int pvcamPcie::writeConfig( const std::string &device, size_t offset, const uint8_t *data, size_t size )
{
    std::string path = device + "/config";
    int         fd   = open( path.c_str(), O_WRONLY | O_CLOEXEC );
    if( fd < 0 )
    {
        return fail( "opening " + path + ": " + strerror( errno ) );
    }

    ssize_t written = pwrite( fd, data, size, offset );
    int     err     = errno;
    close( fd );

    if( written != static_cast<ssize_t>( size ) )
    {
        return fail( "writing " + path + ": " + ( written < 0 ? strerror( err ) : "short write" ) );
    }

    return 0;
}

inline void pvcamPcie::pause( unsigned ms )
{
    std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );
}

inline pvcamPcieLock::pvcamPcieLock( const std::string &path )
{
    m_fd = open( path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0664 );
    if( m_fd < 0 )
    {
        m_errno = errno;
        return;
    }

    if( flock( m_fd, LOCK_EX | LOCK_NB ) < 0 )
    {
        m_errno = errno;
        close( m_fd );
        m_fd = -1;
    }
}

inline pvcamPcieLock::~pvcamPcieLock()
{
    if( m_fd >= 0 )
    {
        close( m_fd );
    }
}

inline bool pvcamPcieLock::locked() const
{
    return m_fd >= 0;
}

inline bool pvcamPcieLock::busy() const
{
    return m_errno == EWOULDBLOCK;
}

inline int pvcamPcieLock::error() const
{
    return m_errno;
}

} // namespace app
} // namespace MagAOX

#endif // pvcamPcie_hpp
