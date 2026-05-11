/** \file edtinc.h
 * \brief Shared EDT SDK stub declarations for MagAO-X test builds.
 * \author OpenAI Codex
 */

#ifndef tests_edtinc_h
#define tests_edtinc_h

typedef unsigned int  uint;
typedef unsigned char u_char;

/// Stub EDT device handle type used by unit tests.
struct EdtDev
{
    int unused;
};

/// Stub EDT camera handle type used by unit tests.
struct PdvDev
{
    int unused;
};

/// Stub EDT dependent-configuration type used by unit tests.
struct Dependent
{
    int unused;
};

/// Stub EDT configuration info structure used by unit tests.
struct Edtinfo
{
    int unused;
};

/// Default EDT interface name used by the production code.
#define EDT_INTERFACE "pdv"

#ifdef __cplusplus
extern "C"
{
#endif

    /// Allocate a stub EDT dependent configuration object.
    Dependent *pdv_alloc_dependent();

    /// Read a stub EDT configuration file.
    int pdv_readcfg( const char *configFile, Dependent *dd_p, Edtinfo *edtinfo );

    /// Open a stub EDT device channel.
    EdtDev *edt_open_channel( const char *deviceName, int unit, int channel );

    /// Report the last stub EDT error message.
    void edt_perror( char *errstr );

    /// Initialize a stub EDT camera instance.
    int pdv_initcam( EdtDev     *edt_p,
                     Dependent  *dd_p,
                     int         unit,
                     Edtinfo    *edtinfo,
                     const char *configFile,
                     char       *bitdir,
                     int         pdv_debug );

    /// Close a stub EDT device channel.
    void edt_close( EdtDev *edt_p );

    /// Open a stub PDV device handle.
    PdvDev *pdv_open_channel( const char *deviceName, int unit, int channel );

    /// Close a stub PDV device handle.
    void pdv_close( PdvDev *pdv_p );

    /// Flush the stub PDV FIFO.
    void pdv_flush_fifo( PdvDev *pdv_p );

    /// Enable stub PDV serial reads.
    void pdv_serial_read_enable( PdvDev *pdv_p );

    /// Return the stub PDV frame width.
    int pdv_get_width( PdvDev *pdv_p );

    /// Return the stub PDV frame height.
    int pdv_get_height( PdvDev *pdv_p );

    /// Return the stub PDV frame bit depth.
    int pdv_get_depth( PdvDev *pdv_p );

    /// Return the stub PDV camera type string.
    char *pdv_get_cameratype( PdvDev *pdv_p );

    /// Configure the stub PDV ring-buffer depth.
    void pdv_multibuf( PdvDev *pdv_p, int numBuffs );

    /// Start stub PDV acquisition for multiple images.
    void pdv_start_images( PdvDev *pdv_p, int numBuffs );

    /// Wait for the last stub PDV image and fill the DMA timestamp.
    u_char *pdv_wait_last_image_timed( PdvDev *pdv_p, uint dmaTimeStamp[2] );

    /// Start acquisition of the next stub PDV image.
    void pdv_start_image( PdvDev *pdv_p );

    /// Read bytes from the stub PDV serial channel.
    int pdv_serial_read( PdvDev *pdv_p, char *buf, int size );

    /// Send a command over the stub PDV serial channel.
    int pdv_serial_command( PdvDev *pdv_p, const char *command );

    /// Wait for serial data on the stub PDV channel.
    int pdv_serial_wait( PdvDev *pdv_p, int timeout, int count );

    /// Return the configured stub PDV serial terminator.
    int pdv_get_waitchar( PdvDev *pdv_p, u_char *waitc );

#ifdef __cplusplus
}
#endif

#endif // tests_edtinc_h
