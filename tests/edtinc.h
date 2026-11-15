/** \file edtinc.h
 * \brief Shared EDT SDK stub declarations for MagAO-X test builds.
 */

#ifndef tests_edtinc_h
#define tests_edtinc_h

typedef unsigned int  uint;
typedef unsigned char u_char;

/// Stub EDT device handle type used by unit tests.
struct EdtDev
{
    int unused; ///< Unused storage distinguishing the simulated SDK structures.
};

/// Forward declaration of the active stub camera configuration.
struct Dependent;

/// Stub EDT camera handle type used by unit tests.
struct PdvDev
{
    int        unused;                   ///< Unused storage distinguishing the simulated SDK structures.
    Dependent *dd_p{ nullptr };          ///< Simulated active camera configuration.
    int        ring_buffer_numbufs{ 0 }; ///< Number of allocated DMA ring buffers.
};

/// Stub EDT dependent-configuration type used by unit tests.
struct Dependent
{
    int  unused;               ///< Unused storage distinguishing the simulated SDK structures.
    int  force_single{ 0 };    ///< Configuration requires immediate single-image acquisition.
    char serial_trigger[32]{}; ///< Optional serial acquisition-trigger command.
};

/// Stub EDT configuration info structure used by unit tests.
struct Edtinfo
{
    int unused; ///< Unused storage distinguishing the simulated SDK structures.
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
    int pdv_readcfg( const char *configFile /**< [in] Configuration filename. */,
                     Dependent  *dd_p /**< [in,out] Simulated dependent configuration. */,
                     Edtinfo    *edtinfo /**< [in,out] Simulated camera configuration. */ );

    /// Open a stub EDT device channel.
    EdtDev *edt_open_channel( const char *deviceName /**< [in] Interface name. */,
                              int         unit /**< [in] Board unit. */,
                              int         channel /**< [in] Board channel. */ );

    /// Report the last stub EDT error message.
    void edt_perror( char *errstr /**< [in] Error-message prefix. */ );

    /// Initialize a stub EDT camera instance.
    int pdv_initcam( EdtDev     *edt_p /**< [in] Simulated EDT handle. */,
                     Dependent  *dd_p /**< [in,out] Simulated dependent configuration. */,
                     int         unit /**< [in] Board unit. */,
                     Edtinfo    *edtinfo /**< [in,out] Simulated camera configuration. */,
                     const char *configFile /**< [in] Configuration filename. */,
                     char       *bitdir /**< [in] Firmware directory. */,
                     int         pdv_debug /**< [in] SDK debug flag. */ );

    /// Close a stub EDT device channel.
    void edt_close( EdtDev *edt_p /**< [in] Simulated EDT handle. */ );

    /// Open a stub PDV device handle.
    PdvDev *pdv_open_channel( const char *deviceName /**< [in] Interface name. */,
                              int         unit /**< [in] Board unit. */,
                              int         channel /**< [in] Board channel. */ );

    /// Close a stub PDV device handle.
    void pdv_close( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Flush the stub PDV FIFO.
    void pdv_flush_fifo( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Enable stub PDV serial reads.
    void pdv_serial_read_enable( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Return the stub PDV frame width.
    int pdv_get_width( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Return the stub PDV frame height.
    int pdv_get_height( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Return the stub PDV frame bit depth.
    int pdv_get_depth( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Return the stub PDV camera type string.
    char *pdv_get_cameratype( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Configure the stub PDV ring-buffer depth.
    void pdv_multibuf( PdvDev *pdv_p /**< [in] Simulated camera handle. */,
                       int     numBuffs /**< [in] Ring capacity in frames. */ );

    /// Start stub PDV acquisition for multiple images.
    void pdv_start_images( PdvDev *pdv_p /**< [in] Simulated camera handle. */,
                           int     numBuffs /**< [in] Ring capacity in frames. */ );

    /// Wait for the last stub PDV image and fill the DMA timestamp.
    u_char *pdv_wait_last_image_timed( PdvDev *pdv_p /**< [in] Simulated camera handle. */,
                                       uint    dmaTimeStamp[2] /**< [out] SDK seconds/nanoseconds timestamp. */ );

    /// Start acquisition of the next stub PDV image.
    void pdv_start_image( PdvDev *pdv_p /**< [in] Simulated camera handle. */ );

    /// Return the cumulative stub acquisition timeout count.
    int pdv_timeouts( PdvDev *pdv_p /**< [in] Stub camera handle. */ );

    /// Read bytes from the stub PDV serial channel.
    int pdv_serial_read( PdvDev *pdv_p /**< [in] Simulated camera handle. */,
                         char   *buf /**< [out] Destination bytes. */,
                         int     size /**< [in] Destination capacity. */ );

    /// Send a command over the stub PDV serial channel.
    int pdv_serial_command( PdvDev     *pdv_p /**< [in] Simulated camera handle. */,
                            const char *command /**< [in] Serial command. */ );

    /// Wait for serial data on the stub PDV channel.
    int pdv_serial_wait( PdvDev *pdv_p /**< [in] Simulated camera handle. */,
                         int     timeout /**< [in] Wait duration in milliseconds. */,
                         int     count /**< [in] Requested byte count. */ );

    /// Return the configured stub PDV serial terminator.
    int pdv_get_waitchar( PdvDev *pdv_p /**< [in] Simulated camera handle. */,
                          u_char *waitc /**< [out] Serial terminator. */ );

#ifdef __cplusplus
}
#endif

#endif // tests_edtinc_h
