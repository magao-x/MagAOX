/** \file sparkleClock.hpp
 * \brief The MagAO-X DM speckle maker header file
 *
 * \ingroup sparkleClock_files
 */

#ifndef sparkleClock_hpp
#define sparkleClock_hpp

#include <mx/improc/eigenCube.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/stringUtils.hpp>
#include <mx/sys/timeUtils.hpp>
#include <mx/sigproc/fourierModes.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup sparkleClock
 * \brief The DM speckle maker app
 *
 * Creates a set of fourier modes to generate speckles, then applies them to a DM channel
 * at a specified rate.
 *
 *
 * <a href="../handbook/operating/software/apps/sparkleClock.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup sparkleClock_files
 * \ingroup sparkleClock
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X DM mode commander
/**
 * \ingroup sparkleClock
 */
class sparkleClock : public MagAOXApp<true>, public dev::telemeter<sparkleClock>
{

    typedef float realT;

    friend class dev::telemeter<sparkleClock>;
    friend class sparkleClock_test;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_dmName; ///< The descriptive name of this dm. Default is the channel name.

    std::string m_dmChannelName; ///< The name of the DM channel to write to.

    std::string m_dmTriggerChannel; ///< The DM channel to monitor as a trigger

    float m_triggerDelay{ 0 }; // 0.000375- 0.5/2000.;

    int m_triggerSemaphore{ 9 }; ///< The semaphore to use (default 9)

    bool m_trigger{ true }; ///< Run in trigger mode if true (default)

    realT m_interval{ 1.0 }; ///< time for one complete cycle of the sparkle clock (e.g. exposure time of some camera) (default 1.0)
    realT m_separation_1{ 10.0 }; ///< The radial separation of the first set of speckles (default 10.0)
    realT m_separation_2{ 20.0 }; ///< The radial separation of the second set of speckles (default 20.0)

    realT m_angle{ 0.0 }; ///< The angle of the speckle pattern c.c.w. from up on camsci1/2 (default 0.0)

    realT m_angleOffset{ 28.0 }; ///< The calibration offset of angle so that up on camsci1/2 is 0

    realT m_amp{ 0.01 }; ///< The speckle amplitude on the DM

    bool m_cross{ true }; ///< If true, also apply the cross speckles rotated by 90 degrees

    realT m_frequency{ 2000 };         ///< The frequency to modulate at if not triggering (default 2000 Hz)
    realT m_sparkleClockInterval{ 1 }; ///< The time in seconds during which the sparkle clock should sweep out both
                                       ///< cycles (e.g. the exposure time of your science cam)

    unsigned m_dwell{ 1 }; ///< The dwell time for each speckle, or for how many frames it is held.

    int m_single{ -1 }; ///< if >= 0 a single frame is non-zero.
    ///@}

    mx::improc::eigenCube<realT> m_shapes;

    IMAGE m_imageStream;
    uint32_t m_width{ 0 };  ///< The width of the image
    uint32_t m_height{ 0 }; ///< The height of the image.

    IMAGE m_triggerStream;

    uint8_t m_dataType{ 0 }; ///< The ImageStreamIO type code.
    size_t m_typeSize{ 0 };  ///< The size of the type, in bytes.

    bool m_opened{ true };
    bool m_restart{ false };

    bool m_modulating{ false };

    bool m_restartSp{ false };

  public:
    /// Default c'tor.
    sparkleClock();

    /// D'tor, declared and defined for noexcept.
    ~sparkleClock() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for sparkleClock.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

  protected:
    int generateSparkleClock();

    /** \name Modulator Thread
     * This thread sends the signal to the dm at the prescribed frequency
     *
     * @{
     */
    int m_modThreadPrio{ 60 }; ///< Priority of the modulator thread, should normally be > 00.

    std::string m_modThreadCpuset; ///< The cpuset for the modulator thread.

    std::thread m_modThread; ///< A separate thread for the modulation

    bool m_modThreadInit{ true }; ///< Synchronizer to ensure f.g. thread initializes before doing dangerous things.

    pid_t m_modThreadID{ 0 }; ///< Modulate thread PID.

    pcf::IndiProperty m_modThreadProp; ///< The property to hold the modulator thread details.

    /// Thread starter, called by modThreadStart on thread construction.  Calls modThreadExec.
    static void modThreadStart( sparkleClock *d /**< [in] a pointer to a sparkleClock instance (normally this) */ );

    /// Execute the frame grabber main loop.
    void modThreadExec();

    ///@}

    // INDI:
  protected:
    // declare our properties
    pcf::IndiProperty m_indiP_dm;
    pcf::IndiProperty m_indiP_trigger;
    pcf::IndiProperty m_indiP_delay;
    pcf::IndiProperty m_indiP_separation_1;
    pcf::IndiProperty m_indiP_separation_2;
    pcf::IndiProperty m_indiP_angle;
    pcf::IndiProperty m_indiP_amp;
    pcf::IndiProperty m_indiP_cross;
    pcf::IndiProperty m_indiP_frequency;
    pcf::IndiProperty m_indiP_interval;
    pcf::IndiProperty m_indiP_dwell;
    pcf::IndiProperty m_indiP_single;
    pcf::IndiProperty m_indiP_modulating;
    pcf::IndiProperty m_indiP_zero;

  public:
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_trigger );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_delay );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_separation_1 );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_separation_2 );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_angle );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_cross );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_amp );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_frequency );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_interval );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_dwell );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_single );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_modulating );
    INDI_NEWCALLBACK_DECL( sparkleClock, m_indiP_zero );

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_sparkleclock * );

    int recordSparkleClock( bool force = false );

    ///@}
};

sparkleClock::sparkleClock() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void sparkleClock::setupConfig()
{
    config.add( "dm.channelName",
                "",
                "dm.channelName",
                argType::Required,
                "dm",
                "channelName",
                false,
                "string",
                "The name of the DM channel to write to." );
    config.add( "dm.triggerChannel",
                "",
                "dm.triggerChannel",
                argType::Required,
                "dm",
                "triggerChannel",
                false,
                "string",
                "The name of the DM channel to trigger on." );
    config.add( "dm.triggerSemaphore",
                "",
                "dm.triggerSemaphore",
                argType::Required,
                "dm",
                "triggerSemaphore",
                false,
                "int",
                "The semaphore to use (default 9)." );
    config.add( "dm.trigger",
                "",
                "dm.trigger",
                argType::True,
                "dm",
                "trigger",
                false,
                "bool",
                "Run in trigger mode if true (default)." );
    config.add( "dm.triggerDelay",
                "",
                "dm.triggerDelay",
                argType::Required,
                "dm",
                "triggerDelay",
                false,
                "float",
                "Delay to apply to the trigger." );

    config.add( "dm.separation_1",
                "",
                "dm.separation_1",
                argType::Required,
                "dm",
                "separation",
                false,
                "float",
                "The radial separation of the first set of speckles (default 10.0)." );
    config.add( "dm.separation_2",
                "",
                "dm.separation_2",
                argType::Required,
                "dm",
                "separation",
                false,
                "float",
                "The radial separation of the first set of speckles (default 20.0)." );
    config.add( "dm.angle",
                "",
                "dm.angle",
                argType::Required,
                "dm",
                "angle",
                false,
                "float",
                "The angle of the speckle pattern c.c.w. from up on camsci1/2 (default 0.0)." );
    config.add( "dm.angleOffset",
                "",
                "dm.angleOffset",
                argType::Required,
                "dm",
                "angleOffset",
                false,
                "float",
                "The calibration offset of angle so that up on camsci1/2 is 0." );
    config.add( "dm.amp",
                "",
                "dm.amp",
                argType::Required,
                "dm",
                "amp",
                false,
                "float",
                "The speckle amplitude on the DM (default 0.01)." );
    config.add( "dm.cross",
                "",
                "dm.cross",
                argType::True,
                "dm",
                "cross",
                false,
                "bool",
                "If true, also apply the cross speckles rotated by 90 degrees." );

    config.add( "dm.frequency",
                "",
                "dm.frequency",
                argType::Required,
                "dm",
                "frequency",
                false,
                "float",
                "The frequency to modulate at if not triggering (default 2000 Hz)." );

    config.add( "dm.dwell",
                "",
                "dm.dwell",
                argType::True,
                "dm",
                "dwell",
                false,
                "int",
                "The dwell time for each speckle, or for how many frames it is held. Default=1." );

    config.add( "modulator.threadPrio",
                "",
                "modulator.threadPrio",
                argType::Required,
                "modulator",
                "threadPrio",
                false,
                "int",
                "The real-time priority of the modulator thread." );

    config.add( "modulator.cpuset",
                "",
                "modulator.cpuset",
                argType::Required,
                "modulator",
                "cpuset",
                false,
                "string",
                "The cpuset to assign the modulator thread to." );
}

int sparkleClock::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_dmChannelName, "dm.channelName" );

    m_dmName = m_dmChannelName;
    _config( m_dmName, "dm.name" );

    _config( m_dmTriggerChannel, "dm.triggerChannel" );

    _config( m_triggerSemaphore, "dm.triggerSemaphore" );

    if( _config.isSet( "dm.trigger" ) )
    {
        _config( m_trigger, "dm.trigger" );
    }

    _config( m_triggerDelay, "dm.triggerDelay" );

    _config( m_separation_1, "dm.separation_1" );
    _config( m_separation_2, "dm.separation_2" );
    _config( m_angle, "dm.angle" );
    _config( m_angleOffset, "dm.angleOffset" );
    _config( m_amp, "dm.amp" );

    if( _config.isSet( "dm.cross" ) )
    {
        _config( m_cross, "dm.cross" );
    }

    _config( m_frequency, "dm.frequency" );
    _config( m_dwell, "dm.dwell" );
    _config( m_modThreadPrio, "modulator.threadPrio" );
    _config( m_modThreadCpuset, "modulator.cpuset" );

    dev::telemeter<sparkleClock>::loadConfig( _config );
    return 0;
}

void sparkleClock::loadConfig()
{
    loadConfigImpl( config );
}

int sparkleClock::appStartup()
{

    REG_INDI_NEWPROP_NOCB( m_indiP_dm, "dm", pcf::IndiProperty::Text );
    m_indiP_dm.add( pcf::IndiElement( "name" ) );
    m_indiP_dm["name"] = m_dmName;
    m_indiP_dm.add( pcf::IndiElement( "channel" ) );
    m_indiP_dm["channel"] = m_dmChannelName;

    createStandardIndiNumber<float>( m_indiP_delay, "delay", 0, 0, 1, "%f" );
    m_indiP_delay["current"] = m_triggerDelay;
    m_indiP_delay["target"] = m_triggerDelay;
    registerIndiPropertyNew( m_indiP_delay, INDI_NEWCALLBACK( m_indiP_delay ) );

    createStandardIndiNumber<float>( m_indiP_separation_1, "separation_1", 2, 24, 100, "%f" );
    m_indiP_separation_1["current"] = m_separation_1;
    m_indiP_separation_1["target"] = m_separation_1;
    registerIndiPropertyNew( m_indiP_separation_1, INDI_NEWCALLBACK( m_indiP_separation_1 ) );

    createStandardIndiNumber<float>( m_indiP_separation_2, "separation_2", 2, 24, 100, "%f" );
    m_indiP_separation_2["current"] = m_separation_2;
    m_indiP_separation_2["target"] = m_separation_2;
    registerIndiPropertyNew( m_indiP_separation_2, INDI_NEWCALLBACK( m_indiP_separation_2 ) );

    createStandardIndiNumber<float>( m_indiP_angle, "angle", 0, 0, 100, "%f" );
    m_indiP_angle["current"] = m_angle;
    m_indiP_angle["target"] = m_angle;
    registerIndiPropertyNew( m_indiP_angle, INDI_NEWCALLBACK( m_indiP_angle ) );

    createStandardIndiToggleSw( m_indiP_cross, "cross" );
    if( registerIndiPropertyNew( m_indiP_cross, INDI_NEWCALLBACK( m_indiP_cross ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
    if( m_cross )
    {
        m_indiP_cross["toggle"] = pcf::IndiElement::On;
    }
    else
    {
        m_indiP_cross["toggle"] = pcf::IndiElement::Off;
    }

    createStandardIndiNumber<float>( m_indiP_amp, "amp", -1, 0, 1, "%f" );
    m_indiP_amp["current"] = m_amp;
    m_indiP_amp["target"] = m_amp;
    registerIndiPropertyNew( m_indiP_amp, INDI_NEWCALLBACK( m_indiP_amp ) );

    createStandardIndiNumber<float>( m_indiP_frequency, "frequency", 0, 0, 10000, "%f" );
    m_indiP_frequency["current"] = m_frequency;
    m_indiP_frequency["target"] = m_frequency;
    registerIndiPropertyNew( m_indiP_frequency, INDI_NEWCALLBACK( m_indiP_frequency ) );

    createStandardIndiNumber<float>( m_indiP_interval, "interval", 0, 0, 10000, "%f" );
    m_indiP_frequency["current"] = m_sparkleClockInterval;
    m_indiP_frequency["target"] = m_sparkleClockInterval;
    registerIndiPropertyNew( m_indiP_interval, INDI_NEWCALLBACK( m_indiP_interval ) );

    createStandardIndiToggleSw( m_indiP_trigger, "trigger" );
    if( registerIndiPropertyNew( m_indiP_trigger, INDI_NEWCALLBACK( m_indiP_trigger ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }
    if( m_trigger )
    {
        m_indiP_trigger["toggle"] = pcf::IndiElement::On;
    }
    else
    {
        m_indiP_trigger["toggle"] = pcf::IndiElement::Off;
    }

    createStandardIndiNumber<int>( m_indiP_dwell, "dwell", 1, 100, 1, "%d" );
    m_indiP_dwell["current"] = m_dwell;
    m_indiP_dwell["target"] = m_dwell;
    registerIndiPropertyNew( m_indiP_dwell, INDI_NEWCALLBACK( m_indiP_dwell ) );

    createStandardIndiNumber<int>( m_indiP_single, "single", -1, 3, 1, "%d" );
    m_indiP_single["current"] = m_single;
    m_indiP_single["target"] = m_single;
    registerIndiPropertyNew( m_indiP_single, INDI_NEWCALLBACK( m_indiP_single ) );

    createStandardIndiToggleSw( m_indiP_modulating, "modulating" );
    if( registerIndiPropertyNew( m_indiP_modulating, INDI_NEWCALLBACK( m_indiP_modulating ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    createStandardIndiRequestSw( m_indiP_zero, "zero" );
    if( registerIndiPropertyNew( m_indiP_zero, INDI_NEWCALLBACK( m_indiP_zero ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( threadStart( m_modThread,
                     m_modThreadInit,
                     m_modThreadID,
                     m_modThreadProp,
                     m_modThreadPrio,
                     m_modThreadCpuset,
                     "modulator",
                     this,
                     modThreadStart ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( dev::telemeter<sparkleClock>::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    state( stateCodes::NOTCONNECTED );

    return 0;
}

int sparkleClock::appLogic()
{
    if( state() == stateCodes::NOTCONNECTED )
    {
        m_opened = false;
        m_restart = false; // Set this up front, since we're about to restart.

        if( ImageStreamIO_openIm( &m_imageStream, m_dmChannelName.c_str() ) == 0 )
        {
            if( m_imageStream.md[0].sem < 10 ) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
            {
                ImageStreamIO_closeIm( &m_imageStream );
            }
            else
            {
                m_opened = true;
            }
        }

        // Only bother to try if previous worked and we have a spec
        if( m_opened == true && m_dmTriggerChannel != "" )
        {
            if( ImageStreamIO_openIm( &m_triggerStream, m_dmTriggerChannel.c_str() ) == 0 )
            {
                if( m_triggerStream.md[0].sem <
                    10 ) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
                {
                    ImageStreamIO_closeIm( &m_triggerStream );
                    m_opened = false;
                }
            }
        }

        if( m_opened )
        {
            state( stateCodes::CONNECTED );
        }
    }

    if( state() == stateCodes::CONNECTED )
    {
        m_dataType = m_imageStream.md[0].datatype;
        m_typeSize = ImageStreamIO_typesize( m_dataType );
        m_width = m_imageStream.md[0].size[0];
        m_height = m_imageStream.md[0].size[1];

        if( m_dataType != _DATATYPE_FLOAT )
        {
            return log<text_log, -1>( "Data type of DM channel is not float.", logPrio::LOG_CRITICAL );
        }

        if( m_typeSize != sizeof( realT ) )
        {
            return log<text_log, -1>( "Type-size mismatch, realT is not float.", logPrio::LOG_CRITICAL );
        }

        state( stateCodes::READY );
    }

    if( telemeter<sparkleClock>::appLogic() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return 0;
    }

    return 0;
}

int sparkleClock::appShutdown()
{
    if( m_modThread.joinable() )
    {
        try
        {
            m_modThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }

    dev::telemeter<sparkleClock>::appShutdown();

    return 0;
}

int sparkleClock::generateSparkleClock()
{
    mx::improc::eigenImage<realT> onesp, onespC;
    onesp.resize( m_width, m_height );
    onespC.resize( m_width, m_height );

    int nFrames = 4 * static_cast<int>( m_sparkleClockInterval / 4.0 * m_frequency );
    int _nFramesRound = static_cast<int>( m_sparkleClockInterval * m_frequency );
    if( nFrames != _nFramesRound )
    {
        std::cerr << "Got nFrames = " << nFrames << " to be evenly divisible by 4, but integer rounding would give "
                  << _nFramesRound << " frames per " << m_sparkleClockInterval << " sec interval given " << m_frequency
                  << " Hz frequency\n";
    }
    m_shapes.resize( m_width, m_height, nFrames );

    // One complete cycle of the sparkle clock spins through 2pi radians, twice, in one m_sparkleClockInterval.
    // Each step in position should be 4 frames. +/-, sin/cos.
    // So we figure out the number of positions as nFrames / 2 (radii) / 4 (phase/parity).
    float nPositions = nFrames / 4 / 2;
    float angleStep = mx::math::two_pi<realT>() / nPositions;
    for( int radii = 0; radii < 2; radii++ )
    {
        float separation = (radii == 0) ? m_separation_1 : m_separation_2;
        for( int pos = 0; pos < nPositions; pos++ )
        {
            int baseIdx = ( nPositions * radii ) + pos;
            int thisIdx = baseIdx;

            realT m = separation * cos( mx::math::dtor<realT>( -1 * m_angle + m_angleOffset + angleStep * pos) );
            realT n = separation * sin( mx::math::dtor<realT>( -1 * m_angle + m_angleOffset + angleStep * pos) );

            mx::sigproc::makeFourierMode( m_shapes.image( thisIdx ), m, n, 1 );

            if( m_cross )
            {
                onesp = m_shapes.image( thisIdx );
                mx::sigproc::makeFourierMode( m_shapes.image( thisIdx ), -n, m, 1 );
                m_shapes.image( thisIdx ) += onesp;
            }

            m_shapes.image( thisIdx ) *= m_amp;
            thisIdx += 1;
            m_shapes.image( thisIdx ) = -1 * m_shapes.image( thisIdx - 1 );

            thisIdx += 1;
            mx::sigproc::makeFourierMode( m_shapes.image( thisIdx ), m, n, -1 );

            if( m_cross )
            {
                onesp = m_shapes.image( thisIdx );
                mx::sigproc::makeFourierMode( m_shapes.image( thisIdx ), -n, m, -1 );
                m_shapes.image( thisIdx ) += onesp;
            }

            m_shapes.image( thisIdx ) *= m_amp;

            thisIdx += 1;
            m_shapes.image( thisIdx ) = -1 * m_shapes.image( thisIdx - 1 );

            mx::fits::fitsFile<realT> ff;
            ff.write( "/tmp/specks.fits", m_shapes );

            // if( m_single >= 0 )
            // {
            //     for( int pp = 0; pp < m_shapes.planes(); ++pp )
            //     {
            //         if( pp != m_single )
            //         {
            //             m_shapes.image( pp ) *= 0;
            //         }
            //     }
            // }
        }
    }

    updateIfChanged( m_indiP_delay, "current", m_triggerDelay );
    updateIfChanged( m_indiP_separation_1, "current", m_separation_1 );
    updateIfChanged( m_indiP_separation_2, "current", m_separation_2 );
    updateIfChanged( m_indiP_angle, "current", m_angle );
    updateIfChanged( m_indiP_amp, "current", m_amp );
    updateIfChanged( m_indiP_frequency, "current", m_frequency );
    updateIfChanged( m_indiP_interval, "current", m_sparkleClockInterval );
    updateIfChanged( m_indiP_dwell, "current", m_dwell );
    updateIfChanged( m_indiP_single, "current", m_single );

    return 0;
}

inline void sparkleClock::modThreadStart( sparkleClock *d )
{
    d->modThreadExec();
}

inline void sparkleClock::modThreadExec()
{
    m_modThreadID = syscall( SYS_gettid );

    // Wait fpr the thread starter to finish initializing this thread.
    while( ( m_modThreadInit == true || state() != stateCodes::READY ) && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    while( m_shutdown == 0 )
    {
        if( !m_modulating && !m_shutdown ) // If we aren't modulating we sleep for 1/2 a second
        {
            mx::sys::milliSleep( 500 );
        }

        if( m_modulating && !m_shutdown )
        {
            m_restartSp = false;
            generateSparkleClock();

            int64_t freqNsec = ( 1.0 / m_frequency ) * 1e9;
            int64_t dnsec;

            int idx = 0;

            timespec modstart;
            timespec currtime;

            bool triggered = false;
            sem_t *sem = nullptr;
            if( m_dmTriggerChannel == "" )
            {
                m_trigger = false;
                indi::updateSwitchIfChanged(
                    m_indiP_trigger, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
            }
            else if( m_trigger == true )
            {
                ImageStreamIO_semflush( &m_triggerStream, m_triggerSemaphore );

                sem = m_triggerStream.semptr[m_triggerSemaphore]; ///< The semaphore to monitor for new image data
            }

            log<text_log>( "started modulating", logPrio::LOG_NOTICE );
            // To send a message
            // log<telem_sparkleclock>( { m_modulating,
            //                       m_trigger,
            //                       m_frequency,
            //                       std::vector<float>( { m_separation } ),
            //                       std::vector<float>( { m_angle } ),
            //                       std::vector<float>( { m_amp } ),
            //                       std::vector<bool>( { m_cross } ) },
            //                     logPrio::LOG_INFO );
            // // The official record:
            // recordSparkleClock( true );

            dnsec = 0;
            clock_gettime( CLOCK_REALTIME, &modstart );

            unsigned dwelled = 0;
            if( m_dwell == 0 )
                m_dwell = 1;

            float triggerDelay = m_triggerDelay / 1e6;

            double t0, t1;

            while( m_modulating && !m_restartSp && !m_shutdown )
            {
                if( m_trigger )
                {
                    timespec ts;

                    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
                    {
                        log<software_critical>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
                        return;
                    }

                    ts.tv_sec += 1;

                    if( sem_timedwait( sem, &ts ) == 0 )
                    {
                        t0 = mx::sys::get_curr_time();
                        t1 = t0;

                        while( t1 - t0 < triggerDelay )
                        {
                            double dt = ( 1e8 ) * ( triggerDelay -
                                                    ( t1 - t0 ) ); // This is 0.1 times remaining time, but in nanosecs
                            if( dt <= 0 )
                                break;
                            mx::sys::nanoSleep( dt );
                            t1 = mx::sys::get_curr_time();
                        }

                        triggered = true;
                    }
                    else
                    {
                        triggered = false;

                        // Check for why we timed out
                        if( errno == EINTR )
                            break; // This indicates signal interrupted us, time to restart or shutdown, loop will exit
                                   // normally if flags set.

                        // ETIMEDOUT just means we should wait more.
                        // Otherwise, report an error.
                        if( errno != ETIMEDOUT )
                        {
                            log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
                            break;
                        }
                    }
                }
                else
                {
                    mx::sys::nanoSleep( 0.5 * dnsec );
                    clock_gettime( CLOCK_REALTIME, &currtime );

                    dnsec =
                        ( currtime.tv_sec - modstart.tv_sec ) * 1000000000 + ( currtime.tv_nsec - modstart.tv_nsec );
                    triggered = false;
                }

                if( dwelled < m_dwell - 1 )
                {
                    ++dwelled;
                }
                else if( dnsec >= freqNsec || triggered )
                {
                    // Do the write
                    dwelled = 0;

                    m_imageStream.md->write = 1;

                    memcpy( m_imageStream.array.raw, m_shapes.image( idx ).data(), m_width * m_height * m_typeSize );

                    m_imageStream.md->atime = currtime;
                    m_imageStream.md->writetime = currtime;

                    if( !m_trigger || triggerDelay > 0 )
                    {
                        m_imageStream.md->cnt0++;
                    }

                    m_imageStream.md->write = 0;
                    ImageStreamIO_sempost( &m_imageStream, -1 );

                    ++idx;
                    if( idx >= m_shapes.planes() )
                        idx = 0;

                    if( !m_trigger )
                    {
                        modstart.tv_nsec += freqNsec;
                        if( modstart.tv_nsec >= 1000000000 )
                        {
                            modstart.tv_nsec -= 1000000000;
                            modstart.tv_sec += 1;
                        }
                        dnsec = freqNsec;
                    }
                }
            }
            if( m_restartSp )
                continue;

            recordSparkleClock( true );
            log<text_log>( "stopped modulating", logPrio::LOG_NOTICE );
            // Always zero when done
            clock_gettime( CLOCK_REALTIME, &currtime );
            m_imageStream.md->write = 1;

            memset( m_imageStream.array.raw, 0.0, m_width * m_height * m_typeSize );

            m_imageStream.md->atime = currtime;
            m_imageStream.md->writetime = currtime;

            if( !m_trigger )
                m_imageStream.md->cnt0++;

            m_imageStream.md->write = 0;
            ImageStreamIO_sempost( &m_imageStream, -1 );
            log<text_log>( "zeroed" );
        }
    }
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_trigger )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_trigger.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
        return 0;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_trigger = false;
        indi::updateSwitchIfChanged( m_indiP_trigger, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_trigger = true;
        indi::updateSwitchIfChanged( m_indiP_trigger, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
    }

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_delay )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_delay.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float del = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        del = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        del = ipRecv["target"].get<float>();
    }

    if( del == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No requested delay" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_triggerDelay = del;
    updateIfChanged( m_indiP_delay, "target", m_triggerDelay );

    m_restartSp = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_separation_1 )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_separation_1.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float sep = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        sep = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        sep = ipRecv["target"].get<float>();
    }

    if( sep == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No requested separation_1" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_separation_1 = sep;
    updateIfChanged( m_indiP_separation_1, "target", m_separation_1 );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_separation_2 )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_separation_2.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float sep = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        sep = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        sep = ipRecv["target"].get<float>();
    }

    if( sep == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No requested separation_2" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_separation_2 = sep;
    updateIfChanged( m_indiP_separation_2, "target", m_separation_2 );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_angle )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_angle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float ang = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        ang = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        ang = ipRecv["target"].get<float>();
    }

    if( ang == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "No angle received" } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_angle = ang;
    updateIfChanged( m_indiP_angle, "target", m_angle );

    m_restartSp = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_amp )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_amp.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float amp = -1000000000;

    if( ipRecv.find( "current" ) )
    {
        amp = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        amp = ipRecv["target"].get<float>();
    }

    if( amp == -1000000000 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested amp: " + std::to_string( amp ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_amp = amp;
    updateIfChanged( m_indiP_amp, "target", m_amp );

    m_restartSp = true;
    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_cross )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_cross.createUniqueKey() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
        return 0;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_cross = false;
        indi::updateSwitchIfChanged( m_indiP_cross, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_cross = true;
        indi::updateSwitchIfChanged( m_indiP_cross, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
    }

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_frequency )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_frequency.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float freq = -1;

    if( ipRecv.find( "current" ) )
    {
        freq = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        freq = ipRecv["target"].get<float>();
    }

    if( freq < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested frequency: " + std::to_string( freq ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_frequency = freq;
    updateIfChanged( m_indiP_frequency, "target", m_frequency );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_interval )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_interval.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float interval = -1;

    if( ipRecv.find( "current" ) )
    {
        interval = ipRecv["current"].get<float>();
    }

    if( ipRecv.find( "target" ) )
    {
        interval = ipRecv["target"].get<float>();
    }

    if( interval < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested interval: " + std::to_string( interval ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_sparkleClockInterval = interval;
    updateIfChanged( m_indiP_interval, "target", m_sparkleClockInterval );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_dwell )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_dwell.createUniqueKey() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    unsigned dwell = 0;

    if( ipRecv.find( "current" ) )
    {
        dwell = ipRecv["current"].get<unsigned>();
    }

    if( ipRecv.find( "target" ) )
    {
        dwell = ipRecv["target"].get<unsigned>();
    }

    if( dwell == 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested dwell: " + std::to_string( dwell ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_dwell = dwell;
    updateIfChanged( m_indiP_dwell, "target", m_dwell );

    m_restartSp = true;

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_single )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_single, ipRecv );

    int single = 0;

    if( ipRecv.find( "current" ) )
    {
        single = ipRecv["current"].get<int>();
    }

    if( ipRecv.find( "target" ) )
    {
        single = ipRecv["target"].get<int>();
    }

    if( single < -1 || single > 3 )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid requested dwell: " + std::to_string( single ) } );
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_single = single;
    updateIfChanged( m_indiP_single, "target", m_single );
    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_modulating )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_modulating.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
        return 0;

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_modulating = false;
        indi::updateSwitchIfChanged( m_indiP_modulating, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_IDLE );
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_modulating = true;
        indi::updateSwitchIfChanged( m_indiP_modulating, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( sparkleClock, m_indiP_zero )
( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_zero.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    if( m_modulating == true )
    {
        log<text_log>( "zero requested but currently modulating", logPrio::LOG_NOTICE );
        return 0;
    }

    if( !ipRecv.find( "request" ) )
        return 0;

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        m_imageStream.md->write = 1;

        memset( m_imageStream.array.raw, 0, m_width * m_height * m_typeSize );
        timespec currtime;
        clock_gettime( CLOCK_REALTIME, &currtime );
        m_imageStream.md->atime = currtime;
        m_imageStream.md->writetime = currtime;

        m_imageStream.md->cnt0++;

        m_imageStream.md->write = 0;
        ImageStreamIO_sempost( &m_imageStream, -1 );
        log<text_log>( "zeroed" );
    }

    return 0;
}

inline int sparkleClock::checkRecordTimes()
{
    return telemeter<sparkleClock>::checkRecordTimes( telem_sparkleclock() );
}

inline int sparkleClock::recordTelem( const telem_sparkleclock * )
{
    return recordSparkleClock( true );
}

inline int sparkleClock::recordSparkleClock( bool force )
{
    static bool lastModulating = m_modulating;
    static bool lastTrigger = m_trigger;
    static float lastFrequency = m_frequency;
    static float lastInterval = m_frequency;
    static float lastSeparation1 = m_separation_1;
    static float lastSeparation2 = m_separation_2;
    static float lastAngle = m_angle;
    static float lastAmp = m_amp;
    static bool lastCross = m_cross;

    if( !( lastModulating == m_modulating ) || !( lastTrigger == m_trigger ) || !( lastFrequency == m_frequency ) || !( lastInterval == m_interval ) ||
        !( lastSeparation1 == m_separation_1 ) ||!( lastSeparation2 == m_separation_2 ) || !( lastAngle == m_angle ) || !( lastAmp == m_amp ) ||
        !( lastCross == m_cross ) || force )
    {
        telem<telem_sparkleclock>({ m_modulating,
                                m_trigger,
                                m_frequency,
                                m_interval,
                                std::vector<float>( { m_separation_1, m_separation_2 } ),
                                m_angle,
                                m_amp });

        lastModulating = m_modulating;
        lastTrigger = m_trigger;
        lastFrequency = m_frequency;
        lastSeparation1 = m_separation_1;
        lastSeparation2 = m_separation_2;
        lastAngle = m_angle;
        lastAmp = m_amp;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // sparkleClock_hpp
