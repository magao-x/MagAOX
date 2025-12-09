/** \file hwpSequencer.hpp
 * \brief The MagAO-X HWP rotation tracker header file
 *
 * \ingroup hwpSequencer_files
 */

#ifndef hwpSequencer_hpp
#define hwpSequencer_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <ImageStreamIO/ImageStreamIO.h>
#include <ImageStreamIO/ImageStruct.h>

#include <mx/math/gslInterpolation.hpp>
#include <mx/ioutils/readColumns.hpp>

/** \defgroup hwpSequencer
 * \brief The MagAO-X application to track pupil rotation with the HWP.
 *
 * <a href="../handbook/operating/software/apps/hwpSequencer.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup hwpSequencer_files
 * \ingroup hwpSequencer
 */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ADC Tracker
/**
 * \ingroup hwpSequencer
 */
class hwpSequencer : public MagAOXApp<true>, public dev::shmimMonitor<hwpSequencer>
{

    // Give the test harness access.
    friend class hwpSequencer_test;

    friend class dev::shmimMonitor<hwpSequencer>;

    typedef dev::shmimMonitor<hwpSequencer> shmimMonitorT;

    protected:
        /** \name Configurable Parameters
         *@{
        */

        std::string m_hwpTracker{ "hwptrack" };

        std::string m_fxngenName{ "fxngensync" };

        std::string m_fxngenChannel{ "C2" };

        std::string m_obsAppName{ "observers" };

        double m_timePerPos{ 0 };

        int m_numCycles{ -1 };

        int m_curCycleNumber{ 0 };

        u_int m_hwpPosIndex{ 0 };

        bool m_sequencing{ false };

        bool m_lastCycle{ false };

        bool m_startSaving{ false };

        std::vector<float> m_hwpPositions{ {0.0, 45.0, 22.5, 67.5} };

        float m_reportedHwpPos{ 0 };
        ///<

        unsigned m_hwpWait {100};  ///< The time to pause between checks of the hwp state during open/shut [msec]. Default is 100.

        unsigned m_hwpTimeout {5000}; ///< Total time to wait for sensor to change state before timing out [msec]. Default is 2000.
        ///@}

        bool m_doMoveHwp {false}; ///< Flag telling the hwp thread that it should actually move the hwp, not just go back to sleep.

        bool m_fxngenOutp {false};

        sem_t m_smSemaphore{ 0 }; ///< Semaphore used to synchronize the hwp thread and the sm thread.


        bool m_sequencerThreadInit {true}; ///< Initialization flag for the open thread.

        pid_t m_sequencerThreadID {0}; ///< Open thread PID.

        pcf::IndiProperty m_sequencerThreadProp; ///< The property to hold the open thread details.

        std::thread m_sequencerThread; ///< The opening thread.

        /// Open thread starter function
        static void sequencerThreadStart( hwpSequencer * h /**< [in] pointer to this */);

        /// Open thread function
        /** Runs until m_shutdown is true.
             */
        int sequencerThreadExec();

        int doHwpAction();

        int startSequencing();

        int stopSequencing();

        int allocate( const dev::shmimT &);

        int processImage( void *, const dev::shmimT &);


    public:
        /// Default c'tor.
        hwpSequencer();

        /// D'tor, declared and defined for noexcept.
        ~hwpSequencer() noexcept
        {
        }

        virtual void setupConfig();

        /// Implementation of loadConfig logic, separated for testing.
        /** This is called by loadConfig().
         */
        int loadConfigImpl(mx::app::appConfigurator &_config); /**< [in] an application configuration from which to load values*/

        virtual void loadConfig();

        /// Startup function
        /**
         *
         */
        virtual int appStartup();

        /// Implementation of the FSM for hwpSequencer.
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

        /** @name INDI
         *
         * @{
         */
    protected:
        pcf::IndiProperty m_indiP_timePerPos;

        pcf::IndiProperty m_indiP_numCycles;

        pcf::IndiProperty m_indiP_curCycleNumber;

        pcf::IndiProperty m_indiP_hwpPosIndex;

        pcf::IndiProperty m_indiP_sequence;

        pcf::IndiProperty m_indiP_lastCycle;

        pcf::IndiProperty m_indiP_hwpTracker_target;

        pcf::IndiProperty m_indiP_hwpTracker_current;

        pcf::IndiProperty m_indiP_fxngenOutput;

        pcf::IndiProperty m_indiP_fxngenOutput_status;

        pcf::IndiProperty m_indiP_obsSaving;


    public:

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_timePerPos );

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_numCycles );

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_sequence );

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_lastCycle );

        INDI_SETCALLBACK_DECL( hwpSequencer, m_indiP_hwpTracker_current );

        INDI_SETCALLBACK_DECL( hwpSequencer, m_indiP_fxngenOutput_status );

};

hwpSequencer::hwpSequencer() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void hwpSequencer::setupConfig()
{
    config.add("hwp.devName",
               "",
               "hwp.devName",
               argType::Required,
               "hwp",
               "devName",
               false,
               "string",
               "HWP tracker app name, default is 'hwptracker'" );

    config.add("fxngen.devName",
               "",
               "fxngen.devName",
               argType::Required,
               "fxngen",
               "devName",
               false,
               "string",
               "Function generator device name, default is 'fxngensync'." );

    config.add("fxngen.channel",
               "",
               "fxngen.channel",
               argType::Required,
               "fxngen",
               "channel",
               false,
               "string",
               "Function generator output channel, default is 'C2'.");

    config.add("observers.devName",
               "",
               "observers.devName",
               argType::Required,
               "observers",
               "devName",
               false,
               "string",
               "Observers application name, default is 'observers'");

    SHMIMMONITOR_SETUP_CONFIG(config);
}

int hwpSequencer::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_hwpTracker, "hwp.devName" );
    _config( m_fxngenName, "fxngen.devName" );
    _config( m_fxngenChannel, "fxngen.channel" );
    _config( m_obsAppName, "observers.devName" );

    SHMIMMONITOR_LOAD_CONFIG(_config);

    return 0;
}

void hwpSequencer::loadConfig()
{
    loadConfigImpl( config );
}

int hwpSequencer::appStartup()
{

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_sequence, "sequence" );

    CREATE_REG_INDI_NEW_TOGGLESWITCH( m_indiP_lastCycle, "lastCycle" );

    CREATE_REG_INDI_NEW_NUMBERD( m_indiP_timePerPos, "timePerPos", 1, 600, 0.1, "%0.1f", "Time per position", "Sequencer");
    m_indiP_timePerPos["current"].set( m_timePerPos );

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_numCycles, "numCycles", -1, 10800, 1, "%d", "Number of cycles", "Sequencer");
    m_indiP_numCycles["current"].set( m_numCycles );


    REG_INDI_NEWPROP_NOCB( m_indiP_hwpPosIndex, "hwpPosIndex", pcf::IndiProperty::Number );
    m_indiP_hwpPosIndex.add( pcf::IndiElement( "value" ) );
    m_indiP_hwpPosIndex["value"].set( m_hwpPosIndex );

    REG_INDI_NEWPROP_NOCB( m_indiP_curCycleNumber, "curCycle", pcf::IndiProperty::Number );
    m_indiP_curCycleNumber.add( pcf::IndiElement( "value" ) );
    m_indiP_curCycleNumber["value"].set( m_curCycleNumber );

    m_indiP_hwpTracker_target = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_hwpTracker_target.setDevice( m_hwpTracker );
    m_indiP_hwpTracker_target.setName( "hwp_position" );
    m_indiP_hwpTracker_target.add( pcf::IndiElement( "target" ) );

    REG_INDI_SETPROP( m_indiP_hwpTracker_current, m_hwpTracker, "hwp_position");

    m_indiP_fxngenOutput = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_fxngenOutput.setDevice( m_fxngenName );
    m_indiP_fxngenOutput.setName( m_fxngenChannel + "outp" );
    m_indiP_fxngenOutput.add( pcf::IndiElement( "value" ) );

    REG_INDI_SETPROP(m_indiP_fxngenOutput_status, m_fxngenName, m_fxngenChannel + "outp");

    m_indiP_obsSaving = pcf::IndiProperty( pcf::IndiProperty::Switch );
    m_indiP_obsSaving.setDevice( m_obsAppName );
    m_indiP_obsSaving.setName( "obs_on" );
    m_indiP_obsSaving.add( pcf::IndiElement( "toggle" ) );


    SHMIMMONITOR_APP_STARTUP;

    if( sem_init( &m_smSemaphore, 0, 0 ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "Initializing S.M. semaphore" } );
        return -1;
    }

    if(threadStart( m_sequencerThread, m_sequencerThreadInit, m_sequencerThreadID, m_sequencerThreadProp, 0, "", "sequencerThread", this, sequencerThreadStart) < 0)
    {
        log<software_error>({__FILE__, __LINE__});
        return -1;
    }

    state( stateCodes::READY );

    return 0;
}

int hwpSequencer::appLogic()
{

    SHMIMMONITOR_APP_LOGIC;
    SHMIMMONITOR_UPDATE_INDI;

    if (!m_doMoveHwp)
    {
        m_hwpPosIndex += 1;
        if (m_hwpPosIndex == m_hwpPositions.size())
        {
            // We've reached the end of the cycle
            m_hwpPosIndex = 0;
            m_curCycleNumber += 1;

            std::cerr << "Current cycle: " << m_curCycleNumber;
            log<text_log>( "Current cycle: " + std::to_string(m_curCycleNumber) );
            if (m_numCycles > 0)
            {
                std::cerr << " / " << m_numCycles;
            }
            std::cerr << std::endl;

            if (m_lastCycle || (m_numCycles > 0 && m_curCycleNumber >= m_numCycles)) {
                stopSequencing();
                return 0;
            }
        }
        m_doMoveHwp = true;
    }

    return 0;
}

int hwpSequencer::appShutdown()
{
    SHMIMMONITOR_APP_SHUTDOWN;

    if (m_sequencing)
        stopSequencing();

    if (m_sequencerThread.joinable())
    {
        pthread_kill(m_sequencerThread.native_handle(), SIGUSR1);
    }

    if(m_sequencerThread.joinable())
    {
        try
        {
            m_sequencerThread.join(); //this will throw if it was already joined
        }
        catch(...) {}
    }

    // Make sure in case of borked state that we turn the fxngen output back on
    m_indiP_fxngenOutput["value"] = "On";
    sendNewProperty(m_indiP_fxngenOutput);

    return 0;
}


void hwpSequencer::sequencerThreadStart( hwpSequencer * h )
{
   h->sequencerThreadExec();
}


int hwpSequencer::sequencerThreadExec( )
{
    // thread prep
    m_sequencerThreadID = syscall(SYS_gettid);

    while( m_sequencerThreadInit == true && shutdown() == 0)
        sleep(1);

    // busy loop
    while(shutdown() == 0)
    {
        if(m_sequencing && m_doMoveHwp)
        {
            if(doHwpAction() < 0)
            {
                log<software_error>({__FILE__,__LINE__});
                return -1;
            }

            mx::sys::sleep(m_timePerPos);

            m_doMoveHwp = false;

            XWC_SEM_FLUSH(m_smSemaphore);
        }

        mx::sys::sleep(0.1);
    }

    return 0;
}

int hwpSequencer::doHwpAction()
{
    // Stop triggering
    m_indiP_fxngenOutput["value"] = "Off";
    sendNewProperty(m_indiP_fxngenOutput);

    // Wait for trigger to stop
    while (m_fxngenOutp)
    {
        mx::sys::milliSleep(m_hwpWait);
    }

    // Wait for current frame to arrive
    if (sem_wait(&m_smSemaphore))
    {
        log<software_error>({ __FILE__, __LINE__, "failed waiting for semaphore" });
        return -1;
    }

    // move HWP
    float target_hwp_angle = m_hwpPositions[m_hwpPosIndex];

    std::cerr << "Current HWP index: " << m_hwpPosIndex + 1 << " / " << m_hwpPositions.size() << std::endl;
    std::cerr << "Moving HWP to " << target_hwp_angle << std::endl;

    m_indiP_hwpTracker_target["target"] = target_hwp_angle;
    sendNewProperty(m_indiP_hwpTracker_target);

    // check if the HWP is in position (note, depends on the callback for m_reportedHwpPos)
    float angle_tol = 0.5;
    double t0 = mx::sys::get_curr_time();
    while (fabs(m_reportedHwpPos - target_hwp_angle) > angle_tol)
    {
        mx::sys::milliSleep(m_hwpWait);
        if ((mx::sys::get_curr_time() - t0) * 1000 > m_hwpTimeout)
        {
            log<software_error>({ __FILE__, __LINE__, "timed out waiting for HWP to move to position" });
            return -1;
        }
    }

    // update index in here so it comes AFTER the hwp has moved
    updateIfChanged( m_indiP_hwpPosIndex, "value", m_hwpPosIndex );
    updateIfChanged( m_indiP_curCycleNumber, "value", m_curCycleNumber );


    // this only triggers when calling startSequencing--we wan't to turn
    // the observer obs_on toggle on once and let the external trigger
    // dictate the intermediate stops and starts
    if (m_startSaving)
    {
        m_indiP_obsSaving["toggle"] = pcf::IndiElement::On;
        sendNewProperty(m_indiP_obsSaving);

        m_startSaving = false;
    }

    m_indiP_fxngenOutput["value"] = "On";
    sendNewProperty(m_indiP_fxngenOutput);

    return 0;
}


int hwpSequencer::startSequencing()
{
    if (m_sequencing) return 0;

    if (m_timePerPos == 0)
    {
        log<text_log>( "Cannot sequence with " + std::to_string(m_timePerPos) + " time per HWP position" );
        return 0;
    }

    // Move HWP to first position
    m_hwpPosIndex = 0;
    updateIfChanged( m_indiP_hwpPosIndex, "value", m_hwpPosIndex );

    m_curCycleNumber = 1;
    updateIfChanged( m_indiP_curCycleNumber, "value", m_curCycleNumber );

    state( stateCodes::OPERATING );

    std::cerr << "Starting sequence" << std::endl;
    log<text_log>( "Starting sequence" );

    m_sequencing = true;
    updateSwitchIfChanged( m_indiP_sequence, "toggle", pcf::IndiElement::On, INDI_IDLE);

    m_startSaving = true;

    m_doMoveHwp = true;

    XWC_SEM_FLUSH(m_smSemaphore);

    return 0;
}

int hwpSequencer::stopSequencing()
{
    if (!m_sequencing) return 0;

    std::cerr << "Stopping sequence" << std::endl;
    log<text_log>( "Stopping sequence" );

    if (sem_wait(&m_smSemaphore))
    {
        log<software_error>({ __FILE__, __LINE__, "failed waiting for semaphore" });
        return -1;
    }

    // Turn the data acquisition off, but let's make sure the camera trigger is still rolling
    m_indiP_obsSaving["toggle"] = pcf::IndiElement::Off;
    sendNewProperty(m_indiP_obsSaving);
    m_startSaving = false;

    m_indiP_fxngenOutput["value"] = "On";
    sendNewProperty(m_indiP_fxngenOutput);

    m_doMoveHwp = false;

    m_sequencing = false;
    updateSwitchIfChanged( m_indiP_sequence, "toggle", pcf::IndiElement::Off, INDI_IDLE);

    m_lastCycle = false;
    updateSwitchIfChanged( m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE);

    state( stateCodes::READY );

    return 0;
}

/* shmimMonitor interface impl */

int hwpSequencer::allocate(const dev::shmimT &)
{
    // nothing to allocate
    return 0;
}

int hwpSequencer::processImage( void*, const dev::shmimT &)
{
    // Now tell the f.g. to get going
    if( sem_post( &m_smSemaphore ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
        return -1;
    }
    return 0;
}


INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_sequence )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sequence, ipRecv );

    if( ipRecv.getName() != m_indiP_sequence.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) ) return 0;

    if (ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
        startSequencing();
    else
        stopSequencing();

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_numCycles )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_numCycles, ipRecv );

    if( ipRecv.getName() != m_indiP_numCycles.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "target" ) )
        return 0;

    if (m_sequencing)
    {
        std::cerr << "Cannot change sequence parameters while sequencing" << std::endl;
        return 0;
    }

    int numCycles = ipRecv["target"].get<int>();

    if (numCycles == 0)
    {
        log<software_error>( { __FILE__, __LINE__, "numCycles cannot be 0" } );
        return -1;
    }

    m_numCycles = numCycles;
    updatesIfChanged<int>( m_indiP_numCycles, { "current", "target" }, { m_numCycles, m_numCycles } );

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_timePerPos )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_timePerPos, ipRecv );

    if( ipRecv.getName() != m_indiP_timePerPos.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "target" ) )
        return 0;

    if (m_sequencing)
    {
        std::cerr << "Cannot change sequence parameters while sequencing" << std::endl;
        return 0;
    }

    double timePerPos = ipRecv["target"].get<double>();

    m_timePerPos = timePerPos;
    updatesIfChanged<double>( m_indiP_timePerPos, { "current", "target" }, { m_timePerPos, m_timePerPos } );

    return 0;
}

INDI_NEWCALLBACK_DEFN( hwpSequencer, m_indiP_lastCycle )( const pcf::IndiProperty &ipRecv )
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_lastCycle, ipRecv );

    if( ipRecv.getName() != m_indiP_lastCycle.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) ) return 0;

    if (!m_sequencing)
    {
        std::cerr << "Cannot stop sequencing while not sequencing" << std::endl;
        return 0;
    }


    if (ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        m_lastCycle = true;

        int remaining_positions = m_hwpPositions.size() - m_hwpPosIndex - 1;

        updateSwitchIfChanged(m_indiP_lastCycle, "toggle", pcf::IndiElement::On, INDI_IDLE);

        log<text_log>( "Last HWP cycle toggled on" );

        std::cerr << "Last HWP cycle toggled. Will finish in " << std::to_string(remaining_positions) << " HWP positions (approx. " << std::to_string(remaining_positions * m_timePerPos) << " s)" << std::endl;
    }
    else
    {
        m_lastCycle = false;

        updateSwitchIfChanged(m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE);

        log<text_log>( "Last HWP cycle toggled off" );
    }

    return 0;
}


INDI_SETCALLBACK_DEFN(hwpSequencer, m_indiP_hwpTracker_current)(const pcf::IndiProperty &ipRecv)
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_hwpTracker_current, ipRecv );

   if(ipRecv.getName() != m_indiP_hwpTracker_current.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});

      return -1;
   }

   if(!ipRecv.find("current")) return 0;

   m_reportedHwpPos = ipRecv["current"].get<float>();

   return 0;
}



INDI_SETCALLBACK_DEFN(hwpSequencer, m_indiP_fxngenOutput_status)(const pcf::IndiProperty &ipRecv)
{

    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fxngenOutput_status, ipRecv );

   if(ipRecv.getName() != m_indiP_fxngenOutput_status.getName())
   {
      log<software_error>({__FILE__,__LINE__,"wrong INDI property received"});

      return -1;
   }

   if(!ipRecv.find("value")) return 0;

   std::string fxngenOutp = ipRecv["value"].get<std::string>();
   if (fxngenOutp == "On")
   {
      m_fxngenOutp = true;
   }
   else if (fxngenOutp == "Off")
   {
      m_fxngenOutp = false;
   }

   return 0;
}


} // namespace app
} // namespace MagAOX

#endif // hwpSequencer_hpp
