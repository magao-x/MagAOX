/** \file hwpSequencer.hpp
 * \brief The MagAO-X HWP rotation tracker header file
 *
 * \ingroup hwpSequencer_files
 */

#ifndef hwpSequencer_hpp
#define hwpSequencer_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

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
class hwpSequencer : public MagAOXApp<true>, public dev::telemeter<hwpSequencer>
{

    // Give the test harness access.
    friend class hwpSequencer_test;

    protected:
        /** \name Configurable Parameters
         *@{
        */

        std::string m_hwpTracker{ "hwptrack" };

        std::string m_observers{ "observers" };

        double m_timePerPos{ 0 };

        int m_numCycles{ -1 };

        int m_curCycleNumber{ 0 };

        u_int m_hwpPosIndex{ 0 };

        bool m_sequencing{ false };

        bool m_lastCycle{ false };

        std::vector<float> m_hwpPositions{ {0.0, 45.0, 22.5, 67.5} };

        float m_updateInterval{ 1 };


        ///<

        unsigned m_hwpWait {100};  ///< The time to pause between checks of the hwp state during open/shut [msec]. Default is 100.

        unsigned m_hwpTimeout {5000}; ///< Total time to wait for sensor to change state before timing out [msec]. Default is 2000.
        ///@}

        bool m_doMoveHwp {false}; ///< Flag telling the open thread that it should actually open the shutter, not just go back to sleep.

        bool m_moveHwpThreadInit {true}; ///< Initialization flag for the open thread.

        pid_t m_moveHwpThreadID {0}; ///< Open thread PID.

        pcf::IndiProperty m_moveHwpThreadProp; ///< The property to hold the open thread details.

        std::thread m_moveHwpThread; ///< The opening thread.

        /// Open thread starter function
        static void moveHwpThreadStart( hwpSequencer * h /**< [in] pointer to this */);

        /// Open thread function
        /** Runs until m_shutdown is true.
             */
        void moveHwpThreadExec();



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
        int loadConfigImpl(
            mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

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

        virtual int

        virtual int moveHwp();

        virtual int startSequencing();

        virtual int stopSequencing();

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

        pcf::IndiProperty m_indiP_observers;


    public:

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_timePerPos );

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_numCycles );

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_sequence );

        INDI_NEWCALLBACK_DECL( hwpSequencer, m_indiP_lastCycle );

};

hwpSequencer::hwpSequencer() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void hwpSequencer::setupConfig()
{
    config.add( "sequencer.devName",
                "",
                "sequencer.devName",
                argType::Required,
                "sequencer",
                "devName",
                false,
                "string",
                "HWP tracker app name" );

    config.add( "sequencer.obsName",
                "",
                "sequencer.obsName",
                argType::Required,
                "sequencer",
                "obsName",
                false,
                "string",
                "Observers app name." );

    config.add( "sequencer.updateInterval",
                "",
                "sequencer.updateInterval",
                argType::Required,
                "sequencer",
                "updateInterval",
                false,
                "float",
                "Update interval, default is 1 second." );

}

int hwpSequencer::loadConfigImpl( mx::app::appConfigurator &config )
{
    config( m_hwpTracker, "sequencer.devName" );
    config( m_observers, "sequencer.obsName" );
    config( m_updateInterval, "sequencer.updateInterval" );

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

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_numCycles, "numCycles", -1, 10800, 1, "%d", "Number of cycles", "Sequencer");


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

    m_indiP_observers = pcf::IndiProperty( pcf::IndiProperty::Switch );
    m_indiP_observers.setDevice( m_observers );
    m_indiP_observers.setName( "obs_on" );
    m_indiP_observers.add( pcf::IndiElement( "toggle" ) );



    if(threadStart( m_moveHwpThread, m_moveHwpThreadInit, m_moveHwpThreadID, m_moveHwpThreadProp, 0, "", "moveHwp", this, moveHwpThreadStart) < 0)
    {
        log<software_error>({__FILE__, __LINE__});
        return -1;
    }



    state( stateCodes::READY );

    return 0;
}

int hwpSequencer::appLogic()
{
    static double lastupdate = 0;
    static double lastchange = 0;

    if (m_sequencing && mx::sys::get_curr_time() - lastupdate > m_updateInterval )
    {
        if (mx::sys::get_curr_time() - lastchange > m_timePerPos)
        {

            m_hwpPosIndex += 1;
            if (m_hwpPosIndex == m_hwpPositions.size())
            {
                // We've reached the end of the cycle
                m_hwpPosIndex = 0;
                m_curCycleNumber += 1;
                updateIfChanged( m_indiP_curCycleNumber, "value", m_curCycleNumber );

                if (m_lastCycle || (m_numCycles > 0 && m_curCycleNumber >= m_numCycles)) {
                    stopSequencing();
                    return 0;
                }
            }
            updateIfChanged( m_indiP_hwpPosIndex, "value", m_hwpPosIndex );

            std::cerr << "Number of completed cycles: " << m_curCycleNumber << " / " << m_numCycles << std::endl;
            // Pause saving and move HWP
            m_indiP_observers["toggle"] = pcf::IndiElement::Off;
            sendNewProperty(m_indiP_observers);

            moveHwp();

            m_indiP_observers["toggle"] = pcf::IndiElement::On;
            sendNewProperty(m_indiP_observers);

            lastchange = mx::sys::get_curr_time();
        }
        lastupdate = mx::sys::get_curr_time();
    }
    else if (!m_sequencing)
    {
        lastupdate = 0;
        lastchange = mx::sys::get_curr_time();
    }

    return 0;
}

int hwpSequencer::appShutdown()
{
    if (m_sequencing) stopSequencing();

    if (m_moveHwpThread.joinable())
    {
        pthread_kill(m_moveHwpThread.native_handle(), SIGUSR1)
    }

    if(m_moveHwpThread.joinable())
    {
        try m_moveHwpThread.join(); //this will throw if it was already joined
        catch(...)
    }
    return 0;
}

int hwpSequencer::moveHwp()
{
    float target_hwp_angle = m_hwpPositions[m_hwpPosIndex];

    std::cerr << "Current HWP index: " << m_hwpPosIndex + 1 << " / " << m_hwpPositions.size() << std::endl;
    std::cerr << "Moving HWP to " << target_hwp_angle << std::endl;

    m_indiP_hwpTracker_target["target"] = target_hwp_angle;
    sendNewProperty(m_indiP_hwpTracker_target);

    // block until HWP in position
    double t0 = mx::sys::get_curr_time();
    float angle_tol = 0.05;
    while (fabs(m_indiP_hwpTracker_target["current"].get<float>() - target_hwp_angle) > angle_tol)
    {
        std::cerr << "Current hwpTracker posn " << m_indiP_hwpTracker_target["current"].get<float>();
        mx::sys::milliSleep(m_hwpWait);
        if ((mx::sys::get_curr_time() - t0) * 1000 > m_hwpTimeout)) break;
    }

    return 0;
}


void hwpSequencer::moveHwpThreadStart( hwpSequencer * h )
{
   h->moveHwpThreadExec();
}

void hwpSequencer::moveHwpThreadExec( )
{
   m_moveHwpThreadID = syscall(SYS_gettid);

   while( m_moveHwpThreadInit == true && this.shutdown() == 0)
   {
      sleep(1);
   }

   while(this.shutdown() == 0)
   {
      if( m_doMoveHwp )
      {
         if(moveHwp() < 0)
         {
            log<software_error>({__FILE__,__LINE__});
         }
         m_doMoveHwp = false;
      }

      sleep(1);

   }

   return;
}

int hwpSequencer::startSequencing()
{
    if (m_sequencing) return 0;
    // Move HWP to first position
    m_hwpPosIndex = 0;
    updateIfChanged( m_indiP_hwpPosIndex, "value", m_hwpPosIndex );
    m_curCycleNumber = 0;
    updateIfChanged( m_indiP_curCycleNumber, "value", m_curCycleNumber );

    state( stateCodes::OPERATING );

    moveHwp();

    std::cerr << "Starting sequence" << std::endl;

    m_sequencing = true;
    updateSwitchIfChanged( m_indiP_sequence, "toggle", pcf::IndiElement::On, INDI_IDLE);

    m_indiP_observers["toggle"] = pcf::IndiElement::On;
    sendNewProperty(m_indiP_observers);

    return 0;
}

int hwpSequencer::stopSequencing()
{
    if (!m_sequencing) return 0;

    std::cerr << "Stopping sequence" << std::endl;

    m_indiP_observers["toggle"] = pcf::IndiElement::Off;
    sendNewProperty(m_indiP_observers);

    m_sequencing = false;
    updateSwitchIfChanged( m_indiP_sequence, "toggle", pcf::IndiElement::Off, INDI_IDLE);

    m_lastCycle = false;
    updateSwitchIfChanged( m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE);

    state( stateCodes::READY );

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

    if( !ipRecv.find( "toggle" ) )
        return 0;

    if (ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On) {
        startSequencing();
    }
    else
    {
        stopSequencing();
    }

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

    if (timePerPos < 1)
    {
        log<software_error>( { __FILE__, __LINE__, "timePerPos cannot be less than 1" } );
        return -1;
    }

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

    if( !ipRecv.find( "toggle" ) )
        return 0;

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

        log<text_log>( "Last HWP cycle toggled on");

        std::cerr << "Last HWP cycle toggled. Will finish in " << std::to_string(remaining_positions) << " HWP positions (approx. " << std::to_string(remaining_positions * m_timePerPos) << " s)" << std::endl;
    }
    else
    {
        m_lastCycle = false;

        updateSwitchIfChanged(m_indiP_lastCycle, "toggle", pcf::IndiElement::Off, INDI_IDLE);

        log<text_log>( "Last HWP cycle toggled off");
    }

    return 0;
}


} // namespace app
} // namespace MagAOX

#endif // hwpSequencer_hpp
