/** \file cacaoInterface.hpp
  * \brief The MagAO-X CACAO Interface header file
  *
  * \ingroup cacaoInterface_files
  */

#ifndef cacaoInterface_hpp
#define cacaoInterface_hpp

#include <atomic>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup cacaoInterface
  * \brief The CACAO Interface to provide loop status
  *
  * <a href="../handbook/apps/cacaoInterface.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup cacaoInterface_files
  * \ingroup cacaoInterface
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X CACAO Interface
/**
  * \ingroup cacaoInterface
  */
class cacaoInterface : public MagAOXApp<true>, public dev::telemeter<cacaoInterface>
{

   //Give the test harness access.
   friend class cacaoInterface_test;

   /// Base providing loop telemetry.
   typedef dev::telemeter<cacaoInterface> telemeterT;

   friend class dev::telemeter<cacaoInterface>;

protected:

   /** \name Configurable Parameters
     *@{
     */
   std::string m_loopNumber; ///< The loop number, X in aolX.  We keep it a string because that's how it gets used.

   ///@}

   std::string m_aoCalDir;         ///< Directory of the active loop calibration.
   std::string m_aoCalArchiveTime; ///< Archive timestamp of the active calibration.
   std::string m_aoCalLoadTime;    ///< Load timestamp of the active calibration.

   std::string m_loopName; ///< the loop name

   std::string m_fpsName; ///< Name of the loop function parameter structure.
   std::string m_fpsFifo; ///< Command FIFO for the loop's FPS controller.

   std::atomic<int> m_loopState{ 0 };          ///< Latest loop state. 0 is open, 1 is paused, 2 is closed.
   unsigned         m_loopCommand{ 0 };        ///< Revision of the last command written to the loop on/off flag.
   bool m_offloadLinked{ false }; ///< Whether HO commands and manual opens coordinate offloading.

   bool m_loopProcesses {false}; ///< Status of the loop processes.
   bool m_loopProcesses_stat {false}; ///< What the cacao status file says the state of loop processes is.

   float m_gain {0.0}; ///< The current loop gain.
   float m_gain_target {0.0}; ///< The target loop gain.

   float m_multCoeff {0.0}; ///< The current multiplicative coefficient (1-leak)
   float m_multCoeff_target {0.0}; ///< The target multiplicative coefficient (1-leak)

   std::vector<int> m_modeBlockStart; ///< First mode index of each calibration block.
   std::vector<int> m_modeBlockN;     ///< Mode count of each calibration block.

   std::vector<float> m_modeBlockGains; ///< Block gain values.
   std::vector<float> m_modeBlockMCs;   ///< Block multiplier values.
   std::vector<float> m_modeBlockLims;  ///< Block limit values.

   std::mutex m_modeBlockMutex; ///< Mutex for block data.

   float m_maxLim {0.0}; ///< The current max limit
   float m_maxLim_target {0.0}; ///< The target max limit

public:
   /// Default c'tor.
   cacaoInterface();

   /// D'tor, declared and defined for noexcept.
   ~cacaoInterface() noexcept
   {}

   /// Register the loop configuration options.
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   /// Load configuration for this loop.
   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for cacaoInterface.
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


   /** \name CACAO Interface Functions
     * @{
     */

   /// Write a command to the CACAO FPS controller.
   int writeFPSCommand( const std::string &command /**< [in] Complete line to send to the controller. */,
                        bool               waitForReader /**< [in] Whether to wait for the FIFO reader. */ );

   /// Set a string-valued FPS parameter.
   int setFPSVal( const std::string &fps /**< [in] FPS prefix. */,
                  const std::string &param /**< [in] FPS parameter. */,
                  const std::string &val /**< [in] Value to set. */,
                  bool               waitForReader = true /**< [in] Whether to wait for the FIFO reader. */ );

   template <typename T>
   /// Set an FPS parameter to the string representation of a numeric value.
   int setFPSVal( const std::string &fps /**< [in] FPS prefix. */,
                  const std::string &param /**< [in] FPS parameter. */,
                  const T           &val /**< [in] Value to set. */ );

   /// Read a string-valued FPS parameter.
   std::string getFPSValStr( const std::string &fps /**< [in] FPS prefix. */,
                             const std::string &param /**< [in] FPS parameter. */ );

   /// Read the numeric fields of an FPS parameter.
   std::string getFPSValNum( const std::string &fps /**< [in] FPS prefix. */,
                             const std::string &param /**< [in] FPS parameter. */ );

   /// Get the calibration details
   /** This is done each loop
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
   int getAOCalib();

   /// Determine mode blocks from the active calibration.
   int getModeBlocks();

   /// Check if the loop processes are running
   /** sets m_loopProcesses to true or false depending on what it finds out.
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
   int checkLoopProcesses();

   /// Set loop gain to the value of m_gain_target;
   /**
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setGain();

   /// Set loop multiplication coefficient to the value of m_multCoeff_target;
   /**
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setMultCoeff();

   /// Set loop max lim to the value of m_maxLim_target;
   /**
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setMaxLim();

   /// Turn the loop on
   /**
     * \returns 0 on success
     * \returns -1 on an error
     */
   int loopOn();

   /// Turn the loop off
   /**
     * \returns 0 on success
     * \returns -1 on an error
     */
   int loopOff();

   /// Request that CACAO zero the loop control channel.
   /**
    * \returns 0 on success
    * \returns -1 on an error
    */
   int loopZero();

   /// Send loop and offloader state commands in the required order.
   int requestLoopState( bool closed /**< [in] Desired loop state. */ );

   /// Handle a fresh settled loop state, including an external open.
   void observeLoopState( bool closed /**< [in] State read from CACAO. */ );

   /// Send a switch command to the tweeter-to-woofer offloader with the link marker on state commands.
   void sendOffloaderSwitch( const char                       *name /**< [in] Offloader property name. */,
                             const char                       *element /**< [in] Offloader switch element. */,
                             pcf::IndiElement::SwitchStateType requested /**< [in] Requested switch state. */ );

   /// Set and publish the offload link switch under the INDI mutex.
   void setOffloadLinked( bool linked /**< [in] Whether HO commands should control offloading. */ );

   /// @}

   /** \name File Monitoring Thread
     * Handling of offloads from the average woofer shape
     * @{
     */
   int m_fmThreadPrio {0}; ///< Priority of the filemonitoring thread.

   std::thread m_fmThread; ///< The file monitoring thread.

   bool m_fmThreadInit {true}; ///< Initialization flag for the file monitoring thread.

   pid_t m_fmThreadID {0}; ///< File monitor thread PID.

   pcf::IndiProperty m_fmThreadProp; ///< The property to hold the f.m. thread details.

   /// File monitoring thread starter function
   static void fmThreadStart( cacaoInterface * c /**< [in] pointer to this */);


   /// File monitoring thread function
   /** Runs until m_shutdown is true.
     */
   void fmThreadExec();


   ///@}

   pcf::IndiProperty m_indiP_loop;          ///< Name and number of the configured loop.
   pcf::IndiProperty m_indiP_loopProcesses; ///< Whether the loop processes are running.

   pcf::IndiProperty m_indiP_loopState;       ///< Loop closed/open state and command.
   pcf::IndiProperty m_indiP_loopZero;        ///< Loop zero request and dispatch status.
   pcf::IndiProperty m_indiP_loopGain;        ///< Current and requested loop gain.
   pcf::IndiProperty m_indiP_multCoeff;       ///< Current and requested loop multiplier.
   pcf::IndiProperty m_indiP_maxLim;          ///< Current and requested loop correction limit.
   pcf::IndiProperty m_indiP_offloadLink;     ///< HO-only switch selecting linked operation.
   pcf::IndiProperty m_indiP_offloadOverride; ///< Notification that the offloader was operated separately.

   /// Handle a loop state command.
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopState);

   /// Handle a loop zero request.
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopZero);

   /// Set whether offloading is linked to the HO loop.
   INDI_NEWCALLBACK_DECL( cacaoInterface, m_indiP_offloadLink );

   /// Release the link after the offloader reports a separate state change.
   INDI_NEWCALLBACK_DECL( cacaoInterface, m_indiP_offloadOverride );

   /// Handle a loop gain command.
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopGain);

   /// Handle a loop multiplier command.
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_multCoeff);

   /// Handle a loop limit command.
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_maxLim);

   /** \name Telemeter Interface
     *
     * @{
     */
   /// Check when loop telemetry is due.
   int checkRecordTimes();

   /// Record loop gain telemetry.
   int recordTelem( const telem_loopgain * /**< [in] Tag selecting loop gain telemetry. */ );

   /// Record loop state and gain settings when changed or forced.
   int recordLoopGain( bool force = false /**< [in] Whether to record unchanged values. */ );

   ///@}


};

cacaoInterface::cacaoInterface() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{

   return;
}

void cacaoInterface::setupConfig()
{
   config.add("loop.number", "", "loop.number", argType::Required, "loop", "number", false, "string", "the loop number");

   telemeterT::setupConfig(config);
}

int cacaoInterface::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_loopNumber, "loop.number");

   if(telemeterT::loadConfig(_config) < 0)
   {
      log<text_log>("Error during telemeter config", logPrio::LOG_CRITICAL);
      m_shutdown = true;
   }

   return 0;
}

void cacaoInterface::loadConfig()
{
   loadConfigImpl(config);
}

int cacaoInterface::appStartup()
{

   if(m_loopNumber == "")
   {
      return log<software_critical, -1>({__FILE__, __LINE__, "loop number not set"});
   }

   createROIndiText( m_indiP_loop, "loop", "name", "Loop Description", "Loop Controls", "Name");
   indi::addTextElement(m_indiP_loop, "number", "Number");
   m_indiP_loop["number"] = m_loopNumber;
   m_indiP_loop["name"] = m_loopName;
   registerIndiPropertyReadOnly(m_indiP_loop);

   createStandardIndiToggleSw( m_indiP_loopProcesses, "loop_processes", "Loop Processes", "Loop Controls");
   registerIndiPropertyReadOnly( m_indiP_loopProcesses);

   createStandardIndiToggleSw( m_indiP_loopState, "loop_state", "Loop State", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopState, INDI_NEWCALLBACK(m_indiP_loopState) );

   createStandardIndiRequestSw( m_indiP_loopZero, "loop_zero", "Loop Zero", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopZero,INDI_NEWCALLBACK(m_indiP_loopZero) );

   if( configName() == "holoop" )
   {
       createStandardIndiToggleSw( m_indiP_offloadLink, "offload_link", "Link Offload", "Loop Controls" );
       createStandardIndiRequestSw( m_indiP_offloadOverride, "offload_override" );
       m_indiP_offloadOverride.setPerm( pcf::IndiProperty::WriteOnly );

       if( registerIndiPropertyNew( m_indiP_offloadLink, INDI_NEWCALLBACK( m_indiP_offloadLink ) ) < 0 ||
           registerIndiPropertyNew( m_indiP_offloadOverride, INDI_NEWCALLBACK( m_indiP_offloadOverride ) ) < 0 )
       {
           return -1;
       }
   }

   createStandardIndiNumber<float>( m_indiP_loopGain, "loop_gain", 0.0, 10.0, 0.01, "%0.3f", "Loop Gain", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopGain, INDI_NEWCALLBACK(m_indiP_loopGain) );

   createStandardIndiNumber<float>( m_indiP_multCoeff, "loop_multcoeff", 0.0, 1.0, 0.001, "%0.3f", "Mult. Coefficient", "Loop Controls");
   registerIndiPropertyNew( m_indiP_multCoeff, INDI_NEWCALLBACK(m_indiP_multCoeff) );

   createStandardIndiNumber<float>( m_indiP_maxLim, "loop_max_limit", 0.0, 10.0, 0.001, "%0.3f", "Max. Limit", "Loop Controls");
   registerIndiPropertyNew( m_indiP_maxLim, INDI_NEWCALLBACK(m_indiP_maxLim) );

   if(threadStart( m_fmThread, m_fmThreadInit, m_fmThreadID, m_fmThreadProp, m_fmThreadPrio, "", "loopmon", this, fmThreadStart) < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }

   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   return 0;
}

int cacaoInterface::appLogic()
{
   //do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_fmThread.native_handle(),0) == 0)
   {
      log<software_critical>({__FILE__, __LINE__, "cacao file monitoring thread has exited"});

      return -1;
   }

   //These could change if a new calibration is loaded
   if(getAOCalib() < 0 )
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get AO calib", logPrio::LOG_ERROR);
      return 0;
   }

   if(m_loopProcesses == 0 || m_loopState == 0) state(stateCodes::READY);
   else state(stateCodes::OPERATING);

   if(telemeterT::appLogic() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return 0;
   }

   std::unique_lock<std::mutex> lock(m_indiMutex);

   updateIfChanged(m_indiP_loop, std::vector<std::string>({"name", "number"}), std::vector<std::string>({m_loopName, m_loopNumber}));

   if(m_loopProcesses)
   {
      updateSwitchIfChanged(m_indiP_loopProcesses, "toggle", pcf::IndiElement::On, INDI_OK);
   }
   else
   {
      updateSwitchIfChanged(m_indiP_loopProcesses, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   }

   if( m_loopState == 0 )
   {
      updateSwitchIfChanged(m_indiP_loopState, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   }
   else if(m_loopState == 1)
   {
      updateSwitchIfChanged(m_indiP_loopState, "toggle", pcf::IndiElement::On, INDI_IDLE);
   }
   else if(m_loopState == 2)
   {
      updateSwitchIfChanged(m_indiP_loopState, "toggle", pcf::IndiElement::On, INDI_OK);
   }

   updateIfChanged(m_indiP_loop, "name", m_loopName);
   updateIfChanged(m_indiP_loop, "number", m_loopNumber);

   updateIfChanged(m_indiP_loopGain, "current", m_gain);
   updateIfChanged(m_indiP_multCoeff, "current", m_multCoeff);
   updateIfChanged(m_indiP_maxLim, "current", m_maxLim);

   return 0;
}

int cacaoInterface::appShutdown()
{

   if(m_fmThread.joinable())
   {
      try
      {
         m_fmThread.join(); //this will throw if it was already joined
      }
      catch(...)
      {
      }
   }

   telemeterT::appShutdown();

   return 0;
}

int cacaoInterface::writeFPSCommand( const std::string &command, bool waitForReader )
{
    int wfd = open( m_fpsFifo.c_str(), waitForReader ? O_WRONLY : O_WRONLY | O_NONBLOCK );
    if( wfd < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, errno, "error opening " + m_fpsFifo } );
    }

    ssize_t written = write( wfd, command.c_str(), command.size() );
    int     error   = errno;
    close( wfd );

    if( written != static_cast<ssize_t>( command.size() ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error, "error on write to " + m_fpsFifo } );
    }

    return 0;
}

int cacaoInterface::setFPSVal( const std::string &fps,
                               const std::string &param,
                               const std::string &val,
                               bool               waitForReader )
{
    return writeFPSCommand( "setval " + fps + "-" + m_loopNumber + "." + param + " " + val + "\n", waitForReader );
}

template<typename T>
int cacaoInterface::setFPSVal( const std::string & fps,
                               const std::string & param,
                               const T & val
                             )
{
   return setFPSVal( fps, param, std::to_string(val));
}

std::string cacaoInterface::getFPSValStr( const std::string &fps, const std::string &param )
{
    std::string outfile = "/dev/shm/" + m_loopName + "_out_" + fps + "-" + m_loopNumber + "." + param;

    if( writeFPSCommand( "fwrval " + fps + "-" + m_loopNumber + "." + param + " " + outfile + "\n", true ) < 0 )
    {
        return "";
    }

    char inbuff [4096];

    int rfd = -1;
    int nr =0;
    while( rfd < 0 && nr < 500 )
    {
        rfd = open(outfile.c_str(), O_RDONLY);
        ++nr;
        mx::sys::milliSleep(10);
    }

    if( rfd < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "could not open FPS reply " + outfile } );
        return "";
    }

    int r = read( rfd, inbuff, sizeof( inbuff ) - 1 );

    if(r < 0)
    {
        log<software_error>({__FILE__, __LINE__, errno, "error on read from " + m_fpsFifo});
        return "";
    }

    close(rfd);

    remove(outfile.c_str());

    inbuff[r] = '\0';

    std::string instr = inbuff;

    size_t ned = instr.find_last_not_of(" \t\r\n");

    if(ned == std::string::npos || ned == 0)
    {
        //log<software_error>({__FILE__, __LINE__, "got empty result from " + m_fpsFifo});
        return "";
    }

    size_t nst = instr.rfind(' ', ned);

    if(nst == 0 || nst == std::string::npos || ned <= nst)
    {
        log<software_error>({__FILE__, __LINE__, "bad format in result from " + m_fpsFifo});
        return "";
    }

    return instr.substr( nst + 1, ned - nst );
}

std::string cacaoInterface::getFPSValNum( const std::string & fps,
                                          const std::string & param
                                        )
{
    std::string outfile = "/dev/shm/" + m_loopName + "_out_" + fps + "-" + m_loopNumber + "." + param;

    std::string comout = "fwrval " + fps + "-" + m_loopNumber + "." + param + " "  + outfile + "\n";

    int wfd = open( m_fpsFifo.c_str(), O_WRONLY);
    if(wfd < 0)
    {
        log<software_error>({__FILE__, __LINE__, errno, "error opening " + m_fpsFifo});
        return "";
    }

    int w = write(wfd, comout.c_str(), comout.size());

    if(w != (int) comout.size())
    {
        log<software_error>({__FILE__, __LINE__, errno, "error on write to " + m_fpsFifo});
        return "";
    }
    close(wfd);

    char inbuff [4096];

    int rfd = -1;
    int nr =0;
    while(rfd < 0 && nr < 500)
    {
        rfd = open(outfile.c_str(), O_RDONLY);
        ++nr;
        mx::sys::milliSleep(10);
    }

    if(rfd < 0)
    {
        log<software_error>({__FILE__, __LINE__, "could not get an open fd on " + m_fpsFifo});
        return "";
    }

    int r = read(rfd, inbuff, sizeof(inbuff));

    close(rfd);

    if(r < 0)
    {
        log<software_error>({__FILE__, __LINE__, errno, "error on read from " + m_fpsFifo});
        return "";
    }

    remove(outfile.c_str());

    inbuff[r] = '\0';

    std::string instr = inbuff;

    size_t ned = instr.find_last_not_of(" \t\r\n");

    if(ned == std::string::npos || ned == 0)
    {
        log<software_error>({__FILE__, __LINE__, "got empty result from " + m_fpsFifo});
        return "";
    }

    size_t nst = instr.rfind(' ', ned);

    if(nst == 0 || nst == std::string::npos || ned <= nst)
    {
        log<software_error>({__FILE__, __LINE__, "bad format in result from " + m_fpsFifo});
        return "";
    }

    return instr.substr(nst);
}

inline
int cacaoInterface::getAOCalib()
{
   std::string calsrc = "/milk/shm/aol" + m_loopNumber + "_calib_source.txt";

   std::ifstream fin;

   //First read in the milk/shm directory name, which could be to a symlinked directory
   fin.open(calsrc);
   if(!fin)
   {
      return 0;
   }
   fin >> calsrc;
   fin.close();

   //Now read in the actual directory
   calsrc += "/calib_dir.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getAOCalib failed to open: " + calsrc});
   }
   std::string aoCalDir;

   fin >> aoCalDir;

   fin.close();

   bool newcal = false;
   if(aoCalDir != m_aoCalDir)
   {
      m_aoCalDir = aoCalDir;
      newcal = true;
   }

   std::string nameFile = m_aoCalDir + "/LOOPNAME";
   fin.open(nameFile);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getAOCalib failed to open: " + nameFile});
   }
   fin >> m_loopName;
   fin.close();

   m_fpsFifo = "/milk/shm/" + m_loopName + "_fpsCTRL.fifo";

   calsrc = "/milk/shm/aol" + m_loopNumber + "_calib_loaded.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getAOCalib failed to open: " + calsrc});
   }
   std::string aoCalLoadTime;
   fin >> aoCalLoadTime;

   if(aoCalLoadTime != m_aoCalLoadTime)
   {
      newcal = true;
      m_aoCalLoadTime = aoCalLoadTime;
   }

   if(newcal)
   {
      log<text_log>("new calibration " + m_aoCalDir + " loaded at: " + m_aoCalLoadTime, logPrio::LOG_INFO);
   }

   fin.close();

   calsrc = m_aoCalDir + "/calib_archived.txt";

   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getAOCalib failed to open: " + calsrc});
   }

   while(!fin.eof())
   {
      fin >> m_aoCalArchiveTime;
   }
   fin.close();


   return 0;
}

int cacaoInterface::checkLoopProcesses()
{
   ///\todo look for actual evidence of processes, such as interrogating ps.

   m_loopProcesses = m_loopProcesses_stat;

   return 0;
}

int cacaoInterface::setGain()
{
   recordLoopGain(true);
   return setFPSVal("mfilt", "loopgain", m_gain_target);
}

int cacaoInterface::setMultCoeff()
{
   recordLoopGain(true);
   return setFPSVal("mfilt", "loopmult", m_multCoeff_target);
}

int cacaoInterface::setMaxLim()
{
   recordLoopGain(true);
   return setFPSVal("mfilt", "looplimit", m_maxLim_target);
}

int cacaoInterface::loopOn()
{
   recordLoopGain(true);
   if( setFPSVal( "mfilt", "loopON", std::string( "ON" ), false ) != 0 )
   {
      return log<software_error,-1>({__FILE__, __LINE__, "error setting FPS val"});
   }

   ++m_loopCommand;
   log<loop_closed>();

   return 0;

}

int cacaoInterface::loopOff()
{
   recordLoopGain(true);
   if( setFPSVal( "mfilt", "loopON", std::string( "OFF" ), false ) != 0 )
   {
      return log<software_error,-1>({__FILE__, __LINE__, "error setting FPS val"});
   }

   ++m_loopCommand;
   if(m_gain == 0)
   {
      log<loop_open>();
   }
   else
   {
      log<loop_paused>();
   }

   return 0;

}

int cacaoInterface::loopZero()
{
    int result = setFPSVal( "mfilt", "loopZERO", std::string( "ON" ), false );
    if( result == 0 )
    {
        log<text_log>( "loop zero requested", logPrio::LOG_NOTICE );
    }

    return result;
}

int cacaoInterface::requestLoopState( bool closed )
{
    if( closed && m_offloadLinked )
    {
        sendOffloaderSwitch( "offload", "toggle", pcf::IndiElement::On );
    }

    int result = closed ? loopOn() : loopOff();

    if( !closed && m_offloadLinked )
    {
        sendOffloaderSwitch( "offload", "toggle", pcf::IndiElement::Off );
    }

    return result;
}

void cacaoInterface::observeLoopState( bool closed )
{
    bool opened = m_loopState == 2 && !closed;
    m_loopState = closed ? 2 : 0;

    if( opened && m_offloadLinked )
    {
        sendOffloaderSwitch( "offload", "toggle", pcf::IndiElement::Off );
    }
}

void cacaoInterface::sendOffloaderSwitch( const char                       *name,
                                          const char                       *element,
                                          pcf::IndiElement::SwitchStateType requested )
{
    pcf::IndiProperty command( pcf::IndiProperty::Switch );
    command.setDevice( "t2wOffloader" );
    command.setName( name );
    command.add( pcf::IndiElement( element, requested ) );

    if( command.getName() == "offload" )
    {
        command.add( pcf::IndiElement( "linked_request", pcf::IndiElement::On ) );
    }

    sendNewProperty( command );
}

void cacaoInterface::setOffloadLinked( bool linked )
{
    std::lock_guard<std::mutex> lock( m_indiMutex );
    m_offloadLinked = linked;
    updateSwitchIfChanged( m_indiP_offloadLink,
                           "toggle",
                           linked ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           linked ? INDI_OK : INDI_IDLE );
}

void cacaoInterface::fmThreadStart( cacaoInterface * c )
{
   c->fmThreadExec();
}


void cacaoInterface::fmThreadExec( )
{
   m_fmThreadID = syscall(SYS_gettid);

   while( m_fmThreadInit == true && shutdown() == 0)
   {
      sleep(1);
   }

   while(shutdown() == 0)
   {
      if(m_fpsFifo == "")
      {
         sleep(1);
         continue;
      }

      unsigned loopCommand;
      { //mutex scope
          std::lock_guard<std::mutex> lock( m_indiMutex );
          loopCommand = m_loopCommand;
      }

      std::string ans = getFPSValStr( "mfilt", "loopON" );
      if( ans == "ON" || ans == "OFF" )
      {
          std::lock_guard<std::mutex> lock( m_indiMutex );
          if( loopCommand == m_loopCommand )
          {
              observeLoopState( ans == "ON" );
          }
      }

      ans = getFPSValNum("mfilt", "loopgain");

      try
      {
         m_gain = std::stof(ans);
      }
      catch(const std::exception& e)
      {
         m_gain = 0;
      }

      ans = getFPSValNum("mfilt", "loopmult");
      try
      {
         m_multCoeff = std::stof(ans);
      }
      catch(...)
      {
         m_multCoeff = 0;
      }
      ans = getFPSValNum("mfilt", "looplimit");
      try
      {
         m_maxLim = std::stof(ans);
      }
      catch(...)
      {
         m_maxLim = 0;
      }

      recordLoopGain();

      mx::sys::milliSleep(500);

   }

   return;
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_loopState )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_loopState, ipRecv);

    if(!ipRecv.find("toggle")) return 0;

    std::unique_lock<std::mutex> lock(m_indiMutex);

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        return requestLoopState( true );
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
    {
        return requestLoopState( false );
    }

    log<software_error>({__FILE__,__LINE__, "switch state fall through."});
    return -1;
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_loopGain )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_loopGain, ipRecv);

    float current = -1;
    float target = -1;

    if(ipRecv.find("current"))
    {
        current = ipRecv["current"].get<double>();
    }

    if(ipRecv.find("target"))
    {
        target = ipRecv["target"].get<double>();
    }

    if(target == -1) target = current;

    if(target == -1)
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard(m_indiMutex);

    m_gain_target = target;

    updateIfChanged(m_indiP_loopGain, "target", m_gain_target);

    return setGain();
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_loopZero )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_loopZero, ipRecv);

    if( !ipRecv.find( "request" ) || ipRecv["request"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    std::lock_guard<std::mutex> lock( m_indiMutex );
    int                         result = loopZero();

    if( m_offloadLinked )
    {
        sendOffloaderSwitch( "zero", "request", pcf::IndiElement::On );
    }

    updateSwitchIfChanged( m_indiP_loopZero, "request", pcf::IndiElement::Off, result == 0 ? INDI_OK : INDI_ALERT );
    return result;
}

INDI_NEWCALLBACK_DEFN( cacaoInterface, m_indiP_offloadLink )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadLink, ipRecv );
    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    auto toggle = ipRecv["toggle"].getSwitchState();
    if( toggle != pcf::IndiElement::On && toggle != pcf::IndiElement::Off )
    {
        return 0;
    }

    setOffloadLinked( toggle == pcf::IndiElement::On );
    return 0;
}

INDI_NEWCALLBACK_DEFN( cacaoInterface, m_indiP_offloadOverride )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_offloadOverride, ipRecv );

    if( ipRecv.find( "request" ) && ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        setOffloadLinked( false );
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_multCoeff )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_multCoeff, ipRecv);

    float current = -1;
    float target = -1;

    if(ipRecv.find("current"))
    {
        current = ipRecv["current"].get<double>();
    }

    if(ipRecv.find("target"))
    {
        target = ipRecv["target"].get<double>();
    }

    if(target == -1) target = current;

    if(target == -1)
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard(m_indiMutex);

    m_multCoeff_target = target;
    updateIfChanged(m_indiP_multCoeff, "target", target);

    return setMultCoeff();
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_maxLim )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_maxLim, ipRecv);

    float current = -1;
    float target = -1;

    if(ipRecv.find("current"))
    {
        current = ipRecv["current"].get<double>();
    }

    if(ipRecv.find("target"))
    {
        target = ipRecv["target"].get<double>();
    }

    if(target == -1) target = current;

    if(target == -1)
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard(m_indiMutex);
    m_maxLim_target = target;
    updateIfChanged(m_indiP_maxLim, "target", target);

    return setMaxLim();
}

inline
int cacaoInterface::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_loopgain());
}

inline
int cacaoInterface::recordTelem( const telem_loopgain * )
{
   return recordLoopGain(true);
}

inline
int cacaoInterface::recordLoopGain( bool force )
{
   static uint8_t state {0};
   static float gain {-1000};
   static float multcoef {0};
   static float limit {0};

   if(state != m_loopState || gain != m_gain || multcoef != m_multCoeff || limit != m_maxLim || force)
   {
      state = m_loopState;
      gain = m_gain;
      multcoef = m_multCoeff;
      limit = m_maxLim;

      telem<telem_loopgain>({state, m_gain, m_multCoeff, m_maxLim});
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //cacaoInterface_hpp
