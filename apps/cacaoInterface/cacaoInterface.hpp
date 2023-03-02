/** \file cacaoInterface.hpp
  * \brief The MagAO-X CACAO Interface header file
  *
  * \ingroup cacaoInterface_files
  */

#ifndef cacaoInterface_hpp
#define cacaoInterface_hpp



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

   typedef dev::telemeter<cacaoInterface> telemeterT;

   friend class dev::telemeter<cacaoInterface>;

protected:

   /** \name Configurable Parameters
     *@{
     */
   std::string m_loopNumber; ///< The loop number, X in aolX.  We keep it a string because that's how it gets used.
   
   ///@}

   std::string m_aoCalDir;
   std::string m_aoCalArchiveTime;
   std::string m_aoCalLoadTime;
   
   std::string m_loopName; ///< the loop name
   

   std::string m_fpsName;
   std::string m_fpsFifo;

   
   int m_loopState {0}; ///< The loop state.  0 = off, 1 = paused (on, 0 gain), 2 = on
   
   bool m_loopProcesses {false}; ///< Status of the loop processes.
   bool m_loopProcesses_stat {false}; ///< What the cacao status file says the state of loop processes is.
   
   float m_gain {0.0}; ///< The current loop gain.
   float m_gain_target {0.0}; ///< The target loop gain.
   
   float m_multCoeff {0.0}; ///< The current multiplicative coefficient (1-leak)
   float m_multCoeff_target {0.0}; ///< The target multiplicative coefficient (1-leak)
   
   std::vector<int> m_modeBlockStart;
   std::vector<int> m_modeBlockN;

   std::vector<float> m_modeBlockGains;
   std::vector<float> m_modeBlockMCs;
   std::vector<float> m_modeBlockLims;

   std::mutex m_modeBlockMutex;


   float m_maxLim {0.0}; ///< The current max limit
   float m_maxLim_target {0.0}; ///< The target max limit
   
public:
   /// Default c'tor.
   cacaoInterface();

   /// D'tor, declared and defined for noexcept.
   ~cacaoInterface() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

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

   int setFPSVal( const std::string & fps,
                  const std::string & param,
                  const std::string & val
                );


   template<typename T>
   int setFPSVal( const std::string & fps,
                  const std::string & param,
                  const T & val
                );

   std::string getFPSValStr(const std::string & fps,
                                          const std::string & param );

   std::string getFPSValNum(const std::string & fps,
                                          const std::string & param );

   /// Get the calibration details
   /** This is done each loop
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */   
   int getAOCalib();
   
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
   
   /// Zero the loop control channel
   /**
     * \returns 0 on success
     * \returns -1 on an error
     */ 
   int loopZero();

   /// @}
   
   /** \name File Monitoring Thread
     * Monitors CACAO files for changes
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
   
   pcf::IndiProperty m_indiP_loop;
   pcf::IndiProperty m_indiP_loopProcesses;
   pcf::IndiProperty m_indiP_modes;

   pcf::IndiProperty m_indiP_loopState;
   pcf::IndiProperty m_indiP_loopZero;
   pcf::IndiProperty m_indiP_loopGain;
   pcf::IndiProperty m_indiP_multCoeff;
   pcf::IndiProperty m_indiP_maxLim;
         
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopState);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopZero);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopGain);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_multCoeff);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_maxLim);

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_loopgain * );

   int recordLoopGain( bool force = false );
   
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
   m_indiP_loop["number"] = m_loopName;
   registerIndiPropertyReadOnly(m_indiP_loop);
   
   createROIndiNumber( m_indiP_modes, "modes", "Loop Modes", "Loop Controls");
   indi::addNumberElement(m_indiP_modes, "total", 0, 1, 25000, "Total Modes");
   indi::addNumberElement(m_indiP_modes, "blocks", 0, 1, 99, "Mode Blocks");
   registerIndiPropertyReadOnly(m_indiP_modes);

   createStandardIndiToggleSw( m_indiP_loopProcesses, "loop_processes", "Loop Processes", "Loop Controls");
   registerIndiPropertyReadOnly( m_indiP_loopProcesses);  

   createStandardIndiToggleSw( m_indiP_loopState, "loop_state", "Loop State", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopState, INDI_NEWCALLBACK(m_indiP_loopState) );  
   
   createStandardIndiRequestSw( m_indiP_loopZero, "loop_zero", "Loop Zero", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopZero,INDI_NEWCALLBACK(m_indiP_loopZero) );  
   
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

   /*if(checkLoopProcesses() < 0)
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get loop name and/or number", logPrio::LOG_ERROR);
      return 0;
   }   
   */

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
   
   if(m_loopState == 0)
   {
      updateSwitchIfChanged(m_indiP_loopState, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   }
   else if(m_loopState == 1)
   {
      updateSwitchIfChanged(m_indiP_loopState, "toggle", pcf::IndiElement::On, INDI_IDLE);
   }
   else if(m_loopState == 2)
   {
      updateSwitchIfChanged(m_indiP_loopState, "toggle", pcf::IndiElement::On, INDI_BUSY);
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

int cacaoInterface::setFPSVal( const std::string & fps,
                               const std::string & param,
                               const std::string & val
                             )
{
   std::string comout = "setval " + fps + "-" + m_loopNumber + "." + param +  " "  + val + "\n";

   int wfd = open( m_fpsFifo.c_str(), O_WRONLY);
   if(wfd < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, "error opening " + m_fpsFifo});
      return -1;
   }

   int w = write(wfd, comout.c_str(), comout.size());
      
   if(w != (int) comout.size())
   {
      log<software_error>({__FILE__, __LINE__, errno, "error on write to " + m_fpsFifo});
      return -1;
   }
      
   close(wfd);

   return 0;
}


template<typename T>
int cacaoInterface::setFPSVal( const std::string & fps,
                               const std::string & param,
                               const T & val
                             )
{
   return setFPSVal( fps, param, std::to_string(val));
}

std::string cacaoInterface::getFPSValStr( const std::string & fps,
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
      while(rfd < 0 && nr < 20)
      {
         rfd = open(outfile.c_str(), O_RDONLY);
         ++nr;
         mx::sys::milliSleep(10);
      }

      int r = read(rfd, inbuff, sizeof(inbuff));

      if(r < 0)
      {
         log<software_error>({__FILE__, __LINE__, errno, "error on read from " + m_fpsFifo});
         return "";
      }

      close(rfd);

      remove(outfile.c_str());
      
      int n = strnlen(inbuff, sizeof(inbuff));

      char * s = inbuff + n;

      while(s != inbuff && *s != ' ') 
      {
         if(*s == '\n' || *s == 'r') *s = '\0';
         --s;
      }
      if(s == inbuff)
      {
         log<software_error>({__FILE__, __LINE__, errno, "error parsing result from " + m_fpsFifo});
         return "";
      }

      ++s;

      return s;
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
      while(rfd < 0 && nr < 20)
      {
         rfd = open(outfile.c_str(), O_RDONLY);
         ++nr;
         mx::sys::milliSleep(10);
      }

      int r = read(rfd, inbuff, sizeof(inbuff));

      close(rfd);

      if(r < 0)
      {
         log<software_error>({__FILE__, __LINE__, errno, "error on read from " + m_fpsFifo});
         return "";
      }

      remove(outfile.c_str());
      
      int n = strlen(inbuff);

      char * s = inbuff + n;

      int ns = 0;
      while(s != inbuff && ns < 4) 
      {
         if(*s == ' ') ++ns;
         if(ns == 4) break;
         if(*s == '\n' || *s == 'r' || *s == ' ') *s = '\0';
         --s;
      }
      if(s == inbuff)
      {
         log<software_error>({__FILE__, __LINE__, errno, "error parsing result from " + m_fpsFifo});
         return "";
      }
      ++s;
      return s;
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
   calsrc += "/aol" +  m_loopNumber + "_calib_dir.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to open: " + calsrc});
   }
   fin >> m_aoCalDir;
   fin.close();

   std::string name = "";
   size_t np = m_aoCalDir.rfind('/');
   int nf = 1;
   while(np != std::string::npos && np != 0 && nf < 4)
   {
      np = m_aoCalDir.rfind('/',np-1);
      ++nf;
   }

   if(np == std::string::npos)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to find loop name in: " + m_aoCalDir});
   }

   ++np;
   size_t ne = m_aoCalDir.find('/', np+1);
   
   if(ne == std::string::npos)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to find loop name in: " + m_aoCalDir});
   }

   
   m_loopName = m_aoCalDir.substr(np, ne-np);

   m_fpsFifo = "/milk/shm/" + m_loopName + "_fpsCTRL.fifo";

   calsrc = "/milk/shm/aol" + m_loopNumber + "_calib_loaded.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to open: " + calsrc});
   }
   fin >> m_aoCalLoadTime;
   fin.close();
   

   calsrc = m_aoCalDir + "/aol" + m_loopNumber + "_calib_archived.txt";
   
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to open: " + calsrc});
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
   if( setFPSVal("mfilt", "loopON", std::string("ON")) != 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "error setting FPS val"});
   }
   
   log<loop_closed>();
      
   return 0;
   
}

int cacaoInterface::loopOff()
{
   recordLoopGain(true);
   if( setFPSVal("mfilt", "loopON", std::string("OFF")) != 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "error setting FPS val"});
   }

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
   if( setFPSVal("mfilt", "loopZERO", std::string("ON")) != 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, "error setting FPS val"});
   }

   log<text_log>("loop zeroed", logPrio::LOG_NOTICE);
   
   return 0;

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

      std::string ans = getFPSValStr("mfilt", "loopON");

      if(ans[1] == 'F')
      {
         m_loopState = 0;
        /* if(m_gain == 0)
         {
            m_loopState = 0; //open
         }
         else 
         {
            m_loopState = 1; //paused -- gains set, but loop not updating so leak not in effect
         }*/
      }
      else m_loopState = 2; //closed

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
      /*
      fin.open( m_loopDir +  "/status/stat_procON.txt");
      
      if(fin.is_open()) 
      {
         fin >> proc;
      }
      fin.close();
      
      */

      mx::sys::milliSleep(1000);

   }
   
   return;
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_loopState )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_loopState.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
      
   if(!ipRecv.find("toggle")) return 0;
           
   std::unique_lock<std::mutex> lock(m_indiMutex);
      
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      return loopOn();
   }   
   else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      return loopOff();
   }
      
   log<software_error>({__FILE__,__LINE__, "switch state fall through."});
   return -1;
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_loopGain )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_loopGain.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
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
   if(ipRecv.getName() != m_indiP_loopZero.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
      
   if(!ipRecv.find("request")) return 0;
           
   std::unique_lock<std::mutex> lock(m_indiMutex);
      
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      return loopZero();
   }   
   std::cerr << "off?\n";
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(cacaoInterface, m_indiP_multCoeff )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_multCoeff.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
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
   if(ipRecv.getName() != m_indiP_maxLim.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
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
