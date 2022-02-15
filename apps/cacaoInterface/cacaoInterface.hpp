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
class cacaoInterface : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class cacaoInterface_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   std::string m_loopDir;
   
   ///@}

   std::string m_loopName;   ///< The loop name.
   std::string m_loopNumber; ///< The loop number, X in aolX.  We keep it a string because that's how it gets used.
   
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
   
   /// Get the loop name and number
   /** This is done once at app startup.
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int getLoopNameNumber();
   
   int getModeBlocks();

   /// Update a status variable in CACAO
   /** In the current version of cacao this means writing to a text file.
     *
     * \returns 0 on success
     * \returns -1 on an error
     */
   template<typename T>
   int updateStatus( const std::string & path, ///< [in] path the to the status variable
                     T & val                   ///< [in] the value to set
                   );
   
   /// Send keys to a tmux session 
   /** Executes "tmux send-keys -t session keys C-m"
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int tmuxSendKeys( const std::string & session,
                     const std::string & keys
                   );
   
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
   
   /// Set the block gain the specified value 
   /**
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setBlockGain( int n,  ///< [in] the block number
                     float g ///< [in] the desired gain
                   );

   /// Set loop multiplication coefficient to the value of m_multCoeff_target;
   /** 
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */ 
   int setMultCoeff();
   
   /// Set the block mult coeff to the specified value 
   /**
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setBlockMC( int n,  ///< [in] the block number
                   float mc ///< [in] the desired mult. coeff
                 );

   /// Set loop max lim to the value of m_maxLim_target;
   /** 
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */ 
   int setMaxLim();
   
   /// Set the block limit to the specified value 
   /**
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setBlockLimit( int n,  ///< [in] the block number
                      float l ///< [in] the desired limit
                    );

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
   
   pcf::IndiProperty m_indiP_loop;
   pcf::IndiProperty m_indiP_loopProcesses;
   pcf::IndiProperty m_indiP_modes;

   pcf::IndiProperty m_indiP_loopState;
   pcf::IndiProperty m_indiP_loopGain;
   pcf::IndiProperty m_indiP_multCoeff;
   pcf::IndiProperty m_indiP_maxLim;
      
   std::vector<pcf::IndiProperty> m_indiP_blockGains;
   std::vector<pcf::IndiProperty> m_indiP_blockMCs;
   std::vector<pcf::IndiProperty> m_indiP_blockLimits;
   
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopState);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_loopGain);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_multCoeff);
   INDI_NEWCALLBACK_DECL(cacaoInterface, m_indiP_maxLim);

   /// The static callback function to be registered for block gains
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_blockGains( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                         const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                       );

   /// Callback to process a NEW block gain request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_blockGains( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// The static callback function to be registered for block mult. coeff.s
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_blockMCs( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                       const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                     );

   /// Callback to process a NEW block mult. coeff.s
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_blockMCs( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// The static callback function to be registered for block limits
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_blockLimits( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                        const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                      );

   /// Callback to process a NEW block limits
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_blockLimits( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

};

cacaoInterface::cacaoInterface() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void cacaoInterface::setupConfig()
{
   config.add("loop.dir", "", "loop.dir", argType::Required, "loop", "dir", false, "string", "The loop working directory.");
}

int cacaoInterface::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_loopDir, "loop.dir");
   
   return 0;
}

void cacaoInterface::loadConfig()
{
   loadConfigImpl(config);
}

int cacaoInterface::appStartup()
{
   
   if(m_loopDir == "")
   {
      log<text_log, -1>("loop directory not set", logPrio::LOG_CRITICAL);
   }
   
   if(m_loopDir.back() != '/')
   {
      m_loopDir += '/';
   }

   createROIndiText( m_indiP_loop, "loop", "name", "Loop Description", "Loop Controls", "Name");
   indi::addTextElement(m_indiP_loop, "number", "Number");
   indi::addTextElement(m_indiP_loop, "working_dir", "Working Directory");
   m_indiP_loop["working_dir"] = m_loopDir;
   registerIndiPropertyReadOnly(m_indiP_loop);
   
   createROIndiNumber( m_indiP_modes, "modes", "Loop Modes", "Loop Controls");
   indi::addNumberElement(m_indiP_modes, "total", 0, 1, 25000, "Total Modes");
   indi::addNumberElement(m_indiP_modes, "blocks", 0, 1, 99, "Mode Blocks");
   registerIndiPropertyReadOnly(m_indiP_modes);

   createStandardIndiToggleSw( m_indiP_loopState, "loop_state", "Loop State", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopState, INDI_NEWCALLBACK(m_indiP_loopState) );  
   
   createStandardIndiToggleSw( m_indiP_loopProcesses, "loop_processes", "Loop Processes", "Loop Controls");
   registerIndiPropertyReadOnly( m_indiP_loopProcesses);  
   
   createStandardIndiNumber<float>( m_indiP_loopGain, "loop_gain", 0.0, 10.0, 0.01, "%0.3f", "Loop Gain", "Loop Controls");
   registerIndiPropertyNew( m_indiP_loopGain, INDI_NEWCALLBACK(m_indiP_loopGain) );  
   
   createStandardIndiNumber<float>( m_indiP_multCoeff, "loop_multcoeff", 0.0, 1.0, 0.001, "%0.3f", "Mult. Coefficient", "Loop Controls");
   registerIndiPropertyNew( m_indiP_multCoeff, INDI_NEWCALLBACK(m_indiP_multCoeff) );  
   
   createStandardIndiNumber<float>( m_indiP_maxLim, "loop_max_limit", 0.0, 10.0, 0.001, "%0.3f", "Max. Limit", "Loop Controls");
   registerIndiPropertyNew( m_indiP_maxLim, INDI_NEWCALLBACK(m_indiP_maxLim) );  
   
   if(getLoopNameNumber() < 0)
   {
      return log<text_log, -1>("Could not get loop name and/or number", logPrio::LOG_CRITICAL);
   }
   
   if(getModeBlocks() < 0)
   {
      return log<text_log, -1>("Could not get mode blocks", logPrio::LOG_CRITICAL);
   }

   if(threadStart( m_fmThread, m_fmThreadInit, m_fmThreadID, m_fmThreadProp, m_fmThreadPrio, "loopmon", this, fmThreadStart) < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
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
   
   //We check this once per loop as a basic health check, and to catch any changes
   if(getLoopNameNumber() < 0)
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get loop name and/or number", logPrio::LOG_ERROR);
      return 0;
   }

   //These could change if a new calibration is loaded
   if(getModeBlocks() < 0 )
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get mode blocks", logPrio::LOG_ERROR);
      return 0;
   }

   if(checkLoopProcesses() < 0)
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get loop name and/or number", logPrio::LOG_ERROR);
      return 0;
   }   
   
   if(m_loopProcesses == 0 || m_loopState == 0) state(stateCodes::READY);
   else state(stateCodes::OPERATING);

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
   
   updateIfChanged(m_indiP_loopGain, "current", m_gain);
   updateIfChanged(m_indiP_multCoeff, "current", m_multCoeff);
   updateIfChanged(m_indiP_maxLim, "current", m_maxLim);

   for(size_t n=0; n < m_indiP_blockGains.size(); ++n)
   {
      updateIfChanged(m_indiP_blockGains[n], "current", m_modeBlockGains[n]);
   }

   for(size_t n=0; n < m_indiP_blockMCs.size(); ++n)
   {
      updateIfChanged(m_indiP_blockMCs[n], "current", m_modeBlockMCs[n]);
   }

   for(size_t n=0; n < m_indiP_blockLimits.size(); ++n)
   {
      updateIfChanged(m_indiP_blockLimits[n], "current", m_modeBlockLims[n]);
   }

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
   
   return 0;
}

int cacaoInterface::getLoopNameNumber()
{
   std::ifstream fin;
   
   fin.open(m_loopDir + "LOOPNAME");
   
   if( !fin )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber failed to open LOOPNAME file"});
   }
   
   if(! (fin >> m_loopName) )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber error reading LOOPNAME"});
   }
   
   fin.close();
   
   if( !fin )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber error on LOOPNAME file"});
   }
   
   if(m_loopName.size() == 0 )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber no LOOPNAME set"});
   }
   
   fin.open(m_loopDir + "LOOPNUMBER");
   
   if( !fin )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber failed to open LOOPNUMBER file"});
   }
   
   if(! (fin >> m_loopNumber) )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber error reading LOOPNUMBER"});
   }
   
   fin.close();
   
   if( !fin )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber error on LOOPNUMBER file"});
   }
   
   if(m_loopNumber.size() == 0 )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getLoopNameNumber no LOOPNUMBER set"});
   }
   
   
   return 0;
   
}

int cacaoInterface::getModeBlocks()
{

   std::vector<std::string> blocks = mx::ioutils::getFileNames(m_loopDir + "conf/", "param_blockoffset_", "", ".txt");

   if(blocks.size() == 0)
   {
      log<text_log>("no mode blocks found", logPrio::LOG_WARNING);
   }

   size_t Nb = blocks.size();

   std::vector<int> modeBlockStart(Nb);
   std::vector<int> modeBlockN(Nb);

   

   std::ifstream fin;
   
   for(size_t n =0; n < Nb; ++n)
   {
      char str[16];
      int nn = n;
      snprintf(str, sizeof(str), "%02d", nn);
      std::string fn = m_loopDir + "conf/param_blockoffset_" + str + ".txt";

      fin.open(fn);
      if( !fin )
      {
         return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getModeBlocks failed to open: " + fn});
      }

      int offs, num;

      fin >> offs;
      fin >> num;

      modeBlockStart[n] = offs;
      modeBlockN[n] = num;
      fin.close();

      if( !fin )
      {
         return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::getModeBlocks error closing: " + fn});
      }
   }

   //now detect changes.
   bool changed = false;
   if(m_modeBlockStart.size() != Nb || m_modeBlockN.size() != Nb || m_modeBlockGains.size() != Nb || m_modeBlockMCs.size() != Nb || m_modeBlockLims.size() != Nb)
   {
      changed = true;
   }
   else
   {
      for(size_t n=0; n < Nb; ++n)
      {
         if(m_modeBlockStart[n] != modeBlockStart[n] || m_modeBlockN[n] != modeBlockN[n])
         {
            changed = true;
            break;
         }
      }
   }

   if(changed)
   {
      std::unique_lock<std::mutex> lock(m_modeBlockMutex);
      m_modeBlockStart.resize(Nb);
      m_modeBlockN.resize(Nb);
      m_modeBlockGains.resize(Nb);
      m_modeBlockMCs.resize(Nb);
      m_modeBlockLims.resize(Nb);

      for(size_t n=0; n < Nb; ++n)
      {
         m_modeBlockStart[n] = modeBlockStart[n];
         m_modeBlockN[n] = modeBlockN[n];
      }
      lock.unlock();

      //-- modify INDI vars --
      std::unique_lock<std::mutex> indilock(m_indiMutex);
      m_indiP_modes["total"] = m_modeBlockStart[m_modeBlockStart.size()-1] + m_modeBlockN[m_modeBlockN.size()-1];
      m_indiP_modes["blocks"] = m_modeBlockStart.size();

      //First just delete all existing blockXX elements
      for(int n=0; n < 100; ++n)
      {
         char str[16];
         snprintf(str, sizeof(str), "%02d", n);
         std::string en = "block";
         en += str;

         if(m_indiP_modes.find(en)) m_indiP_modes.remove(en);
      }
      
      //Erase existing block gains
      if(m_indiP_blockGains.size() > 0)
      {
         for(size_t n=0; n < m_indiP_blockGains.size(); ++n)
         {
            if(m_indiDriver) m_indiDriver->sendDelProperty(m_indiP_blockGains[n]);
            m_indiNewCallBacks.erase(m_indiP_blockGains[n].getName());
         }
      }
      m_indiP_blockGains.clear(); //I don't know what happens if you try to re-use an INDI property

      if(m_indiP_blockMCs.size() > 0)
      {
         for(size_t n=0; n < m_indiP_blockMCs.size(); ++n)
         {
            if(m_indiDriver) m_indiDriver->sendDelProperty(m_indiP_blockMCs[n]);
            m_indiNewCallBacks.erase(m_indiP_blockMCs[n].getName());
         }
      }
      m_indiP_blockMCs.clear(); //I don't know what happens if you try to re-use an INDI property

      if(m_indiP_blockLimits.size() > 0)
      {
         for(size_t n=0; n < m_indiP_blockLimits.size(); ++n)
         {
            if(m_indiDriver) m_indiDriver->sendDelProperty(m_indiP_blockLimits[n]);
            m_indiNewCallBacks.erase(m_indiP_blockLimits[n].getName());
         }
      }
      m_indiP_blockLimits.clear(); //I don't know what happens if you try to re-use an INDI property

      m_indiP_blockGains.resize(Nb);
      m_indiP_blockMCs.resize(Nb);
      m_indiP_blockLimits.resize(Nb);

      //Then add in what we want.
      for(size_t n=0; n < Nb; ++n)
      {
         char str[16];
         int nn = n;
         snprintf(str, sizeof(str), "%02d", nn);
         std::string en = "block";
         en += str;
         indi::addNumberElement(m_indiP_modes, en, 0, 1, 99, "Block " + std::to_string(nn));
         m_indiP_modes[en] = m_modeBlockN[n];

         createStandardIndiNumber<float>( m_indiP_blockGains[n], en + "_gain", 0.0, 10.0, 0.01, "%0.3f", en + " Gain", "Loop Controls");
         registerIndiPropertyNew( m_indiP_blockGains[n],  st_newCallBack_blockGains);  
         if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_blockGains[n]);

         createStandardIndiNumber<float>( m_indiP_blockMCs[n], en + "_multcoeff", 0.0, 1.0, 0.01, "%0.3f", en + " Mult. Coeff", "Loop Controls");
         registerIndiPropertyNew( m_indiP_blockMCs[n],  st_newCallBack_blockMCs);  
         if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_blockMCs[n]);

         createStandardIndiNumber<float>( m_indiP_blockLimits[n], en + "_limit", 0.0, 100.0, 0.01, "%0.3f", en + " Limit", "Loop Controls");
         registerIndiPropertyNew( m_indiP_blockLimits[n],  st_newCallBack_blockLimits);  
         if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_blockLimits[n]);
      }

      if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_modes); //might not exist yet!
      
   }
   

   return 0;
}

template<typename T>
int cacaoInterface::updateStatus( const std::string & path,
                                  T & val                 
                                )
{
   std::ofstream fout;
   
   fout.open(m_loopDir + path);
   
   if( !fout )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::updateStatus failed to open output file"});
   }
   
   if( !(fout << val << std::endl)) //Flush to get errors.
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::updateStatus failed writing to output file"});
   }
   
   fout.close();
   
   if( !fout )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::updateStatus error in output file"});
   }
   
   return 0;
}

int cacaoInterface::tmuxSendKeys( const std::string & session,
                                  const std::string & keys
                                )
{
   std::vector<std::string> tmuxcom = {"tmux", "send-keys", "-t", session, keys, "C-m"};
   std::vector<std::string> errout, output;
   
   if(sys::runCommand(output, errout, tmuxcom) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::tmuxSendKeys error executing command"});
   }

   if(errout.size() > 0)
   {
      //Here we ignore session not found, since it just means AO loop not running
      if( errout[0].find("session not found") == std::string::npos)
      {
         for(size_t n=0; n < errout.size(); ++n)
         {
            log<text_log>("tmuxSendKeys stderr: " + errout[n], logPrio::LOG_ERROR);
         }
         return -1;
      }
   }
   
   //should be empty, but for later investigation if something comes out
   if(output.size() > 0)
   {
      std::cout << "send-keys returned: ";
      for(size_t n=0; n<output.size(); ++n) std::cout << "\n   " << output[n];
      std::cout << '\n';
   }
   
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
   //This executes the commands in the file aolconfscripts/aolconf_menucontrolloop for "g)"
   
   //echo "$loopgain" > ./conf/param_loopgain.txt
   if(updateStatus("conf/param_loopgain.txt", m_gain_target) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setGain error updating gain status"});
   }
   
   //tmux send-keys -t aol${LOOPNUMBER}-ctr "aolsetgain ${loopgain}" C-m
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", "aolsetgain " + std::to_string(m_gain_target)) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setGain error from tmuxSendKeys"});
   }
   
   //aoconfLogStatusUpdate "${SLOOPNAME}_gain ${loopgain}"
   //aoconflogext "set gain ${loopgain}"

   return 0;
}

int cacaoInterface::setBlockGain( int n,
                                  float g
                                )
{

   //This executes the commands in the file aolconfscripts/aolconf_menucontrolloop for "gbXX)"

   char gi[16];
   snprintf(gi, sizeof(gi), "%02d", n);

   //echo "${gainb[10#${gi}]}" > ./conf/param_gainb${gi}.txt
   if(updateStatus(std::string("conf/param_gainb") + gi + ".txt", g) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setBlockGain error updating gain status"});
   }

   //tmux send-keys -t aol${LOOPNUMBER}-ctr "aolsetmbgain ${gi} ${gainb[10#${gi}]} 1" C-m
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", std::string("aolsetmbgain ") + gi + " " + std::to_string(g) + " 1") < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setBlockGain error from tmuxSendKeys"});
   }

   //aoconflogext "block ${gi} set gain to ${value}"

   return 0;
}

int cacaoInterface::setMultCoeff()
{
   //This executes the commands in the file aolconfscripts/aolconf_menucontrolloop for "e)"
   
   //echo "$loopmultcoeff" > ./conf/param_loopmultcoeff.txt
   if(updateStatus("conf/param_loopmultcoeff.txt", m_multCoeff_target) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setMultCoeff error updating mult coeff status"});
   }
   
   //tmux send-keys -t aol${LOOPNUMBER}-ctr "aolsetmult ${loopmultcoeff}" C-m
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", "aolsetmult " + std::to_string(m_multCoeff_target)) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setMultCoeff error from tmuxSendKeys"});
   }
   
   //aoconfLogStatusUpdate "${SLOOPNAME}_leak ${loopmultcoeff}"
   //aoconflogext "set mult coeff ${loopmultcoeff}"

   return 0;
}

int cacaoInterface::setBlockMC( int n,
                                float mc
                              )
{

   //This executes the commands in the file aolconfscripts/aolconf_menucontrolloop for "mbXX)"

   char gi[16];
   snprintf(gi, sizeof(gi), "%02d", n);

   //echo "${gainb[10#${gi}]}" > ./conf/param_multfb${gi}.txt
   if(updateStatus(std::string("conf/param_multfb") + gi + ".txt", mc) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setBlockMC error updating gain status"});
   }

   //tmux send-keys -t aol${LOOPNUMBER}-ctr "aolsetmultfb ${gi} ${multfb[10#${gi}]}" C-m
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", std::string("aolsetmultfb ") + gi + " " + std::to_string(mc)) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setBlockMC error from tmuxSendKeys"});
   }


   return 0;
}

int cacaoInterface::setMaxLim()
{
   //This executes the commands in the file aolconfscripts/aolconf_menucontrolloop for "m)"
   
   //echo "$loopmaxlim" > ./conf/param_loopmaxlim.txt
   if(updateStatus("conf/param_loopmaxlim.txt", m_maxLim_target) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setMaxLim error updating max lim status"});
   }
   
   //tmux send-keys -t aol${LOOPNUMBER}-ctr "aolsetmaxlim ${loopmaxlim}" C-m
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", "aolsetmaxlim " + std::to_string(m_maxLim_target)) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setMaxLim error from tmuxSendKeys"});
   }

   //aoconflogext "set max limit ${loopmaxlim}"

   return 0;
}

int cacaoInterface::setBlockLimit( int n,
                                   float l
                                 )
{

   //This executes the commands in the file aolconfscripts/aolconf_menucontrolloop for "lbXX)"

   char gi[16];
   snprintf(gi, sizeof(gi), "%02d", n);

   //echo "${gainb[10#${gi}]}" > ./conf/param_limitb${gi}.txt
   if(updateStatus(std::string("conf/param_limitb") + gi + ".txt", l) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setBlockLim error updating gain status"});
   }

   //tmux send-keys -t aol${LOOPNUMBER}-ctr "aolsetlimitb ${gi} ${limitb[10#${gi}]}" C-m
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", std::string("aolsetlimitb ") + gi + " " + std::to_string(l)) < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::setBlockLim error from tmuxSendKeys"});
   }

   return 0;
}

int cacaoInterface::loopOn()
{
   if(!m_loopProcesses)
   {
      log<text_log>("loop processes are not running.", logPrio::LOG_WARNING);
      return 0;
   }
   
   //This is equivalent to function_LOOP_ON "NULL" # in aolconf_controlloop_funcs
   //But we don't write the script, we just execute everything directly.
   
   //echo "echo \" ON\" > ./status/stat_loopON.txt" >> $scriptfile
   if(updateStatus("status/stat_loopON.txt", " ON") < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::loopOn error updating loop status"});
   }
   
   //echo "./aolconfscripts/aollog -e \"$LOOPNAME\" \"LOOP ON [gain = ${loopgain}   maxlim = ${loopmaxlim}   multcoeff = ${loopmultcoeff}]\"" >> $scriptfile
   
   //echo "tmux send-keys -t aol${LOOPNUMBER}-ctr \"aolon\" C-m" >> $scriptfile
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", "aolon") < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::loopOn error from tmuxSendKeys"});
   }
   
   log<loop_closed>();
   
   ///\todo we don't seem to have setupAOloopON, do we need it?
   //echo "./setupAOloopON" >> $scriptfile
   
   //if [ "$logMode" = "1" ]; then   # log when loop is closed
   //      start_Telemetrylog_all
   //fi
      
   ///\todo implement telemetry logging from cacao
   /*if(m_logMode == 1)
   {
      //start_Telemetrylog_all
   }*/
   
   return 0;
   
}

int cacaoInterface::loopOff()
{
   //This is equivalent to function_LOOP_OFF "NULL" # in aolconf_controlloop_funcs
   //But we don't write the script, we just execute everything directly.
   
   //echo "echo \" OFF\" > ./status/stat_loopON.txt" >> $scriptfile
   if(updateStatus("status/stat_loopON.txt", "OFF") < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::loopOff error updating loop status"});
   }
   
   //echo "./aolconfscripts/aollog -e \"$LOOPNAME\" \"LOOP OFF\"" >> $scriptfile
   
   //echo "tmux send-keys -t aol${LOOPNUMBER}-ctr \"aoloff\" C-m" >> $scriptfile
   if(tmuxSendKeys( "aol" + m_loopNumber + "-ctr", "aoloff") < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "cacaoInterface::loopOff error from tmuxSendKeys"});
   }
   
   log<loop_open>();
   
   ///\todo we don't seem to have setupAOloopOff, do we need it?
   //echo "./setupAOloopO" >> $scriptfile
   
   //if [ "$logMode" = "1" ]; then   # log when loop is closed
   //      stop_Telemetrylog_all
   //fi
      
   ///\todo implement telemetry logging from cacao
   /*if(m_logMode == 1)
   {
      //stop_Telemetrylog_all
   }*/
   
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
   
   std::ifstream fin;
   std::string proc, loop;
   
   while(shutdown() == 0)
   {
      fin.open( m_loopDir +  "/status/stat_procON.txt");
      
      if(fin.is_open()) 
      {
         fin >> proc;
      }
      fin.close();
      
      fin.open( m_loopDir +  "/status/stat_loopON.txt");
      
      if(fin.is_open()) 
      {
         fin >> loop;
      }
      fin.close();
      
      fin.open(m_loopDir +  "/conf/param_loopgain.txt");
      
      if(fin.is_open()) 
      {
         fin >> m_gain;
      }
      fin.close();
      
      fin.open(m_loopDir +  "/conf/param_loopmaxlim.txt");
      
      if(fin.is_open()) 
      {
         fin >> m_maxLim;
      }
      fin.close();
      
      fin.open(m_loopDir +  "/conf/param_loopmultcoeff.txt");
      
      if(fin.is_open()) 
      {
         fin >> m_multCoeff;
      }
      fin.close();

      std::unique_lock<std::mutex> lock(m_modeBlockMutex);
      for(size_t n =0; n < m_modeBlockGains.size(); ++n)
      {
         char str[16];
         int nn = n;
         snprintf(str, sizeof(str), "%02d", nn);
         std::string fn = m_loopDir + "conf/param_gainb" + str + ".txt";

         fin.open(fn);
         if( fin )
         {
            fin >> m_modeBlockGains[n];
         }
         fin.close();

         fn = m_loopDir + "conf/param_multfb" + str + ".txt";
         fin.open(fn);
         if( fin )
         {
            fin >> m_modeBlockMCs[n];
         }
         fin.close();

         fn = m_loopDir + "conf/param_limitb" + str + ".txt";
         fin.open(fn);
         if( fin )
         {
            fin >> m_modeBlockLims[n];
         }
         fin.close();
      }
      lock.unlock();

      if(proc[1] == 'F')
      {
         m_loopProcesses_stat = false;
      }
      else
      {
         m_loopProcesses_stat = true;
      }
      
      if(loop[1] == 'F')
      {
         m_loopState = 0; //open
      }
      else if(m_gain == 0)
      {
         m_loopState = 1; //paused
      }
      else m_loopState = 2; //closed
            
      mx::sys::milliSleep(100);

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

int cacaoInterface::st_newCallBack_blockGains( void * app,
                                               const pcf::IndiProperty &ipRecv
                                             )
{
   cacaoInterface * _app = static_cast<cacaoInterface *>(app);
   return _app->newCallBack_blockGains(ipRecv);
}

int cacaoInterface::newCallBack_blockGains( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName().size() < 8) return -1;

   int n = std::stoi(ipRecv.getName().substr(5,2));

   std::cout << "got: " << ipRecv.getName() << " " << n << "\n";

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

   std::cerr << "wants: " << target << "\n";

   updateIfChanged(m_indiP_blockGains[n], "target", target);

   return setBlockGain(n, target);

}

int cacaoInterface::st_newCallBack_blockMCs( void * app,
                                             const pcf::IndiProperty &ipRecv
                                           )
{
   cacaoInterface * _app = static_cast<cacaoInterface *>(app);
   return _app->newCallBack_blockMCs(ipRecv);
}

int cacaoInterface::newCallBack_blockMCs( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName().size() < 8) return -1;

   int n = std::stoi(ipRecv.getName().substr(5,2));

   std::cout << "got: " << ipRecv.getName() << " " << n << "\n";

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

   std::cerr << "wants: " << target << "\n";

   updateIfChanged(m_indiP_blockMCs[n], "target", target);

   return setBlockMC(n, target);

}

int cacaoInterface::st_newCallBack_blockLimits( void * app,
                                              const pcf::IndiProperty &ipRecv
                                           )
{
   cacaoInterface * _app = static_cast<cacaoInterface *>(app);
   return _app->newCallBack_blockLimits(ipRecv);
}

int cacaoInterface::newCallBack_blockLimits( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName().size() < 8) return -1;

   int n = std::stoi(ipRecv.getName().substr(5,2));

   std::cout << "got: " << ipRecv.getName() << " " << n << "\n";

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

   std::cerr << "wants: " << target << "\n";

   updateIfChanged(m_indiP_blockLimits[n], "target", target);

   return setBlockLimit(n, target);

}


} //namespace app
} //namespace MagAOX

#endif //cacaoInterface_hpp
