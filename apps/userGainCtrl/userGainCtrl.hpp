/** \file userGainCtrl.hpp
  * \brief The MagAO-X user gain control app
  *
  * \ingroup app_files
  */

#ifndef userGainCtrl_hpp
#define userGainCtrl_hpp

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

struct gainShmimT 
{
   static std::string configSection()
   {
      return "gainShmim";
   };
   
   static std::string indiPrefix()
   {
      return "gainShmim";
   };
};

struct multcoeffShmimT 
{
   static std::string configSection()
   {
      return "multcoeffShmim";
   };
   
   static std::string indiPrefix()
   {
      return "multcoeffShmim";
   };
};

struct limitShmimT 
{
   static std::string configSection()
   {
      return "limitShmim";
   };
   
   static std::string indiPrefix()
   {
      return "limitShmim";
   };
};

/** \defgroup userGainCtrl User Interface to Cacao Gains
  * \brief Tracks the cacao gain factor vector and updates upon request, using blocks if desired.
  *
  * <a href="../handbook/operating/software/apps/userGainCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup userGainCtrl_files User Gain Control
  * \ingroup userGainCtrl
  */

/** MagAO-X application to provide a user interface to cacao gains
  *
  * \ingroup userGainCtrl
  * 
  */
class userGainCtrl : public MagAOXApp<true>, public dev::shmimMonitor<userGainCtrl,gainShmimT>, 
                     public dev::shmimMonitor<userGainCtrl,multcoeffShmimT>,  public dev::shmimMonitor<userGainCtrl,limitShmimT>,
                     public dev::telemeter<userGainCtrl>
{

   //Give the test harness access.
   friend class userGainCtrl_test;

   friend class dev::shmimMonitor<userGainCtrl,gainShmimT>;
   friend class dev::shmimMonitor<userGainCtrl,multcoeffShmimT>;
   friend class dev::shmimMonitor<userGainCtrl,limitShmimT>;

   typedef dev::telemeter<userGainCtrl> telemeterT;

   friend class dev::telemeter<userGainCtrl>;

public:

   //The base shmimMonitor type
   typedef dev::shmimMonitor<userGainCtrl,gainShmimT> shmimMonitorT;
   typedef dev::shmimMonitor<userGainCtrl,multcoeffShmimT> mcShmimMonitorT;
   typedef dev::shmimMonitor<userGainCtrl,limitShmimT> limitShmimMonitorT;

   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   int m_loopNumber;
   bool m_splitTT {false};

   ///@}
 
   std::string m_aoCalDir;
   std::string m_aoCalArchiveTime;
   std::string m_aoCalLoadTime;

   mx::improc::eigenImage<realT> m_gainsCurrent; ///< The current gains.
   mx::improc::eigenImage<realT> m_gainsTarget; ///< The target gains.
   
   realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
   
   mx::improc::eigenImage<realT> m_mcsCurrent; ///< The current gains.
   mx::improc::eigenImage<realT> m_mcsTarget; ///< The target gains.

   realT (*mc_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.

   mx::improc::eigenImage<realT> m_limitsCurrent; ///< The current gains.
   mx::improc::eigenImage<realT> m_limitsTarget; ///< The target gains.

   realT (*limit_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.

   std::vector<int> m_modeBlockStart;
   std::vector<int> m_modeBlockN;
   
   int m_totalNModes {0}; ///< The total number of WFS modes in the calib.

   std::vector<float> m_modeBlockGains;
   std::vector<uint8_t> m_modeBlockGainsConstant;

   std::vector<float> m_modeBlockMCs;
   std::vector<uint8_t> m_modeBlockMCsConstant;

   std::vector<float> m_modeBlockLims;
   std::vector<uint8_t> m_modeBlockLimsConstant;

   std::mutex m_modeBlockMutex;

   mx::fits::fitsFile<float> m_ff;

   int m_singleModeNo {0};
   
public:
   /// Default c'tor.
   userGainCtrl();

   /// D'tor, declared and defined for noexcept.
   ~userGainCtrl() noexcept
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

   /// Implementation of the FSM for userGainCtrl.
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

   int checkAOCalib(); ///< Test if the AO calib is accessible.

   int getAOCalib();

   int getModeBlocks();

   int allocate( const gainShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const gainShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
   
   int writeGains();

   int setBlockGain( int n,
                     float g
                   );

   int allocate( const multcoeffShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const multcoeffShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );

   int writeMCs();

   int setBlockMC( int n,
                    float mc
                 );

   int allocate( const limitShmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const limitShmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );

   int writeLimits();

   int setBlockLimit( int n,
                      float l
                    );

   int setSingleModeNo (int m);
   int setSingleGain( float g );

   int setSingleMC( float mc );
   
   void updateSingles();

   pcf::IndiProperty m_indiP_modes;

   pcf::IndiProperty m_indiP_zeroAll;
      
   std::vector<pcf::IndiProperty> m_indiP_blockGains;
   std::vector<pcf::IndiProperty> m_indiP_blockMCs;
   std::vector<pcf::IndiProperty> m_indiP_blockLimits;

   pcf::IndiProperty m_indiP_singleModeNo;
   pcf::IndiProperty m_indiP_singleGain;
   pcf::IndiProperty m_indiP_singleMC;

public:

   INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_zeroAll);

   INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_singleModeNo);

   INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_singleGain);
   
   INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_singleMC);

   
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

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_blockgains * );

   int recordBlockGains( bool force = false );
   
   ///@}
};

inline
userGainCtrl::userGainCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{

   shmimMonitorT::m_getExistingFirst = true;
   mcShmimMonitorT::m_getExistingFirst = true;
   limitShmimMonitorT::m_getExistingFirst = true;
   return;
}

inline
void userGainCtrl::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   mcShmimMonitorT::setupConfig(config);
   limitShmimMonitorT::setupConfig(config);

   config.add("loop.number", "", "loop.number", argType::Required, "loop", "number", false, "int", "The loop number");
   config.add("blocks.splitTT", "", "blocks.splitTT", argType::Required, "blocks", "splitTT", false, "bool", "If true, the first block is split into two modes.");

   telemeterT::setupConfig(config);
}

inline
int userGainCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_loopNumber, "loop.number");
   _config(m_splitTT, "blocks.splitTT");
   
   shmimMonitorT::m_shmimName = "aol" + std::to_string(m_loopNumber) + "_mgainfact";   
   shmimMonitorT::loadConfig(config);

   mcShmimMonitorT::m_shmimName = "aol" + std::to_string(m_loopNumber) + "_mmultfact";
   mcShmimMonitorT::loadConfig(config);

   limitShmimMonitorT::m_shmimName = "aol" + std::to_string(m_loopNumber) + "_mlimitfact";
   limitShmimMonitorT::loadConfig(config);

   if(telemeterT::loadConfig(_config) < 0)
   {
      log<text_log>("Error during telemeter config", logPrio::LOG_CRITICAL);
      m_shutdown = true;
   }

   return 0;
}

inline
void userGainCtrl::loadConfig()
{
   loadConfigImpl(config);
}

inline
int userGainCtrl::appStartup()
{
   createROIndiNumber( m_indiP_modes, "modes", "Loop Modes", "Loop Controls");
   indi::addNumberElement(m_indiP_modes, "total", 0, 1, 25000, "Total Modes");
   indi::addNumberElement(m_indiP_modes, "blocks", 0, 1, 99, "Mode Blocks");
   registerIndiPropertyReadOnly(m_indiP_modes);


   createStandardIndiRequestSw( m_indiP_zeroAll, "zero_all");
   if( registerIndiPropertyNew( m_indiP_zeroAll, INDI_NEWCALLBACK(m_indiP_zeroAll)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   createStandardIndiNumber<int>( m_indiP_singleModeNo, "singleModeNo", 0, 2400 ,0, "%0d", "");
   m_indiP_singleModeNo["current"].set(m_singleModeNo);
   m_indiP_singleModeNo["target"].set(m_singleModeNo);
   registerIndiPropertyNew(m_indiP_singleModeNo, INDI_NEWCALLBACK(m_indiP_singleModeNo));
   
   createStandardIndiNumber<int>( m_indiP_singleGain, "singleGain", 0, 1.5 ,0, "%0.2f", "");
   m_indiP_singleGain["current"].set(1);
   m_indiP_singleGain["target"].set(1);
   registerIndiPropertyNew(m_indiP_singleGain, INDI_NEWCALLBACK(m_indiP_singleGain));

   createStandardIndiNumber<int>( m_indiP_singleMC, "singleMC", 0, 1.0 ,0, "%0.2f", "");
   m_indiP_singleMC["current"].set(1);
   m_indiP_singleMC["target"].set(1);
   registerIndiPropertyNew(m_indiP_singleMC, INDI_NEWCALLBACK(m_indiP_singleMC));

   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(mcShmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(limitShmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
  
   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::NODEVICE);
    
   return 0;
}

inline
int userGainCtrl::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( mcShmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( limitShmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if(checkAOCalib() < 0)
   {
      state(stateCodes::NODEVICE);
      if(!stateLogged()) log<text_log>("Could not find AO calib");
   }
   else 
   {
      if(!(state() == stateCodes::READY || state() == stateCodes::OPERATING)) state(stateCodes::NOTCONNECTED);
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING || state() == stateCodes::CONNECTED || state() == stateCodes::NOTCONNECTED  )
   {

      //Now we go on:

      //These could change if a new calibration is loaded
      if(getAOCalib() < 0 )
      {
         state(stateCodes::NOTCONNECTED);
         if(!stateLogged()) log<text_log>("Error getting AO calib", logPrio::LOG_ERROR);
         return 0;
      }

      //These could change if a new calibration is loaded
      if(getModeBlocks() < 0 )
      {
         state(stateCodes::NOTCONNECTED);
         if(!stateLogged()) log<text_log>("Could not get mode blocks", logPrio::LOG_ERROR);
         return 0;
      }

      if(state() == stateCodes::NOTCONNECTED) state(stateCodes::CONNECTED);
      if(state() == stateCodes::CONNECTED) state(stateCodes::READY);
      if(state() == stateCodes::READY) state(stateCodes::OPERATING); //we just progress all the way through to operating so shmimMonitor will go.

   }

   if(state() == stateCodes::OPERATING)
   {
      if(telemeterT::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
   }

   std::unique_lock<std::mutex> lock(m_indiMutex);

   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   if(mcShmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   if(limitShmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
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

   updateSingles();

   return 0;
}

inline
int userGainCtrl::appShutdown()
{
   shmimMonitorT::appShutdown();
   mcShmimMonitorT::appShutdown();
   limitShmimMonitorT::appShutdown();

   telemeterT::appShutdown();

   return 0;
}

inline
int userGainCtrl::checkAOCalib()
{
   std::string calsrc = "/milk/shm/aol" + std::to_string(m_loopNumber) + "_calib_source.txt";

   std::ifstream fin;
   //First read in the milk/shm directory name, which could be to a symlinked directory
   fin.open(calsrc);
   if(!fin)
   {
      return -1;
   }
   fin >> calsrc;
   fin.close();

   //Now read in the actual directory
   calsrc += "/aol" +  std::to_string(m_loopNumber) + "_calib_dir.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return -1;
   }
   fin >> m_aoCalDir;
   fin.close();

   return 0;
}

inline
int userGainCtrl::getAOCalib()
{
   std::string calsrc = "/milk/shm/aol" + std::to_string(m_loopNumber) + "_calib_source.txt";

   std::ifstream fin;
   //First read in the milk/shm directory name, which could be to a symlinked directory
   fin.open(calsrc);
   if(!fin)
   {
      return 0; //this can happen if cacao not started up, etc.
   }
   fin >> calsrc;
   fin.close();

   //Now read in the actual directory
   calsrc += "/aol" +  std::to_string(m_loopNumber) + "_calib_dir.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to open: " + calsrc});
   }
   fin >> m_aoCalDir;
   fin.close();

   calsrc = "/milk/shm/aol" + std::to_string(m_loopNumber) + "_calib_loaded.txt";
   fin.open(calsrc);
   if(!fin)
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to open: " + calsrc});
   }
   fin >> m_aoCalLoadTime;
   fin.close();
   

   calsrc = m_aoCalDir + "/aol" + std::to_string(m_loopNumber) + "_calib_archived.txt";
   
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
   
   int totalNModes = m_gainsCurrent.rows(); //m_ff.naxes(2);

   //Is this a change?
   if(totalNModes != m_totalNModes)
   {
      log<text_log>("Found " + std::to_string(totalNModes) + " total modes in calib.");
   }

   m_totalNModes = totalNModes;

   //Now we check if this matches the shared-memory.  If not, this would mean that the cal changed and the shmims did not
   static bool mmlog = false; //log once per occurrence.  this could go on for a long time until mfilt is started.
   ///\todo this can fail if it never compares as equal but shmimMonitors are connected.  shmimMonitor needs a "connected" flag.
   if(m_totalNModes != (int) shmimMonitorT::m_width || m_totalNModes != (int) mcShmimMonitorT::m_width || m_totalNModes != (int) limitShmimMonitorT::m_width)
   {
      if(!mmlog) 
      {
         log<text_log>("Detected calib and gain shmim mismatch, reconnecting shmim.");
      
         shmimMonitorT::m_restart = true;
         mcShmimMonitorT::m_restart = true;
         limitShmimMonitorT::m_restart = true;
      }
      mmlog = true;
   }   
   else mmlog = false; //reset for next time

   return 0;
}

inline
int userGainCtrl::getModeBlocks()
{

   std::vector<std::string> blocks = mx::ioutils::getFileNames(m_aoCalDir, "aol" + std::to_string(m_loopNumber) + "_block", "_NBmodes", ".txt");

   if(blocks.size() == 0)
   {
      log<text_log>("no mode blocks found", logPrio::LOG_WARNING);
   }

   std::ifstream fin;

   fin.open(m_aoCalDir + "/aol" + std::to_string(m_loopNumber) + "_NBmodeblocks.txt");
   if( !fin )
   {
      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getModeBlocks failed to open: NBmodeblocks.txt"});
   }

   size_t Nb;
   fin >> Nb;

   fin.close();

   std::vector<int> modeBlockStart(Nb);
   std::vector<int> modeBlockN(Nb);

   
   for(size_t n =0; n < Nb; ++n)
   {
      fin.open(blocks[n]);
      if( !fin )
      {
         return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getModeBlocks failed to open: " + blocks[n]});
      }

      int offs, num;

      //fin >> offs;
      fin >> num;

      if(n == 0) offs = 0;
      else offs = modeBlockStart[n-1] + modeBlockN[n-1];


      modeBlockStart[n] = offs;
      modeBlockN[n] = num;
      fin.close();

      if( !fin )
      {
         return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getModeBlocks error closing: " + blocks[n]});
      }
   }

   if(modeBlockStart.back() + modeBlockN.back() < m_totalNModes)
   {
      int st0 = modeBlockStart.back();
      int N0 = modeBlockN.back();

      modeBlockStart.push_back(st0 + N0);
      modeBlockN.push_back(m_totalNModes - (st0+N0));
      ++Nb;

   }

   //split tt here
   if(m_splitTT)
   {
      if(modeBlockN[0] !=2 )
      {
         log<software_warning>({__FILE__, __LINE__, errno, "userGainCtrl::getModeBlocks splitTT is set but block00 does not have 2 modes"});
      }
      else if(modeBlockStart[0] != 0)
      {
         log<software_warning>({__FILE__, __LINE__, errno, "userGainCtrl::getModeBlocks splitTT is set but block00 does not start at mode 0"});
      }
      else
      {
         modeBlockStart.insert(modeBlockStart.begin(), 0);
         modeBlockStart[1] = 1;

         modeBlockN.insert(modeBlockN.begin(), 1);
         modeBlockN[1] = 1;
         ++Nb;
      }
   }

   //now detect changes.
   bool changed = false;
   if(m_modeBlockStart.size() != Nb || m_modeBlockN.size() != Nb || m_modeBlockGains.size() != Nb || m_modeBlockMCs.size() != Nb || m_modeBlockLims.size() != Nb)
   {
      changed = true;
   }
   else //All are correct size, look for differences within:
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
      state(stateCodes::READY);
      log<text_log>("loading new gain block structure");

      shmimMonitorT::m_restart = true;
      mcShmimMonitorT::m_restart = true;
      limitShmimMonitorT::m_restart = true;

      std::unique_lock<std::mutex> lock(m_modeBlockMutex);
      m_modeBlockStart.resize(Nb);
      m_modeBlockN.resize(Nb);
      m_modeBlockGains.resize(Nb);
      m_modeBlockGainsConstant.resize(Nb);
      m_modeBlockMCs.resize(Nb);
      m_modeBlockMCsConstant.resize(Nb);
      m_modeBlockLims.resize(Nb);
      m_modeBlockLimsConstant.resize(Nb);

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

inline
int userGainCtrl::allocate(const gainShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
  
   std::unique_lock<std::mutex> lock(m_indiMutex);
     
   m_gainsCurrent.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   m_gainsTarget.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   
   pixget = getPixPointer<realT>(shmimMonitorT::m_dataType);

   return 0;
}

inline
int userGainCtrl::processImage( void * curr_src, 
                                const gainShmimT & dummy 
                              )
{
   static_cast<void>(dummy); //be unused

   recordBlockGains();

   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   realT * data = m_gainsCurrent.data();
      
   for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
   {
      data[nn] = pixget(curr_src, nn);
   }
   
   //update blocks here.
   std::cerr << "gains updated\n";

   for(int cc =0; cc < m_gainsCurrent.cols(); ++cc)
      for(int rr=0; rr < m_gainsCurrent.rows(); ++rr)
         std::cout << m_gainsCurrent(rr,cc) << " ";
   std::cout << "\n";

   for(size_t n =0; n < m_modeBlockStart.size(); ++n)
   {
      double mng = 0;

      int NN = 0;
      
      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         if(m_modeBlockStart[n] + m >= m_gainsCurrent.rows()) break;
         mng += m_gainsCurrent(m_modeBlockStart[n] + m,0);
         ++NN;
      }

      m_modeBlockGains[n] = mng / NN; 

      bool constant = true;
      
      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         if(m_modeBlockStart[n] + m >= m_gainsCurrent.rows()) break;
         if(m_gainsCurrent(m_modeBlockStart[n] + m,0) != m_modeBlockGains[n])
         {
            constant = false;
            break;
         }
      }

      m_modeBlockGainsConstant[n] = constant;

   }

   for(size_t n=0; n < m_indiP_blockGains.size(); ++n)
   {
      updateIfChanged(m_indiP_blockGains[n], "current", m_modeBlockGains[n]);
   }

   lock.unlock();

   recordBlockGains();
   
   return 0;
}

inline
int userGainCtrl::writeGains()
{
   shmimMonitorT::m_imageStream.md->write=1;
   char * dest = (char *) shmimMonitorT::m_imageStream.array.raw;// + next_cnt1*m_width*m_height*m_typeSize;

   memcpy(dest, m_gainsTarget.data(), shmimMonitorT::m_width*shmimMonitorT::m_height*shmimMonitorT::m_typeSize  ); 

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &shmimMonitorT::m_imageStream.md->writetime);

   //Set the image acquisition timestamp
   shmimMonitorT::m_imageStream.md->atime = shmimMonitorT::m_imageStream.md->writetime;
         
   //Update cnt1
   //m_imageStream.md->cnt1 = next_cnt1;
          
   //Update cnt0
   shmimMonitorT::m_imageStream.md->cnt0++;
         
   //And post
   shmimMonitorT::m_imageStream.md->write=0;
   ImageStreamIO_sempost(&(shmimMonitorT::m_imageStream),-1);

   return 0;
}

int userGainCtrl::setBlockGain( int n,
                                float g
                              )
{
   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   m_gainsTarget = m_gainsCurrent;

   //Apply a delta to each mode in the block
   //to preserve intra-block differences
   for(int m =0; m < m_modeBlockN[n]; ++m)
   {
      if(m_modeBlockStart[n] + m > m_gainsTarget.rows() -1) break;
      m_gainsTarget(m_modeBlockStart[n] + m,0) = m_gainsCurrent(m_modeBlockStart[n] + m,0) + (g - m_modeBlockGains[n]);
   }
   lock.unlock();
   recordBlockGains(true);
   writeGains();
   return 0;
}

inline
int userGainCtrl::allocate(const multcoeffShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
  
   std::unique_lock<std::mutex> lock(m_indiMutex);
     
   m_mcsCurrent.resize(mcShmimMonitorT::m_width, mcShmimMonitorT::m_height);
   m_mcsTarget.resize(mcShmimMonitorT::m_width, mcShmimMonitorT::m_height);
   
   mc_pixget = getPixPointer<realT>(mcShmimMonitorT::m_dataType);

   return 0;
}

inline
int userGainCtrl::processImage( void * curr_src, 
                                const multcoeffShmimT & dummy 
                              )
{
   static_cast<void>(dummy); //be unused

   recordBlockGains();

   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   realT * data = m_mcsCurrent.data();
      
   for(unsigned nn=0; nn < mcShmimMonitorT::m_width*mcShmimMonitorT::m_height; ++nn)
   {
      data[nn] = mc_pixget(curr_src, nn);
   }
   
   //update blocks here.
   std::cerr << "multcoeff updated\n";

   for(int cc =0; cc < m_mcsCurrent.cols(); ++cc)
      for(int rr=0; rr < m_mcsCurrent.rows(); ++rr)
         std::cout << m_mcsCurrent(rr,cc) << " ";
   std::cout << "\n";


   for(size_t n =0; n < m_modeBlockStart.size(); ++n)
   {
      double mng = 0;

      int NN = 0;
      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         if(m_modeBlockStart[n] + m >= m_mcsCurrent.rows()) break;
         mng += m_mcsCurrent(m_modeBlockStart[n] + m,0);
         ++NN;
      }

      m_modeBlockMCs[n] = mng / NN; 
   

      bool constant = true;
      
      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         if(m_modeBlockStart[n] + m >= m_mcsCurrent.rows()) break;
         if(m_mcsCurrent(m_modeBlockStart[n] + m,0) != m_modeBlockMCs[n])
         {
            constant = false;
            break;
         }
      }

      m_modeBlockMCsConstant[n] = constant;
   }

   for(size_t n=0; n < m_indiP_blockMCs.size(); ++n)
   {
      updateIfChanged(m_indiP_blockMCs[n], "current", m_modeBlockMCs[n]);
   }

   lock.unlock();

   recordBlockGains();

   return 0;
}

inline
int userGainCtrl::writeMCs()
{
   mcShmimMonitorT::m_imageStream.md->write=1;
   char * dest = (char *) mcShmimMonitorT::m_imageStream.array.raw;// + next_cnt1*m_width*m_height*m_typeSize;

   memcpy(dest, m_mcsTarget.data(), mcShmimMonitorT::m_width*mcShmimMonitorT::m_height*mcShmimMonitorT::m_typeSize  ); 

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &mcShmimMonitorT::m_imageStream.md->writetime);

   //Set the image acquisition timestamp
   mcShmimMonitorT::m_imageStream.md->atime = mcShmimMonitorT::m_imageStream.md->writetime;
         
   //Update cnt1
   //mcShmimMonitorT::m_imageStream.md->cnt1 = next_cnt1;
          
   //Update cnt0
   mcShmimMonitorT::m_imageStream.md->cnt0++;
         
   //And post
   mcShmimMonitorT::m_imageStream.md->write=0;
   ImageStreamIO_sempost(&(mcShmimMonitorT::m_imageStream),-1);

   return 0;
}

int userGainCtrl::setBlockMC( int n,
                              float mc
                            )
{
   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   m_mcsTarget = m_mcsCurrent;

   //Apply a delta to each mode in the block
   //to preserve intra-block differences
   for(int m =0; m < m_modeBlockN[n]; ++m)
   {
      if(m_modeBlockStart[n] + m > m_mcsTarget.rows() -1) break;
      m_mcsTarget(m_modeBlockStart[n] + m,0) = m_mcsCurrent(m_modeBlockStart[n] + m,0) + (mc- m_modeBlockMCs[n]);
   }
   lock.unlock();
   recordBlockGains(true);
   writeMCs();
   return 0;
}

inline
int userGainCtrl::allocate(const limitShmimT & dummy)
{
   static_cast<void>(dummy); //be unused
  
   std::unique_lock<std::mutex> lock(m_indiMutex);
     
   m_limitsCurrent.resize(limitShmimMonitorT::m_width, limitShmimMonitorT::m_height);
   m_limitsTarget.resize(limitShmimMonitorT::m_width, limitShmimMonitorT::m_height);
   
   limit_pixget = getPixPointer<realT>(limitShmimMonitorT::m_dataType);

   return 0;
}

inline
int userGainCtrl::processImage( void * curr_src, 
                                const limitShmimT & dummy 
                              )
{
   static_cast<void>(dummy); //be unused

   recordBlockGains();

   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   realT * data = m_limitsCurrent.data();
      
   for(unsigned nn=0; nn < limitShmimMonitorT::m_width*limitShmimMonitorT::m_height; ++nn)
   {
      data[nn] = limit_pixget(curr_src, nn);
   }
   
   //update blocks here.
   std::cerr << "limits updated\n";

   for(int cc =0; cc < m_limitsCurrent.cols(); ++cc)
      for(int rr=0; rr < m_limitsCurrent.rows(); ++rr)
         std::cout << m_limitsCurrent(rr,cc) << " ";
   std::cout << "\n";


   for(size_t n =0; n < m_modeBlockStart.size(); ++n)
   {
      double mng = 0;

      int NN = 0;
      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         if(m_modeBlockStart[n] + m >= m_limitsCurrent.rows()) break;
         mng += m_limitsCurrent(m_modeBlockStart[n] + m,0);
         ++NN;
      }

      m_modeBlockLims[n] = mng / NN; 

      bool constant = true;
      
      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         if(m_modeBlockStart[n] + m >= m_limitsCurrent.rows()) break;
         if(m_limitsCurrent(m_modeBlockStart[n] + m,0) != m_modeBlockLims[n])
         {
            constant = false;
            break;
         }
      }

      m_modeBlockLimsConstant[n] = constant;
   }

   for(size_t n=0; n < m_indiP_blockLimits.size(); ++n)
   {
      updateIfChanged(m_indiP_blockLimits[n], "current", m_modeBlockLims[n]);
   }

   lock.unlock();

   recordBlockGains();

   return 0;
}

inline
int userGainCtrl::writeLimits()
{
   limitShmimMonitorT::m_imageStream.md->write=1;
   char * dest = (char *) limitShmimMonitorT::m_imageStream.array.raw;// + next_cnt1*m_width*m_height*m_typeSize;

   memcpy(dest, m_limitsTarget.data(), limitShmimMonitorT::m_width*limitShmimMonitorT::m_height*limitShmimMonitorT::m_typeSize  ); 

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &limitShmimMonitorT::m_imageStream.md->writetime);

   //Set the image acquisition timestamp
   limitShmimMonitorT::m_imageStream.md->atime = limitShmimMonitorT::m_imageStream.md->writetime;
         
   //Update cnt1
   //mcShmimMonitorT::m_imageStream.md->cnt1 = next_cnt1;
          
   //Update cnt0
   limitShmimMonitorT::m_imageStream.md->cnt0++;
         
   //And post
   limitShmimMonitorT::m_imageStream.md->write=0;
   ImageStreamIO_sempost(&(limitShmimMonitorT::m_imageStream),-1);

   return 0;
}

int userGainCtrl::setBlockLimit( int n,
                                 float l
                               )
{
   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   m_limitsTarget = m_limitsCurrent;

   //Apply a delta to each mode in the block
   //to preserve intra-block differences
   for(int m =0; m < m_modeBlockN[n]; ++m)
   {
      if(m_modeBlockStart[n] + m > m_limitsTarget.rows() -1) break;
      m_limitsTarget(m_modeBlockStart[n] + m,0) = m_limitsCurrent(m_modeBlockStart[n] + m,0) + (l- m_modeBlockLims[n]);
   }
   lock.unlock();
   recordBlockGains(true);
   writeLimits();

   return 0;
}

int userGainCtrl::setSingleModeNo ( int m )
{
   m_singleModeNo = m;

   updateIfChanged(m_indiP_singleModeNo, "current", m);

   if(m_singleModeNo < 0 || m_singleModeNo >= m_gainsCurrent.rows()) return -1;
   float g =  m_gainsCurrent (m_singleModeNo,0);

   updateIfChanged(m_indiP_singleGain, std::vector<std::string>({"current", "target"}), std::vector<float>({g,g}));

   if(m_singleModeNo < 0 || m_singleModeNo >= m_mcsCurrent.rows()) return -1;
   float mc =  m_mcsCurrent(m_singleModeNo,0);

   updateIfChanged(m_indiP_singleMC, std::vector<std::string>({"current", "target"}), std::vector<float>({mc,mc}));

   return 0;
}

int userGainCtrl::setSingleGain( float g )
{
   if(m_singleModeNo < 0 || m_singleModeNo >= m_gainsCurrent.rows()) return -1;
   recordBlockGains();
   std::unique_lock<std::mutex> lock(m_modeBlockMutex);
   m_gainsTarget(m_singleModeNo,0) = g;
   lock.unlock();
   recordBlockGains(true);
   writeGains();
   return 0;
}

int userGainCtrl::setSingleMC( float mc )
{
   if(m_singleModeNo < 0 || m_singleModeNo >= m_mcsCurrent.rows()) return -1;
   recordBlockGains();
   std::unique_lock<std::mutex> lock(m_modeBlockMutex);
   m_mcsTarget(m_singleModeNo,0) = mc;
   lock.unlock();
   recordBlockGains(true);
   writeMCs();
   return 0;
}

void userGainCtrl::updateSingles()
{
   if(m_singleModeNo < 0 || m_singleModeNo >= m_gainsCurrent.rows()) return;
   float g =  m_gainsCurrent (m_singleModeNo,0);

   updateIfChanged(m_indiP_singleGain, std::vector<std::string>({"current", "target"}), std::vector<float>({g,g}));

   if(m_singleModeNo < 0 || m_singleModeNo >= m_mcsCurrent.rows()) return;
   float mc =  m_mcsCurrent(m_singleModeNo,0);

   updateIfChanged(m_indiP_singleMC, std::vector<std::string>({"current", "target"}), std::vector<float>({mc,mc}));

}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_zeroAll)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_zeroAll, ipRecv);

   if(!ipRecv.find("request")) return 0;
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      std::unique_lock<std::mutex> lock(m_indiMutex);

      std::cerr << "Got zero all\n";
      m_gainsTarget.setZero();
      writeGains();
      
      updateSwitchIfChanged(m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE);
   }
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_singleModeNo)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_singleModeNo, ipRecv);
   
   int target;
   
   if( indiTargetUpdate( m_indiP_singleModeNo, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   setSingleModeNo(target);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_singleGain)(const pcf::IndiProperty &ipRecv)
{
   INDI_VALIDATE_CALLBACK_PROPS(m_indiP_singleGain, ipRecv);

   float target;
   
   if( indiTargetUpdate( m_indiP_singleGain, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   setSingleGain(target);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_singleMC)(const pcf::IndiProperty &ipRecv)
{
   INDI_VALIDATE_CALLBACK_PROPS(m_indiP_singleMC, ipRecv);

   float target;
   
   if( indiTargetUpdate( m_indiP_singleMC, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   setSingleMC(target);
   
   return 0;
}


int userGainCtrl::st_newCallBack_blockGains( void * app,
                                               const pcf::IndiProperty &ipRecv
                                             )
{
   userGainCtrl * _app = static_cast<userGainCtrl *>(app);
   return _app->newCallBack_blockGains(ipRecv);
}

int userGainCtrl::newCallBack_blockGains( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getDevice() != m_configName)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI device"});
      #endif

      return -1;
   }

   if(ipRecv.getName().find("block") != 0)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   if(ipRecv.getName().find("_gain") != 7)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   if(ipRecv.getName().size() != 12)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
   return 0;
   #endif

   int n = std::stoi(ipRecv.getName().substr(5,2));

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

   updateIfChanged(m_indiP_blockGains[n], "target", target);

   return setBlockGain(n, target);

}

int userGainCtrl::st_newCallBack_blockMCs( void * app,
                                             const pcf::IndiProperty &ipRecv
                                           )
{
   userGainCtrl * _app = static_cast<userGainCtrl *>(app);
   return _app->newCallBack_blockMCs(ipRecv);
}

int userGainCtrl::newCallBack_blockMCs( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getDevice() != m_configName)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI device"});
      #endif

      return -1;
   }

   if(ipRecv.getName().find("block") != 0)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   if(ipRecv.getName().find("_multcoeff") != 7)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   if(ipRecv.getName().size() != 17)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
   return 0;
   #endif

   int n = std::stoi(ipRecv.getName().substr(5,2));

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

   updateIfChanged(m_indiP_blockMCs[n], "target", target);

   return setBlockMC(n, target);

}

int userGainCtrl::st_newCallBack_blockLimits( void * app,
                                              const pcf::IndiProperty &ipRecv
                                           )
{
   userGainCtrl * _app = static_cast<userGainCtrl *>(app);
   return _app->newCallBack_blockLimits(ipRecv);
}

int userGainCtrl::newCallBack_blockLimits( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getDevice() != m_configName)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI device"});
      #endif

      return -1;
   }

   if(ipRecv.getName().find("block") != 0)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   if(ipRecv.getName().find("_limit") != 7)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

    if(ipRecv.getName().size() != 13)
   {
      #ifndef XWCTEST_INDI_CALLBACK_VALIDATION
      log<software_error>({__FILE__, __LINE__, "wrong INDI property"});
      #endif

      return -1;
   }

   #ifdef XWCTEST_INDI_CALLBACK_VALIDATION
   return 0;
   #endif


   int n = std::stoi(ipRecv.getName().substr(5,2));

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

   updateIfChanged(m_indiP_blockLimits[n], "target", target);

   return setBlockLimit(n, target);

}

inline
int userGainCtrl::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_blockgains());
}

inline
int userGainCtrl::recordTelem( const telem_blockgains * )
{
   return recordBlockGains(true);
}

inline
int userGainCtrl::recordBlockGains( bool force )
{
   static std::vector<float> modeBlockGains;
   static std::vector<uint8_t> modeBlockGainsConstant;

   static std::vector<float> modeBlockMCs;
   static std::vector<uint8_t> modeBlockMCsConstant;

   static std::vector<float> modeBlockLims;
   static std::vector<uint8_t> modeBlockLimsConstant;

   if(!force)
   {
      if(!(m_modeBlockGains == modeBlockGains)) force = true;
   }

   if(!force)
   {
      if(!(m_modeBlockGainsConstant == modeBlockGainsConstant)) force = true;
   }

   if(!force)
   {
      if(!(m_modeBlockMCs == modeBlockMCs)) force = true;
   }

   if(!force)
   {
      if(!(m_modeBlockMCsConstant == modeBlockMCsConstant)) force = true;
   }

   if(!force)
   {
      if(!(m_modeBlockLims == modeBlockLims)) force = true;
   }

   if(!force)
   {
      if(!(m_modeBlockLimsConstant == modeBlockLimsConstant)) force = true;
   }

   if(force)
   {
      telem<telem_blockgains>({m_modeBlockGains, m_modeBlockGainsConstant, m_modeBlockMCs, m_modeBlockMCsConstant, m_modeBlockLims, m_modeBlockLimsConstant});
      modeBlockGains = m_modeBlockGains;
      modeBlockGainsConstant = m_modeBlockGainsConstant;
      modeBlockMCs = m_modeBlockMCs;
      modeBlockMCsConstant = m_modeBlockMCsConstant;
      modeBlockLims = m_modeBlockLims;
      modeBlockLimsConstant = m_modeBlockLimsConstant;
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //userGainCtrl_hpp
