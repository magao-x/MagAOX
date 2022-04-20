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
                     public dev::shmimMonitor<userGainCtrl,multcoeffShmimT>,  public dev::shmimMonitor<userGainCtrl,limitShmimT>
{

   //Give the test harness access.
   friend class userGainCtrl_test;

   friend class dev::shmimMonitor<userGainCtrl,gainShmimT>;
   friend class dev::shmimMonitor<userGainCtrl,multcoeffShmimT>;
   friend class dev::shmimMonitor<userGainCtrl,limitShmimT>;

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

   std::vector<float> m_modeBlockGains;
   std::vector<float> m_modeBlockMCs;
   std::vector<float> m_modeBlockLims;

   std::mutex m_modeBlockMutex;

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

   pcf::IndiProperty m_indiP_modes;

   pcf::IndiProperty m_indiP_zeroAll;
      
   std::vector<pcf::IndiProperty> m_indiP_blockGains;
   std::vector<pcf::IndiProperty> m_indiP_blockMCs;
   std::vector<pcf::IndiProperty> m_indiP_blockLimits;

   INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_zeroAll);

   
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
   
}

inline
int userGainCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_loopNumber, "loop.number");

   shmimMonitorT::m_shmimName = "aol" + std::to_string(m_loopNumber) + "_mgainfact";   
   shmimMonitorT::loadConfig(config);

   mcShmimMonitorT::m_shmimName = "aol" + std::to_string(m_loopNumber) + "_mmultfact";
   mcShmimMonitorT::loadConfig(config);

   limitShmimMonitorT::m_shmimName = "aol" + std::to_string(m_loopNumber) + "_mlimitfact";
   limitShmimMonitorT::loadConfig(config);


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
  
   state(stateCodes::OPERATING);
    
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

   //These could change if a new calibration is loaded
   if(getAOCalib() < 0 )
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get AO calib", logPrio::LOG_ERROR);
      return 0;
   }

   //These could change if a new calibration is loaded
   if(getModeBlocks() < 0 )
   {
      state(stateCodes::ERROR, true);
      if(!stateLogged()) log<text_log>("Could not get mode blocks", logPrio::LOG_ERROR);
      return 0;
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

   return 0;
}

inline
int userGainCtrl::appShutdown()
{
   shmimMonitorT::appShutdown();
   mcShmimMonitorT::appShutdown();
   limitShmimMonitorT::appShutdown();

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
      return 0; //this can hapapend if cacao not started up, etc.
//      return log<software_error, -1>({__FILE__, __LINE__, errno, "userGainCtrl::getAOCalib failed to open: " + calsrc});
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

   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   for(size_t n =0; n < m_modeBlockStart.size(); ++n)
   {
      double mng = 0;

      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         mng += m_gainsCurrent(m_modeBlockStart[n] + m,0);
      }

      m_modeBlockGains[n] = mng / m_modeBlockN[n]; 
   }

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
      m_gainsTarget(m_modeBlockStart[n] + m,0) = m_gainsCurrent(m_modeBlockStart[n] + m,0) + (g - m_modeBlockGains[n]);
   }
   lock.unlock();
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

   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   for(size_t n =0; n < m_modeBlockStart.size(); ++n)
   {
      double mng = 0;

      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         mng += m_mcsCurrent(m_modeBlockStart[n] + m,0);
      }

      m_modeBlockMCs[n] = mng / m_modeBlockN[n]; 
   }

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
      m_mcsTarget(m_modeBlockStart[n] + m,0) = m_mcsCurrent(m_modeBlockStart[n] + m,0) + (mc- m_modeBlockMCs[n]);
   }
   lock.unlock();
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

   std::unique_lock<std::mutex> lock(m_modeBlockMutex);

   for(size_t n =0; n < m_modeBlockStart.size(); ++n)
   {
      double mng = 0;

      for(int m =0; m < m_modeBlockN[n]; ++m)
      {
         mng += m_limitsCurrent(m_modeBlockStart[n] + m,0);
      }

      m_modeBlockLims[n] = mng / m_modeBlockN[n]; 
   }

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
      m_limitsTarget(m_modeBlockStart[n] + m,0) = m_limitsCurrent(m_modeBlockStart[n] + m,0) + (l- m_modeBlockLims[n]);
   }
   lock.unlock();
   writeLimits();
   return 0;
}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_zeroAll)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_zeroAll.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
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
int userGainCtrl::st_newCallBack_blockGains( void * app,
                                               const pcf::IndiProperty &ipRecv
                                             )
{
   userGainCtrl * _app = static_cast<userGainCtrl *>(app);
   return _app->newCallBack_blockGains(ipRecv);
}

int userGainCtrl::newCallBack_blockGains( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName().size() < 8) return -1;

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
   if(ipRecv.getName().size() < 8) return -1;

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
   if(ipRecv.getName().size() < 8) return -1;

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

} //namespace app
} //namespace MagAOX

#endif //userGainCtrl_hpp
