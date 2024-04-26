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

uint16_t modesAtBlock( uint16_t b )
{
    int16_t N = 2* (2*b+1 + 1);

    return (N+1)*(N+1) - 1;
}

/// Calculate the number of modes in 1 block
/** A block is 2 Fourier mode indices wide.  At index m, there are 2m linear degrees of freedom.
  * This gives [(2m+1)(2m+1)-1] total Fourier modes.  By considering the difference for 2m and 2(m-1) we 
  * find the number of modes in one index is 16m + 8.  Note that m here starts from 1.  
  * 
  * Block number b starts from 0, and is related to m by m = 2b + 1 
  */
uint16_t modesInBlock( uint16_t b /**< [in] the block number */)
{
    return 32*b + 24;
}

/// Calculate the number of blocks and the number of modes per block
/** A block is 2 Fourier mode m-indices wide, going around to the m < 0 side.  At index m, there are 2m linear degrees of freedom.
  * Block number b starts from 0, and is related to m by m = 2b + 1.  So for b+1 blocks, there are N = 2* (2*b+1 + 1) linear
  * degrees of freedom, giving (N+1)*(N+1) - 1 total Fourier modes, with 32*b + 24 modes per block b.
  * 
  * Complicating this is the usual practice of putting pure Zernike modes into the beginning of the basis.  This accounts for 
  * this if desired, always splitting Tip/Tilt and Focus into separate blocks.  Tip/Tilt can optionally be 2 separate blocks.
  */
int blockModes( std::vector<uint16_t> & blocks,   ///< [out] the block structure.  The size is the number of blocks, and each entry contains the nubmer of modes in that block
                std::vector<std::string> & names, ///< [out] the name of each block
                uint16_t Nmodes,                  ///< [in] the total number of modes
                uint16_t Nzern,                   ///< [in] the number of Zernikes appended at the front
                bool splitTT                      ///< [in] whether or not to split tip and tilt
              )
{
    double Nblocksd = (sqrt(1.0+Nmodes) - 1)/4.;
    int Nblocks = Nblocksd;

    if(Nblocks < Nblocksd) 
    {
        ++Nblocks;
    }

    blocks.clear();
    names.clear();

    uint16_t tot = 0;
    if(Nzern > 0)
    {
        if(Nzern < 2) //not enough modes for this
        {
            //This is dumb, whomever is doing this, you should know.
            Nblocks += 1;
            blocks.push_back(1);
            names.push_back("Tip");
            tot = 1;
        }
        else if(splitTT)
        {
            Nblocks += 2;
            blocks.push_back(1);
            names.push_back("Tip");
            blocks.push_back(1);
            names.push_back("Tilt");
            tot = 2;
        }
        else
        {
            Nblocks += 1;
            blocks.push_back(2);
            names.push_back("Tip/Tilt");
            tot = 2;
        }

        if(Nzern > 2)
        {
            //Focus
            Nblocks += 1;
            blocks.push_back(1);
            names.push_back("Focus");
            ++tot;

            if(Nzern > 3)
            {
                Nblocks += 1;
                blocks.push_back(Nzern - 3);
                names.push_back("Z " + std::to_string(4)+"-" + std::to_string(Nzern));
                tot += blocks.back();
            }
        }
    }

    if(tot >= Nmodes) //Here we handle the case of Nzern >= Nmodes.
    {
        uint16_t sum = 0;
        for(size_t n=0; n < blocks.size(); ++n)
        {
            sum += blocks[n];
        }

        if(sum != Nmodes && sum != Nzern)
        {
            return -4;
        }

        return 0;
    }

    uint16_t currb = 0;

    while(currb < Nblocks)
    {
        uint16_t NAtThis = modesAtBlock(currb);

        if(NAtThis <= tot) //Still in the Zernikes at the beginning
        {
            //--Nblocks;
            ++currb;
            continue;
        }

        uint16_t Nthis = NAtThis - tot;

        if(tot + Nthis > Nmodes)
        {
            Nthis = Nmodes - tot;
        }

        if(Nthis == 0)
        {
            break;
        }

        blocks.push_back(Nthis);
        names.push_back("Block " + std::to_string(blocks.size()-1));
        tot += Nthis;
        ++currb;
    }

    uint16_t sum = 0;
    for(size_t n=0; n < blocks.size(); ++n)
    {
        sum += blocks[n];
    }

    if(sum != Nmodes)
    {
        return -3;
    }

    return 0;

}

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
  * \brief Tracks the cacao gain factor vector and updates upon request, using blocks.
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
    int m_loopNumber {-1};
    int m_nZern {0};
    bool m_splitTT {false};
 
    ///@}
  
    mx::improc::eigenImage<realT> m_gainsCurrent; ///< The current gains.
    mx::improc::eigenImage<realT> m_gainsTarget; ///< The target gains.
    
    realT (*pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
    
    mx::improc::eigenImage<realT> m_mcsCurrent; ///< The current gains.
    mx::improc::eigenImage<realT> m_mcsTarget; ///< The target gains.
 
    realT (*mc_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
 
    mx::improc::eigenImage<realT> m_limitsCurrent; ///< The current gains.
    mx::improc::eigenImage<realT> m_limitsTarget; ///< The target gains.
 
    realT (*limit_pixget)(void *, size_t) {nullptr}; ///< Pointer to a function to extract the image data as our desired type realT.
 
    std::vector<uint16_t> m_modeBlockStart;
    std::vector<uint16_t> m_modeBlockN;
    std::vector<std::string> m_modeBlockNames;
    
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
    
    float m_powerLawIndex {2};
    float m_powerLawFloor {0.05};

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

    //int checkAOCalib(); ///< Test if the AO calib is accessible.

    //int getAOCalib();
 
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
 
    void powerLawIndex( float pli );
 
    void powerLawFloor( float plf );
 
    void powerLawSet();
 
    pcf::IndiProperty m_indiP_modes;
 
    pcf::IndiProperty m_indiP_zeroAll;
       
    std::vector<pcf::IndiProperty> m_indiP_blockGains;
    std::vector<pcf::IndiProperty> m_indiP_blockMCs;
    std::vector<pcf::IndiProperty> m_indiP_blockLimits;
 
    pcf::IndiProperty m_indiP_singleModeNo;
    pcf::IndiProperty m_indiP_singleGain;
    pcf::IndiProperty m_indiP_singleMC;
 
    pcf::IndiProperty m_indiP_powerLawIndex;
    pcf::IndiProperty m_indiP_powerLawFloor;
    pcf::IndiProperty m_indiP_powerLawSet;

public:

    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_zeroAll);
 
    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_singleModeNo);
 
    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_singleGain);
    
    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_singleMC);
 
    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_powerLawIndex);
    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_powerLawFloor);
    INDI_NEWCALLBACK_DECL(userGainCtrl, m_indiP_powerLawSet);
 
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
   config.add("blocks.nZern", "", "blocks.nZern", argType::Required, "blocks", "nZern", false, "int", "Number of Zernikes at beginning.  T/T and F are split, the rest in their own block.");

   telemeterT::setupConfig(config);
}

inline
int userGainCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   _config(m_loopNumber, "loop.number");
   _config(m_splitTT, "blocks.splitTT");
   _config(m_nZern, "blocks.nZern");
   
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

   createStandardIndiNumber<int>( m_indiP_powerLawIndex, "pwrlaw_index", 0, 10.0 ,0, "%0.2f", "");
   m_indiP_powerLawIndex["current"].set(m_powerLawIndex);
   m_indiP_powerLawIndex["target"].set(m_powerLawIndex);
   registerIndiPropertyNew(m_indiP_powerLawIndex, INDI_NEWCALLBACK(m_indiP_powerLawIndex));

   createStandardIndiNumber<int>( m_indiP_powerLawFloor, "pwrlaw_floor", 0, 1.0 ,0, "%0.2f", "");
   m_indiP_powerLawFloor["current"].set(m_powerLawFloor);
   m_indiP_powerLawFloor["target"].set(m_powerLawFloor);
   registerIndiPropertyNew(m_indiP_powerLawFloor, INDI_NEWCALLBACK(m_indiP_powerLawFloor));

   createStandardIndiRequestSw( m_indiP_powerLawSet, "pwrlaw_set");
   if( registerIndiPropertyNew( m_indiP_powerLawSet, INDI_NEWCALLBACK(m_indiP_powerLawSet)) < 0)
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
  
   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::CONNECTED);
    
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

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING 
              || state() == stateCodes::CONNECTED || state() == stateCodes::NOTCONNECTED  )
   {
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
int userGainCtrl::getModeBlocks()
{    
    blockModes(m_modeBlockN, m_modeBlockNames, shmimMonitorT::m_width, m_nZern, m_splitTT);

    uint16_t Nb = m_modeBlockN.size();

    m_modeBlockStart.resize(m_modeBlockN.size());
    m_modeBlockStart[0] = 0;
    for(size_t n = 1; n < m_modeBlockN.size(); ++n)
    {
        m_modeBlockStart[n] = m_modeBlockStart[n-1] + m_modeBlockN[n-1];
    }
    
    log<text_log>("loading new gain block structure");

    m_modeBlockGains.resize(Nb);
    m_modeBlockGainsConstant.resize(Nb);
      
    m_modeBlockMCs.resize(Nb);
    m_modeBlockMCsConstant.resize(Nb);
      
    m_modeBlockLims.resize(Nb);
    m_modeBlockLimsConstant.resize(Nb);

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
            if(!m_indiNewCallBacks.erase(m_indiP_blockGains[n].createUniqueKey()))
            {
                log<software_error>({__FILE__, __LINE__, "failed to erase " + m_indiP_blockGains[n].createUniqueKey()});
            }
       }
    }
    m_indiP_blockGains.clear(); 

    //Erase existing block mult. coeffs
    if(m_indiP_blockMCs.size() > 0)
    {
        for(size_t n=0; n < m_indiP_blockMCs.size(); ++n)
        {
            if(m_indiDriver) m_indiDriver->sendDelProperty(m_indiP_blockMCs[n]);
            if(!m_indiNewCallBacks.erase(m_indiP_blockMCs[n].createUniqueKey()))
            {
                log<software_error>({__FILE__, __LINE__, "failed to erase " + m_indiP_blockMCs[n].createUniqueKey()});
            }
       }
    }
    m_indiP_blockMCs.clear(); 

    //Erase existing block limits
    if(m_indiP_blockLimits.size() > 0)
    {
        for(size_t n=0; n < m_indiP_blockLimits.size(); ++n)
        {
            if(m_indiDriver) m_indiDriver->sendDelProperty(m_indiP_blockLimits[n]);
            if(!m_indiNewCallBacks.erase(m_indiP_blockLimits[n].createUniqueKey()))
            {
                log<software_error>({__FILE__, __LINE__, "failed to erase " + m_indiP_blockLimits[n].createUniqueKey()});
            }
        }
    }
    m_indiP_blockLimits.clear(); 

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

        createStandardIndiNumber<float>( m_indiP_blockGains[n], en + "_gain", 0.0, 10.0, 0.01, "%0.3f", m_modeBlockNames[n] + " Gain", "Loop Controls");
        registerIndiPropertyNew( m_indiP_blockGains[n],  st_newCallBack_blockGains);  
        if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_blockGains[n]);

        createStandardIndiNumber<float>( m_indiP_blockMCs[n], en + "_multcoeff", 0.0, 1.0, 0.01, "%0.3f", m_modeBlockNames[n] + " Mult. Coeff", "Loop Controls");
        registerIndiPropertyNew( m_indiP_blockMCs[n],  st_newCallBack_blockMCs);  
        if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_blockMCs[n]);

        createStandardIndiNumber<float>( m_indiP_blockLimits[n], en + "_limit", 0.0, 100.0, 0.01, "%0.3f", m_modeBlockNames[n] + " Limit", "Loop Controls");
        registerIndiPropertyNew( m_indiP_blockLimits[n],  st_newCallBack_blockLimits);  
        if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_blockLimits[n]);
    }

    if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_modes); //might not exist yet!   

    return 0;
}

inline
int userGainCtrl::allocate(const gainShmimT & dummy)
{
    static_cast<void>(dummy); //be unused
  
    std::unique_lock<std::mutex> lock(m_modeBlockMutex);

    m_gainsCurrent.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
    m_gainsTarget.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   
    getModeBlocks();

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
   char * dest = static_cast<char *>(shmimMonitorT::m_imageStream.array.raw);

   memcpy(dest, m_gainsTarget.data(), shmimMonitorT::m_width*shmimMonitorT::m_height*shmimMonitorT::m_typeSize  ); 

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &shmimMonitorT::m_imageStream.md->writetime);

   //Set the image acquisition timestamp
   shmimMonitorT::m_imageStream.md->atime = shmimMonitorT::m_imageStream.md->writetime;
                   
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
   //lock.unlock();
   recordBlockGains(true);
   writeGains();
   return 0;
}

inline
int userGainCtrl::allocate(const multcoeffShmimT & dummy)
{
    static_cast<void>(dummy); //be unused
   
    int n = 0;

    while( mcShmimMonitorT::m_width != shmimMonitorT::m_width && n < 100 )
    {
        mx::sys::milliSleep(100);
        ++n;
    }

    if( mcShmimMonitorT::m_width != shmimMonitorT::m_width )
    {
        return -1;
    }

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
   char * dest = static_cast<char *>(mcShmimMonitorT::m_imageStream.array.raw);

   memcpy(dest, m_mcsTarget.data(), mcShmimMonitorT::m_width*mcShmimMonitorT::m_height*mcShmimMonitorT::m_typeSize  ); 

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &mcShmimMonitorT::m_imageStream.md->writetime);

   //Set the image acquisition timestamp
   mcShmimMonitorT::m_imageStream.md->atime = mcShmimMonitorT::m_imageStream.md->writetime;
                   
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
  
    int n = 0;

    while( limitShmimMonitorT::m_width != shmimMonitorT::m_width && n < 100 )
    {
        mx::sys::milliSleep(100);
        ++n;
    }

    if( limitShmimMonitorT::m_width != shmimMonitorT::m_width )
    {
        return -1;
    }

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
   char * dest = static_cast<char *>(limitShmimMonitorT::m_imageStream.array.raw);// + next_cnt1*m_width*m_height*m_typeSize;

   memcpy(dest, m_limitsTarget.data(), limitShmimMonitorT::m_width*limitShmimMonitorT::m_height*limitShmimMonitorT::m_typeSize  ); 

   //Set the time of last write
   clock_gettime(CLOCK_REALTIME, &limitShmimMonitorT::m_imageStream.md->writetime);

   //Set the image acquisition timestamp
   limitShmimMonitorT::m_imageStream.md->atime = limitShmimMonitorT::m_imageStream.md->writetime;
                   
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

void userGainCtrl::powerLawIndex( float pli )
{
    if(pli < 0)
    {
        pli = 0;
    }

    m_powerLawIndex = pli;
    updateIfChanged(m_indiP_powerLawIndex, std::vector<std::string>({"current", "target"}), std::vector<float>({pli,pli}));
}

void userGainCtrl::powerLawFloor( float plf )
{
    m_powerLawFloor = plf;

    updateIfChanged(m_indiP_powerLawFloor, std::vector<std::string>({"current", "target"}), std::vector<float>({plf,plf}));
}

void userGainCtrl::powerLawSet()
{
    uint16_t block0 = 0;

    if(m_nZern > 0)
    {
        if(m_nZern > 1)
        {
            if(m_splitTT)
            {
                block0 = 2;
            }
            else
            {
                block0 = 1;
            }
        }

        if(m_nZern > 2)
        {
            ++block0;
        }

        if(m_nZern > 3)
        {
            ++block0;
        }
        //Now have accounted for T/T and focus.

        uint16_t currb = 1;
        while(modesAtBlock(currb) < m_nZern)
        {
            ++currb;
            ++block0;
        }
    }

    if(block0 >= m_modeBlockStart.size())
    {
        return;
    }

    if(m_powerLawIndex < 0)
    {
        m_powerLawIndex = 0;
    }

    float mode0 = m_modeBlockStart[block0] + 0.5*m_modeBlockN[block0];
    float gain0 = m_modeBlockGains[block0];
    for(size_t n=block0+1; n < m_modeBlockStart.size(); ++n)
    {
        float mode = m_modeBlockStart[n] + 0.5*m_modeBlockN[n];

        float imd1=(mode-mode0)/(m_totalNModes-mode0);
        float imd2=pow(1.0-imd1, -m_powerLawIndex) * gain0;
        float gain=(1-m_powerLawFloor)*imd2+m_powerLawFloor;

        if(gain < 0) gain = 0;

        setBlockGain(n, gain);

        //Now wait on the update, otherwise the next command can overwrite from m_gainsCurrent
        int nt = 0;
        while(fabs(m_modeBlockGains[n] - gain) > 1e-5 && nt < 100)
        {
            mx::sys::milliSleep(5);
            ++nt;
        }
    }

    log<text_log>("Set power law: " + std::to_string(m_powerLawIndex) + " " + std::to_string(m_powerLawFloor) + 
                    " starting from block " + std::to_string(block0) + " " + std::to_string(gain0));


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

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_powerLawIndex)(const pcf::IndiProperty &ipRecv)
{
   INDI_VALIDATE_CALLBACK_PROPS(m_indiP_powerLawIndex, ipRecv);

   float target;
   
   if( indiTargetUpdate( m_indiP_powerLawIndex, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   powerLawIndex(target);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_powerLawFloor)(const pcf::IndiProperty &ipRecv)
{
   INDI_VALIDATE_CALLBACK_PROPS(m_indiP_powerLawFloor, ipRecv);

   float target;
   
   if( indiTargetUpdate( m_indiP_powerLawFloor, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   powerLawFloor(target);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(userGainCtrl, m_indiP_powerLawSet)(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_powerLawSet, ipRecv);

    if(!ipRecv.find("request")) return 0;
   
    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        std::unique_lock<std::mutex> lock(m_indiMutex);

        std::cerr << "Got Power Law\n";
        powerLawSet();

        updateSwitchIfChanged(m_indiP_powerLawSet, "request", pcf::IndiElement::Off, INDI_IDLE);
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
