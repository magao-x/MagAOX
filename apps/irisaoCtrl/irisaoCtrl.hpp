/** \file irisaoCtrl.hpp
  * \brief The MagAO-X IrisAO PTTL DM controller header file
  *
  * \ingroup irisaoCtrl_files
  */

/*
To do:
X get the code compiling
X create conf file
X rewrite zero_dm fcn
* figure out best way to check for saturation (does position query state before or after sending commands to the driver?)
* figure out shmim shape (3x37 or 37x3?)
* error checking (IrisAO API functions don't return error codes)
* figure out install process for irisAO .so and .h
X re-enable power management
*/



// #define _GLIBCXX_USE_CXX11_ABI 0


#ifndef irisaoCtrl_hpp
#define irisaoCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


/* IrisAO SDK C Header */
#include <irisao.mirrors.h>


/** \defgroup irisaoCtrl 
  * \brief The MagAO-X application to control an IrisAO DM
  *
  * <a href="..//apps_html/page_module_irisaoCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup irisaoCtrl_files
  * \ingroup irisaoCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X IrisAO DM Controller
/** 
  * \ingroup irisaoCtrl
  */
class irisaoCtrl : public MagAOXApp<true>, public dev::dm<irisaoCtrl,float>, public dev::shmimMonitor<irisaoCtrl>
{

   //Give the test harness access.
   friend class irisaoCtrl_test;
   
   friend class dev::dm<irisaoCtrl,float>;
   
   friend class dev::shmimMonitor<irisaoCtrl>;
   
   typedef float realT;  ///< This defines the datatype used to signal the DM using the ImageStreamIO library.
   
   size_t m_nsat {0};
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_mserialNumber; ///< The IrisAO MIRROR serial number
   std::string m_dserialNumber; ///< The IrisAO DRIVER serial number
   bool m_hardwareDisable; ///< Hardware disable flag (set to true to disable sending commands)
   
   ///@}

public:
   /// Default c'tor.
   irisaoCtrl();

   /// D'tor.
   ~irisaoCtrl() noexcept;

   /// Setup the configuration system.
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   /// Load the configuration
   virtual void loadConfig();

   /// Startup function
   /** Sets up INDI, and starts the shmim thread.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for irisaoCtrl.
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

   /// Cleanup after a power off.
   /**
     */ 
   virtual int onPowerOff();
   
   /// Maintenace while powered off.
   /**
     */
   virtual int whilePowerOff();
   
   /** \name DM Base Class Interface
     *
     *@{
     */
   
   /// Initialize the DM and prepare for operation.
   /** Application is in state OPERATING upon successful conclusion.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int initDM();
   
   /// Zero all commands on the DM
   /** This does not update the shared memory buffer.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */
   int zeroDM();
   
   /// Send a command to the DM
   /** This is called by the shmim monitoring thread in response to a semaphore trigger.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */
   int commandDM(void * curr_src);
   
   /// Release the DM, making it safe to turn off power.
   /** The application will be state READY at the conclusion of this.
     *  
     * \returns 0 on success 
     * \returns -1 on error
     */
   int releaseDM();
   
   ///@}
   
   /** \name IrisAO Interface
     *@{
     */
   
protected:

   uint32_t m_nbAct {0}; ///< The number of actuators
   
   double * m_dminputs {nullptr}; ///< Pre-allocated command vector, used only in commandDM
   
   MirrorHandle m_dm; ///< IrisAO SDK handle for the DM.
   
   bool m_dmopen {false}; ///< Track whether the DM connection has been opened
   
public:
   
};

irisaoCtrl::irisaoCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   return;
}

irisaoCtrl::~irisaoCtrl() noexcept
{
   //f(m_actuator_mapping) free(m_actuator_mapping);
   if(m_dminputs) free(m_dminputs);
   
}   
   
void irisaoCtrl::setupConfig()
{
   config.add("dm.mserialNumber", "", "dm.mserialNumber", argType::Required, "dm", "mserialNumber", false, "string", "The IrisAO MIRROR serial number used to find correct DM Profile.");
   config.add("dm.dserialNumber", "", "dm.dserialNumber", argType::Required, "dm", "dserialNumber", false, "string", "The IrisAO DRIVER serial number used to find correct DM Profile.");
   config.add("dm.hardwareDisable", "", "dm.hardwareDisable", argType::Required, "dm", "hardwareDisable", false, "bool", "Set to true to disable hardware for testing purposes.");
   config.add("dm.calibRelDir", "", "dm.calibRelDir", argType::Required, "dm", "calibRelDir", false, "string", "Used to find the default config directory.");
   dev::dm<irisaoCtrl,float>::setupConfig(config);
   
}

int irisaoCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   config(m_calibRelDir, "dm.calibRelDir");
   config(m_mserialNumber, "dm.mserialNumber");
   config(m_dserialNumber, "dm.dserialNumber");
   config(m_hardwareDisable, "dm.hardwareDisable");
         
   dev::dm<irisaoCtrl,float>::loadConfig(_config);
   
   return 0;
}

void irisaoCtrl::loadConfig()
{
   loadConfigImpl(config);
   
}

int irisaoCtrl::appStartup()
{

   dev::dm<irisaoCtrl,float>::appStartup();
   shmimMonitor<irisaoCtrl>::appStartup();
   
   return 0;
}

int irisaoCtrl::appLogic()
{
   dev::dm<irisaoCtrl,float>::appLogic();
   shmimMonitor<irisaoCtrl>::appLogic();
   
   if(state()==stateCodes::POWEROFF) return 0;
   
   if(state()==stateCodes::POWERON)
   {
      log<text_log>("detected POWERON");
      sleep(5);
      return initDM();
   }
   
   if(m_nsat > 0)
   {
      log<text_log>("Saturated actuators in last second: " + std::to_string(m_nsat), logPrio::LOG_WARNING);
   }
   m_nsat = 0;
   
   return 0;
}

int irisaoCtrl::appShutdown()
{
   if(m_dmopen) releaseDM();
      
   dev::dm<irisaoCtrl,float>::appShutdown();
   shmimMonitor<irisaoCtrl>::appShutdown();
   
   return 0;
}

int irisaoCtrl::onPowerOff()
{
   return dm<irisaoCtrl,float>::onPowerOff();;
}

int irisaoCtrl::whilePowerOff()
{
   return dm<irisaoCtrl,float>::whilePowerOff();;
}

int irisaoCtrl::initDM()
{
   log<text_log>("trying to init DM");
   sleep(2);

   if(m_dmopen)
   {
      log<text_log>("DM is already initialized.  Release first.", logPrio::LOG_ERROR);
      return -1;
   }

   std::string mser = mx::ioutils::toUpper(m_mserialNumber);
   std::string dser = mx::ioutils::toUpper(m_dserialNumber);

   log<text_log>("Okay, connecting now");
   sleep(2);
   try
   {
      m_dm = MirrorConnect(mser.c_str(), dser.c_str(), m_hardwareDisable); // segfault
   }
   catch (...)
   {
      return -1;
   }
   log<text_log>("GOT THE MIRROR HANDLE");
   sleep(2);

   // not sure the irisAO API gives us any output to check for success/failure
   m_dmopen = true;

   /*if(ret == NO_ERR) m_dmopen = true; // remember that the DM connection has been opened 

   if(ret != NO_ERR)
   {
      const char *err;
      err = IrisAOErrorString(ret);
      log<text_log>(std::string("DM initialization failed: ") + err, logPrio::LOG_ERROR);
      
      m_dm = {};
      return -1;
   }
   
   if (!m_dmopen)
   {
      log<text_log>("DM initialization failed. Couldn't open DM handle.", logPrio::LOG_ERROR);
      return -1;
   }*/
   
   log<text_log>("IrisAO mirror " + mser + "with driver " +  dser + " initialized", logPrio::LOG_NOTICE);

   // Get number of actuators
   // this is stupid, but I don't know how else to get this number
   SegmentNumber segment = 0; // don't know if this should start at 0 or 1
   while (MirrorIterate(m_dm, segment)){
      segment++;
   }
   m_nbAct = 3*segment; // 3*(segment+1)
   log<text_log>("Found " + std::to_string(segment) + " segments for IrisAO mirror " + mser, logPrio::LOG_NOTICE);

   // cacao input -- FIX ME?
   if(m_dminputs) free(m_dminputs);
   m_dminputs = (double*) calloc( m_nbAct, sizeof( double ) );
   
   if(zeroDM() < 0)
   {
      log<text_log>("DM initialization failed.  Error zeroing DM.", logPrio::LOG_ERROR);
      return -1;
   }
   
   state(stateCodes::OPERATING);
   
   return 0;
}

int irisaoCtrl::zeroDM()
{
   if(!m_dmopen)
   {
      log<text_log>("DM not initialized (NULL pointer)", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(m_nbAct == 0)
   {
      log<text_log>("DM not initialized (number of actuators)", logPrio::LOG_ERROR);
      return -1;
   }
   
   /* Send the all 0 command to the DM */
   SegmentNumber segment = 0;
   while (MirrorIterate(m_dm, segment)){
      SetMirrorPosition(m_dm, segment, 0, 0, 0); // z, xgrad, ygrad
      segment++;
   }
   MirrorCommand(m_dm, MirrorSendSettings);
   
   log<text_log>("DM zeroed");
   return 0;
}

int irisaoCtrl::commandDM(void * curr_src)
{
   //Based on Alex Rodack's IrisAO script
   SegmentNumber segment = 0; // start at 0 or 1?
   MirrorPosition position;
   int idx;
   while (MirrorIterate(m_dm, segment)){

      // need shmim array formatted in a way that's consistent with this loop
      idx = segment * 3; // may need (segment - 1) if they start counting segments at 1
      SetMirrorPosition(m_dm, segment, ((float *)curr_src)[idx], ((float *)curr_src)[idx+1], ((float *)curr_src)[idx+2]); // z, xgrad, ygrad
   
      // check if the current segment was saturated
      // not sure you can do this here. might need to send the commands first (depends on what this is actually querying)
      // not sure how to handle ptt in the m_instSatMap. I guess I need to saturate a whole row/column at once??
      GetMirrorPosition(m_dm, segment, &position);
      if (!position.reachable)
      {
         m_instSatMap.data()[idx] = 1;
         m_instSatMap.data()[idx+1] = 1;
         m_instSatMap.data()[idx+2] = 1;
      }
      else
      {
         m_instSatMap.data()[idx] = 0;
         m_instSatMap.data()[idx+1] = 0;
         m_instSatMap.data()[idx+2] = 0;
      }

       segment++;
   }

   /* Finally, send the command to the DM */
   MirrorCommand(m_dm, MirrorSendSettings);
   
   return 0 ;
}

int irisaoCtrl::releaseDM()
{
   // Safe DM shutdown on interrupt

   if(!m_dmopen)
   {
      log<text_log>("dm is not initialized", logPrio::LOG_ERROR);
      return -1;
   }
   
   state(stateCodes::READY);
   
   if(!shutdown())
   {
      pthread_kill(m_smThread.native_handle(), SIGUSR1);
   }
   
   sleep(1); ///\todo need to trigger shmimMonitor loop to pause it.
   
   if(zeroDM() < 0)
   {
      log<text_log>("DM release failed.  Error zeroing DM.", logPrio::LOG_ERROR);
      return -1;
   }
   
   // Close IrisAO connection
   MirrorRelease(m_dm);

   m_dmopen = false;
   m_dm = {};
   
   log<text_log>("IrisAO " + m_mserialNumber + " with driver " + m_dserialNumber + " reset and released", logPrio::LOG_NOTICE);
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //irisaoCtrl_hpp
