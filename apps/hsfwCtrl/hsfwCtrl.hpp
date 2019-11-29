/** \file hsfwCtrl.hpp
  * \brief The MagAO-X Optec HSFW Filter Wheel Controller
  *
  * \ingroup hsfwCtrl_files
  */


#ifndef hsfwCtrl_hpp
#define hsfwCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "libhsfw.h"

/** \defgroup hsfwCtrl Optec HSFW Filter Wheel Control
  * \brief Control of an Optec HSFW f/w.
  *
  * <a href="../handbook/operating/software/apps/hsfwCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup hsfwCtrl_files Filter Wheel Control Files
  * \ingroup hsfwCtrl
  */

namespace MagAOX
{
namespace app
{

/** MagAO-X application to control an Optec High Speed Filter Wheel (HSFW).
  *
  * \todo add tests
  *
  * \ingroup hsfwCtrl
  */
class hsfwCtrl : public MagAOXApp<>,  public dev::stdMotionStage<hsfwCtrl>, public dev::telemeter<hsfwCtrl>
{

   friend class dev::stdMotionStage<hsfwCtrl>;
   
   friend class dev::telemeter<hsfwCtrl>;
   
protected:

   /** \name Non-configurable parameters
     *@{
     */


   ///@}

   /** \name Configurable Parameters
     * @{
     */
   
   std::wstring m_serialNumber;

   ///@}

   /** \name Status
     * @{
     */

   hsfw_wheel* m_wheel {nullptr};
   
   double m_pos {0};
   
   ///@}


public:

   /// Default c'tor.
   hsfwCtrl();

   /// D'tor, declared and defined for noexcept.
   ~hsfwCtrl() noexcept
   {}

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Setsup the INDI vars.
     *
     * \returns 0 on success
     * \returns -1 on error.
     */
   virtual int appStartup();

   /// Implementation of the FSM for the TTM Modulator
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   /**
     * \returns 0 on success
     * \returns -1 on error.
     */
   virtual int appShutdown();


   /// This method is called when the change to poweroff is detected.
   /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */
   virtual int onPowerOff();

   /// This method is called while the power is off, once per FSM loop.
   /**
     * \returns 0 on success.
     * \returns -1 on any error which means the app should exit.
     */
   virtual int whilePowerOff();


protected:
  
   

   /// Start a high-level homing sequence.
   /** For this device this includes the homing dither.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int startHoming();
   
   int presetNumber();
   
   /// Start a low-level homing sequence.
   /** This initiates the device homing sequence.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int home();

   /// Stop the wheel motion immediately.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int stop();

   /// Move to an absolute position in filter units.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveTo( const double & filters /**< [in] The new position in absolute filter units*/ );

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_stage * );
   
   int recordStage( bool force = false );
   
};

inline
hsfwCtrl::hsfwCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_presetNotation = "filter"; //sets the name of the configs, etc.
   
   m_powerMgtEnabled = true;
   
   return;
}

inline
void hsfwCtrl::setupConfig()
{
   config.add("stage.serialNumber", "", "stage.serialNumber", argType::Required, "stage", "serialNumber", false, "string", "The device serial number.");
   
   dev::stdMotionStage<hsfwCtrl>::setupConfig(config);
   
   dev::telemeter<hsfwCtrl>::setupConfig(config);
   
}

inline
void hsfwCtrl::loadConfig()
{
   std::string serNum;
   config(serNum, "stage.serialNumber");
   m_serialNumber.assign(serNum.begin(), serNum.end());
   
   dev::stdMotionStage<hsfwCtrl>::loadConfig(config);

   dev::telemeter<hsfwCtrl>::loadConfig(config);
}

inline
int hsfwCtrl::appStartup()
{
   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

   
   if( dev::stdMotionStage<hsfwCtrl>::appStartup() < 0)
   {
      return log<software_critical,-1>({__FILE__,__LINE__});
   }
   
   if(dev::telemeter<hsfwCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   return 0;
}

inline
int hsfwCtrl::appLogic()
{
   if( state() == stateCodes::INITIALIZED )
   {
      log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

   if( state() == stateCodes::POWERON )
   {
      state(stateCodes::NODEVICE);
   }
   
   if( state() == stateCodes::NODEVICE )
   {
      hsfw_wheel_info *devs, *cur_dev;

      devs = enumerate_wheels();
      
      if(devs == NULL)
      {
         return 0;
      }
      
      cur_dev = devs;
      while (cur_dev) 
      {
         if(m_serialNumber == cur_dev->serial_number)
         {
            char logs[1024];
         
            snprintf(logs, sizeof(logs), "Device Found - type: %04hx %04hx serial_number: %ls",cur_dev->vendor_id, cur_dev->product_id, cur_dev->serial_number);
            log<text_log>(logs);
         
            state(stateCodes::NOTCONNECTED);
            break;
         }
         cur_dev = cur_dev->next;
      }
      wheels_free_enumeration(devs);
      
      if(state() != stateCodes::NOTCONNECTED)
      {
         if(!stateLogged())
         {
            log<text_log>("Device " + std::string(m_serialNumber.begin(), m_serialNumber.end()) + " not found");
         }
         return 0;
      }
   }

   if( state() == stateCodes::NOTCONNECTED )
   {
      hsfw_wheel_info *devs, *cur_dev;
      devs = enumerate_wheels();
      
      if(devs == NULL)
      {
         state(stateCodes::NODEVICE);
         return 0;
      }

      cur_dev = devs;
      
      while (cur_dev) 
      {        
         if(m_serialNumber == cur_dev->serial_number) 
         {
            break;
         }
         
         cur_dev = cur_dev->next;
      }

      if(cur_dev == NULL)
      {
         wheels_free_enumeration(devs);
         state(stateCodes::NODEVICE);
         return 0;
      }
     
      if(m_wheel) close_hsfw(m_wheel);
      
      euidCalled();
      m_wheel = open_hsfw(cur_dev->vendor_id, cur_dev->product_id, cur_dev->serial_number);
      euidReal();
   
      if(m_wheel == NULL)
      {
         state(stateCodes::NODEVICE);
         return 0;
      }
           
     
      state(stateCodes::CONNECTED);
      log<text_log>("Connected to HSFW " + std::string(m_serialNumber.begin(), m_serialNumber.end()));
   }
   
   
   //If here, we're connected.
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   wheel_status status;
   if (get_hsfw_status(m_wheel, &status) < 0) 
   {
      printf("ERROR");
      return 0;
   }
   
   if (status.error_state != 0) 
   {
      printf("Clearing Error\n");
      clear_error_hsfw(m_wheel);
   }
   
   m_pos = status.position;
   
   if(!status.is_homed && !status.is_homing)
   {
      state(stateCodes::NOTHOMED);
      m_moving = -1;
      
      if(m_powerOnHome) 
      {
         startHoming();
      }
   }
   else if( status.is_homing)
   {
      m_moving=2;
      state(stateCodes::HOMING);
   }
   else if (status.is_moving)
   {
      m_moving = 1;
      state(stateCodes::OPERATING);
   }
   else 
   {
      m_moving = 0;
      state(stateCodes::READY);
   }
   
   int n = presetNumber();
   if(n == -1)
   {
      m_preset = 0;
      m_preset_target = 0;
   }
   else
   {
      m_preset = n+1;
      m_preset_target = n+1;
   }

   //record telem if there have been any changes
   recordStage();
   
   
   dev::stdMotionStage<hsfwCtrl>::updateINDI();
   
   //record telem if it's been longer than 10 sec:
   if(telemeter<hsfwCtrl>::appLogic() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return 0;
   }
   
   
   return 0;
}



inline
int hsfwCtrl::appShutdown()
{
   if(m_wheel) close_hsfw(m_wheel);
   
   exit_hsfw();
            
   return 0;
}

inline
int hsfwCtrl::onPowerOff()
{
   if( stdMotionStage<hsfwCtrl>::onPowerOff() < 0)
   {
      log<software_error>({__FILE__,__LINE__});
   }
   
   recordStage();

   return 0;
}
   

inline
int hsfwCtrl::whilePowerOff()
{
   if( stdMotionStage<hsfwCtrl>::whilePowerOff() < 0)
   {
      log<software_error>({__FILE__,__LINE__});
   }
   
   //record telem if it's been longer than 10 sec:
   if(telemeter<hsfwCtrl>::appLogic() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   return 0;
}

   
int hsfwCtrl::startHoming()
{
   updateSwitchIfChanged(m_indiP_home, "request", pcf::IndiElement::Off, INDI_IDLE);
   
   
   if(home_hsfw(m_wheel)) 
   {
      log<software_error>({__FILE__,__LINE__, "libhswf error"});
      return -1;
   }
   
   m_moving = 2;
   
   return 0;
}

int hsfwCtrl::presetNumber()
{
   return m_pos-1;
}



int hsfwCtrl::stop()
{
   updateSwitchIfChanged(m_indiP_stop, "request", pcf::IndiElement::Off, INDI_IDLE);
   return 0;
}


int hsfwCtrl::moveTo( const double & filters )
{
   double ffilters = filters;
   if(ffilters< 0.5)
   {
      while(ffilters < 8.5) ffilters += 8;
      if(ffilters >= 8.5)
      {
         return log<software_error,-1>({__FILE__,__LINE__, "error getting modulo filter number"});
      }
   }

   m_moving = 1;   
   recordStage();
   
   if( move_hsfw(m_wheel, (unsigned short) (ffilters + 0.5)) < 0)
   {
      
      return log<software_error,-1>({__FILE__,__LINE__, "libhsfw error"});
   }
   
   return 0;
}

int hsfwCtrl::checkRecordTimes()
{
   return dev::telemeter<hsfwCtrl>::checkRecordTimes(telem_stage());
}
   
int hsfwCtrl::recordTelem( const telem_stage * )
{
   return recordStage(true);
}

int hsfwCtrl::recordStage( bool force )
{
   return dev::stdMotionStage<hsfwCtrl>::recordStage(force);
}

} //namespace app
} //namespace MagAOX

#endif //hsfwCtrl_hpp
