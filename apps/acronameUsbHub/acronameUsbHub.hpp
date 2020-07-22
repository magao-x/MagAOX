/** \file acronameUsbHub.hpp
  * \brief The MagAO-X Acroname USB Hub controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup acronameUsbHub_files
  */

#ifndef acronameUsbHub_hpp
#define acronameUsbHub_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "BrainStem2/BrainStem-all.h"

namespace MagAOX
{
namespace app
{


/** \defgroup acronameUsbHub Acroname USB Hub Controller
  * \brief Control of an Acroname USB 3.0 8-port hub
  *
  * <a href="../handbook/operating/software/apps/acronameUsbHub.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup acronameUsbHub_files Acroname USB Hub controller Files
  * \ingroup acronameUsbHub
  */

/** MagAO-X application to control an Acroname USB 3.0 8-port hub
  *
  * \todo add current, temperature, etc. monitoring
  * \todo telemetry
  * 
  * \ingroup acronameUsbHub
  * 
  */
class acronameUsbHub : public MagAOXApp<>, public dev::outletController<acronameUsbHub>
{

protected:

   /** \name configurable parameters 
     *@{
     */ 

   uint32_t m_serialNumber {0}; ///< The Acroname device serial number.
   
   ///@}
   
   
   aUSBHub3p m_hub; ///< BrainStem library handle
   
   bool m_connected {false}; ///< Whether or not the hub is currently connected
   
public:

   ///Default c'tor
   acronameUsbHub();

   ///Destructor
   ~acronameUsbHub() noexcept;

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Sets up the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for the Siglent SDG
   virtual int appLogic();

   /// Implementation of the on-power-off FSM logic
   virtual int onPowerOff();

   /// Implementation of the while-powered-off FSM
   virtual int whilePowerOff();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   
   /// Get the state of the outlet from the device.
   /** dev::outletController interface.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateOutletState( int outletNum /**< [in] the outlet number to update */);

   /// Turn an outlet on.
   /** dev::outletController interface.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int turnOutletOn( int outletNum /**< [in] the outlet number to turn on */);

   /// Turn an outlet off.
   /** dev::outletController interface.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int turnOutletOff( int outletNum /**< [in] the outlet number to turn off */);


};

inline
acronameUsbHub::acronameUsbHub() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   
   setNumberOfOutlets(8);
   
   return;
}

inline
acronameUsbHub::~acronameUsbHub() noexcept
{
   // Disconnect
    
   m_hub.disconnect();
    
   return;
}

inline
void acronameUsbHub::setupConfig()
{
   config.add("device.serialNumber", "", "device.serialNumber", argType::Required, "device", "serialNumber", false, "uint32", "The identifying serial number of the hub.");
   
   dev::outletController<acronameUsbHub>::setupConfig(config);
}


///\todo mxlib loadConfig needs to return int to propagate errors!

inline
void acronameUsbHub::loadConfig()
{
   config(m_serialNumber, "device.serialNumber");
   dev::outletController<acronameUsbHub>::loadConfig(config);
}



inline
int acronameUsbHub::appStartup()
{

   if(dev::outletController<acronameUsbHub>::setupINDI() < 0)
   {
      return log<text_log,-1>("Error setting up INDI for outlet control.", logPrio::LOG_CRITICAL);
   }
   
   return 0;

}

inline
int acronameUsbHub::appLogic()
{
   if( state() == stateCodes::POWERON)
   {
      state(stateCodes::NOTCONNECTED);
   }
   
   if( state() == stateCodes::NOTCONNECTED )
   {

      if(m_connected)
      {
         m_hub.disconnect();
         m_connected = false;
      }
      
      aErr err = aErrNone;

      elevatedPrivileges ep(this);
      
      //std::cerr << m_serialNumber << "\n";
      //err = m_hub.discoverAndConnect(USB, m_serialNumber);
      err = m_hub.connect(USB, m_serialNumber);
      if (err != aErrNone) 
      {
         if(!stateLogged())
         {
            log<text_log>("Failed to connect to usb hub", logPrio::LOG_ERROR);
         }
         
         m_connected = false;
         return 0;
      } 
      else 
      {
         state(stateCodes::CONNECTED);

         SystemClass sys;
         sys.init(&m_hub,0);
   
         uint8_t model;
         sys.getModel(&model);
         std::string modelName = aDefs_GetModelName(model);
         
         uint32_t version;
         sys.getVersion(&version);
         char versionStr[256];
         aVersion_ParseString(version, versionStr, sizeof(versionStr));
      
         uint32_t serial;
         sys.getSerialNumber(&serial);
   
         log<text_log>("Connected to " + modelName + " #" + std::to_string(serial) + " w/fimrware version " + versionStr, logPrio::LOG_INFO);
         
         m_connected = true;
         state(stateCodes::READY);
      }
   }
   
   if( state() == stateCodes::READY )
   {
      if(! m_hub.isConnected() )
      {
         m_hub.disconnect();
         m_connected = false;
         state(stateCodes::NOTCONNECTED);
         return 0;
      }
      
      
      dev::outletController<acronameUsbHub>::updateOutletStates();
      
      std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before doing INDI
      dev::outletController<acronameUsbHub>::updateINDI();
   }
   
   return 0;

}

inline
int acronameUsbHub::onPowerOff()
{
   if(m_connected)
   {
      m_hub.disconnect();
      m_connected = false;
   }
   
   for(size_t n=0;n<m_outletStates.size();++n)
   {
      m_outletStates[n] = OUTLET_STATE_OFF;
   }

   std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before doing INDI
   dev::outletController<acronameUsbHub>::updateINDI(); //Update the outlets and channel states

   //Update INDI targets to off.
   for(auto it = m_channels.begin(); it != m_channels.end(); ++it)
   {
      updateIfChanged( it->second.m_indiP_prop, "target", "Off"); 
   }


   
   return 0;
}

inline
int acronameUsbHub::whilePowerOff()
{
   return 0;
}

inline
int acronameUsbHub::appShutdown()
{
   return 0;
}

inline
int acronameUsbHub::updateOutletState( int outletNum )
{
   uint32_t state = 0;
   
   aErr err = m_hub.usb.getPortState(outletNum, &state);
   
   if(err != aErrNone)
   {
      if(err == aErrTimeout)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "timeout enabling port"});
      }
      if(err == aErrConnection)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "loss of connection detected when enabling port"});
      }
   }
   
   if(state & 1)
   {
      m_outletStates[outletNum] = OUTLET_STATE_ON;
   }
   else
   {
      m_outletStates[outletNum] = OUTLET_STATE_OFF;
   }

   return 0;
}

inline
int acronameUsbHub::turnOutletOn( int outletNum )
{
   aErr err = m_hub.usb.setPortEnable(outletNum);
   
   if(err != aErrNone)
   {
      if(err == aErrTimeout)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "timeout enabling port"});
      }
      if(err == aErrConnection)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "loss of connection detected when enabling port"});
      }
   }
      
   return 0;
}

inline
int acronameUsbHub::turnOutletOff( int outletNum )
{
   aErr err = m_hub.usb.setPortDisable(outletNum);

   if(err != aErrNone)
   {
      if(err == aErrTimeout)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "timeout disabling port"});
      }
      if(err == aErrConnection)
      {
         return log<software_error,-1>({__FILE__, __LINE__, "loss of connection detected when disabling port"});
      }
   }   
   return 0;
}




}//namespace app
} //namespace MagAOX
#endif
