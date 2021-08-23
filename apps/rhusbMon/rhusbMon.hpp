/** \file rhusbMon.hpp
  * \brief The MagAO-X RH USB monitor
  *
  * \ingroup rhusbMon_files
  */

#ifndef rhusbMon_hpp
#define rhusbMon_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include "rhusbMonParsers.hpp"

/** \defgroup rhusbMon
  * \brief Application to monitor an Omega RH USB probe.
  *
  * <a href="../handbook/operating/software/apps/rhusbMon.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup rhusbMon_files
  * \ingroup rhusbMon
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X RH-USB monitoring class
/** Interacts with the Omega RH-USB probe used for DM chamber humidity monitoring.
  * 
  * \todo need a test mode (compile-time) which adds a way (INDI?) to initiate testing of parameter limits.
  * 
  * \ingroup rhusbMon
  */
class rhusbMon : public MagAOXApp<true>, public tty::usbDevice, public dev::ioDevice, public dev::telemeter<rhusbMon>
{

   //Give the test harness access.
   friend class rhusbMon_test;

   //Let telemeter work.
   friend class dev::telemeter<rhusbMon>;
   
protected:

   /** \name Configurable Parameters
     *@{
     */   
   float m_warnTemp {30};  ///< This is abnormally high if the system is working, but still safe.
   float m_alertTemp {35}; ///< This is the actual limit, shut down should occur.
   float m_emergTemp {40}; ///< Must shutdown immediately.
   
   float m_warnHumid {18};  ///< This is abnormally high if the system is working, but still safe.
   float m_alertHumid {20}; ///< This is the actual limit, shut down should occur.
   float m_emergHumid {22}; ///< Must shutdown immediately.

   ///@}
   
   float m_temp {-999};
   float m_rh {-999};
   
   pcf::IndiProperty m_indiP_temp;
   pcf::IndiProperty m_indiP_rh;
   

public:
   /// Default c'tor.
   rhusbMon();

   /// D'tor, declared and defined for noexcept.
   ~rhusbMon() noexcept
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

   /// Implementation of the FSM for rhusbMon.
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

   /// Connect to the probe
   /** Search for the USB device in udev and attempt ot open it.
     * The result is reported via the FSM state (NODEVICE, NOTCONNECTED, CONNECTED).
     * 
     * \returns -1 on an error attempting to read udev 
     * \returns 0 if device not found, or connection does not work, or if connected. 
     */
   int connect();

   /// Read current values from the RH-USB probe
   /** Issues the 'C' and 'H' commands to get temperature and humidity.
     * 
     * \returns -1 on error writing or reading, or on a parsing error
     * 
     * \see \parseC( float &, const std::string)
     * \see \parseH( float &, const std::string)
     */ 
   int readProbe();

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_rhusb * );
   
protected:

   int recordRH( bool force = false );

   ///@}
   
};

rhusbMon::rhusbMon() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   dev::ioDevice::m_readTimeout = 2000;
   dev::ioDevice::m_writeTimeout = 1000;

   return;
}

void rhusbMon::setupConfig()
{
   config.add("temp.warning", "", "temp.warning", argType::Required, "temp", "warning", false, "float", "Temperature at which to issue a warning.  Default is 30.");
   config.add("temp.alert", "", "temp.alert", argType::Required, "temp", "alert", false, "float", "Temperature at which to issue an alert.  Default is 35.");
   config.add("temp.emergency", "", "temp.emergency", argType::Required, "temp", "emergency", false, "float", "Temperature at which to issue an emergency.  Default is 40.");
   
   config.add("humid.warning", "", "humid.warning", argType::Required, "humid", "warning", false, "float", "Humidity at which to issue a warning.  Default is 18.");
   config.add("humid.alert", "", "humid.alert", argType::Required, "humid", "alert", false, "float", "Humidity at which to issue an alert.  Default is 20.");
   config.add("humid.emergency", "", "humid.emergency", argType::Required, "humid", "emergency", false, "float", "Humidity at which to issue an emergency.  Default is 22.");
   
   tty::usbDevice::setupConfig(config);
   dev::ioDevice::setupConfig(config);

   dev::telemeter<rhusbMon>::setupConfig(config);
}

int rhusbMon::loadConfigImpl( mx::app::appConfigurator & _config )
{

   _config(m_warnTemp, "temp.warning");
   _config(m_alertTemp, "temp.alert");
   _config(m_emergTemp, "temp.emergency");
   
   _config(m_warnHumid, "humid.warning");
   _config(m_alertHumid, "humid.alert");
   _config(m_emergHumid, "humid.emergency");

   tty::usbDevice::loadConfig(_config);
   dev::ioDevice::loadConfig(_config);

   dev::telemeter<rhusbMon>::loadConfig(_config);
   
   return 0;
}

void rhusbMon::loadConfig()
{
   loadConfigImpl(config);
}

int rhusbMon::appStartup()
{
   createROIndiNumber( m_indiP_temp, "temperature", "Temperature [C]");
   indi::addNumberElement<float>( m_indiP_temp, "current", -20., 120., 0, "%0.1f");
   m_indiP_temp["current"] = -999;
   registerIndiPropertyReadOnly(m_indiP_temp);
   
   createROIndiNumber( m_indiP_rh, "humidity", "Relative Humidity [%]");
   indi::addNumberElement<float>( m_indiP_rh, "current", 0., 100., 0, "%0.1f");
   m_indiP_rh["current"] = -999;
   registerIndiPropertyReadOnly(m_indiP_rh);

   
   if(dev::telemeter<rhusbMon>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   return connect();
}



int rhusbMon::appLogic()
{
   if(state() == stateCodes::NODEVICE || state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      int rv = connect();
      if(rv < 0) return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(state() == stateCodes::CONNECTED || state() == stateCodes::OPERATING)
   {
      int rv = readProbe();
      if(rv == 0)
      {
         state(stateCodes::OPERATING);
      }
      else
      {
         state(stateCodes::ERROR);
         return log<software_error,0>({__FILE__, __LINE__});
      }
   }
   
   pcf::IndiProperty::PropertyStateType rhState = pcf::IndiProperty::Ok;
   //Check warning and alert values
   if(m_rh > m_emergHumid)
   {
      log<text_log>("RH > " + std::to_string(m_emergHumid) + "% : " + std::to_string(m_rh) + "%!  Shutdown immediately!", logPrio::LOG_EMERGENCY); 
      rhState = pcf::IndiProperty::Alert;
   }
   else if(m_rh > m_alertHumid)
   {
      log<text_log>("RH > " + std::to_string(m_alertHumid) + "% : " + std::to_string(m_rh) + "%.  Fix or shutdown.", logPrio::LOG_ALERT); 
      rhState = pcf::IndiProperty::Alert;
   }
   else if(m_rh > m_warnHumid)
   {
      log<text_log>("RH > " + std::to_string(m_warnHumid) + "% : " + std::to_string(m_rh) + "%.", logPrio::LOG_WARNING);
      rhState = pcf::IndiProperty::Alert;
   }

   pcf::IndiProperty::PropertyStateType tState = pcf::IndiProperty::Ok;
   //Check warning and alert values
   if(m_temp > m_emergTemp)
   {
      log<text_log>("Temp > " + std::to_string(m_emergTemp) + "C : " + std::to_string(m_temp) + "C!  Shutdown immediately!", logPrio::LOG_EMERGENCY); 
      tState = pcf::IndiProperty::Alert;
   }
   else if(m_temp > m_alertTemp)
   {
      log<text_log>("Temp > " + std::to_string(m_alertTemp) + "C : " + std::to_string(m_temp) + "C.  Fix or shutdown.", logPrio::LOG_ALERT); 
      tState = pcf::IndiProperty::Alert;
   }
   else if(m_temp > m_warnTemp)
   {
      log<text_log>("Temp > " + std::to_string(m_warnTemp) + "C : " + std::to_string(m_temp) + "C.", logPrio::LOG_WARNING);
      tState = pcf::IndiProperty::Alert;
   }

   //Scope for mutex
   { 
      std::unique_lock<std::mutex> lock(m_indiMutex);
      updateIfChanged(m_indiP_temp, "current", m_temp, tState);

      m_indiP_rh["current"].set<float>(-999); //Force the update to get a new timestamp
      updateIfChanged(m_indiP_rh, "current", m_rh, rhState); ///\todo updateIfChanged should have a force flag
   }

   if(telemeter<rhusbMon>::appLogic() < 0)
   {
      return log<software_error,0>({__FILE__, __LINE__});
   }
      
   return 0;
}

int rhusbMon::appShutdown()
{
   return 0;
}

int rhusbMon::connect()
{
   int rv = tty::usbDevice::getDeviceName();
   if(rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
   {
      //There is no device reason for this to error.  Something is wrong.
      state(stateCodes::FAILURE);
      return log<software_critical, -1>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }

   if(rv == TTY_E_DEVNOTFOUND || rv == TTY_E_NODEVNAMES)
   {
      state(stateCodes::NODEVICE);

      if(!stateLogged())
      {
         std::stringstream logs;
         logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " not found in udev";
         log<text_log>(logs.str());
      }
      return 0;
   }
   else
   {
      state(stateCodes::NOTCONNECTED);
      if(!stateLogged())
      {
         std::stringstream logs;
         logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
         log<text_log>(logs.str());
      }

      //scope for elevated priv
      {
         elevatedPrivileges elPriv(this);
         rv = tty::usbDevice::connect();
      }

      if(rv == TTY_E_NOERROR)
      {
         state(stateCodes::CONNECTED);
         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "Connected to " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " @ " << m_deviceName;
            log<text_log>(logs.str());
         }
      }
      else
      {
         //There is no power or other reason this should happen.  It means something is wrong and needs to be corrected.
         state(stateCodes::FAILURE);
         return log<software_critical, -1>({__FILE__,__LINE__, errno, rv, "Error opening connection: " + tty::ttyErrorString(rv)});
      }
   }

   return 0;
}

int rhusbMon::readProbe()
{
   std::string strRead;

   int rv = tty::ttyWriteRead( strRead, "C\r", "\r\n>", false, m_fileDescrip, dev::ioDevice::m_writeTimeout, dev::ioDevice::m_readTimeout);
   if(rv != TTY_E_NOERROR)
   {
      return log<software_error,-1>({__FILE__, __LINE__,  0, rv, "Error reading temp: " + tty::ttyErrorString(rv)});
   }

   rv = RH::parseC(m_temp, strRead);
   if(rv != 0)
   {
      if( rv == -1 )
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing temp, no EOT"});
      }
      else if (rv == -2 )
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing temp, no value"});
      }
      else if (rv == -3)
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing temp, does not begin with digit"});
      }
      else
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing temp."});
      }
   }

   std::cout << m_temp << "\n";

   rv = tty::ttyWriteRead( strRead, "H\r", "\r\n>", false, m_fileDescrip, dev::ioDevice::m_writeTimeout, dev::ioDevice::m_readTimeout);
   if(rv != TTY_E_NOERROR)
   {
      return log<software_error,-1>({__FILE__, __LINE__,  0, rv, "Error reading RH: " + tty::ttyErrorString(rv)});
   }

   rv = RH::parseH(m_rh, strRead);
   if(rv != 0)
   {
      if( rv == -1 )
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing humid, no EOT"});
      }
      else if (rv == -2 )
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing humid, no value"});
      }
      else if (rv == -3)
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing temp, does not begin with digit"});
      }
      else
      {
         return log<software_error, -1>({__FILE__, __LINE__, "Error parsing humid."});
      }
   }

   std::cout << m_rh << "\n";

   return 0;
}

int rhusbMon::checkRecordTimes()
{
   return dev::telemeter<rhusbMon>::checkRecordTimes(telem_rhusb());
}
   
int rhusbMon::recordTelem( const telem_rhusb * )
{
   return recordRH(true);
}

inline 
int rhusbMon::recordRH(bool force)
{
   static float lastTemp = -99;
   static float lastRH = -99;
   
   if(force || m_temp != lastTemp || m_rh != lastRH)
   {
      telem<telem_rhusb>({m_temp, m_rh});
   }

   lastTemp = m_temp;
   lastRH = m_rh;

   return 0;

}

} //namespace app
} //namespace MagAOX

#endif //rhusbMon_hpp
