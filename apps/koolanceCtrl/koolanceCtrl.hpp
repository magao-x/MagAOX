/** \file koolanceCtrl.hpp
  * \brief The MagAO-X Koolance Controller header file
  *
  * \ingroup koolanceCtrl_files
  */

#ifndef koolanceCtrl_hpp
#define koolanceCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup koolanceCtrl
  * \brief The MagAO-X application to monitor and control a Koolance cooler
  *
  * <a href="../handbook/operating/software/apps/koolanceCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup koolanceCtrl_files
  * \ingroup koolanceCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X Koolance Controller
/** This application will monitor a Koolance v1 or v2 protocol controller.  If v2, will also allow
  * changing of pump and fan settings via INDI.
  * 
  * \ingroup koolanceCtrl
  */
class koolanceCtrl : public MagAOXApp<true>, public tty::usbDevice, public dev::telemeter<koolanceCtrl>
{

   //Give the test harness access.
   friend class koolanceCtrl_test;

   friend class dev::telemeter<koolanceCtrl>;
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}

   size_t m_protocolChars {0}; ///< Will be set to 43 if protocol 1, and set to 51 if protocol 2.
   
   bool m_indiSetup {false}; ///< Whether or not INDI has been set up after initial protocol determination.
   
   float m_liqTemp {0}; ///< The liquid temperature
   float m_flowRate {0}; ///< The flow rate
   int m_pumpLvl {0}; ///< The pump power level, 1-10
   int m_pumpRPM {0}; ///< The pump RPM
   int m_fanRPM {0}; ///< The fan RPM
   int m_fanLvl {0}; ///< The fan power level, 0-100



public:
   /// Default c'tor.
   koolanceCtrl();

   /// D'tor, declared and defined for noexcept.
   ~koolanceCtrl() noexcept
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

   /// Implementation of the FSM for koolanceCtrl.
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

   /// Initial connection to controller
   /** Determine protocol in use based on number of characters in response
     * and sets up INDI appropriately.
     *
     * Calls to this function should be mutexed.
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int initialConnect();
   
   /// Get status from controller and updated INDI.
   /** 
     *
     * Calls to this function should be mutexed.
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int getStatus();
   
   /// Set the pump level
   /** 
     * Calls to this function should be mutexed.
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setPumpLvl(int lvl /**< [in] the new level */);
   
   /// Set the fan level
   /** 
     * Calls to this function should be mutexed.
     * 
     * \returns 0 on success
     * \returns -1 on an error
     */
   int setFanLvl(int lvl /**< [in] the new level */);
   
   /** \name INDI 
     *
     *@{
     */ 
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_status;
   pcf::IndiProperty m_indiP_pumplvl;
   pcf::IndiProperty m_indiP_fanlvl;

public:
   INDI_NEWCALLBACK_DECL(koolanceCtrl, m_indiP_pumplvl);
   INDI_NEWCALLBACK_DECL(koolanceCtrl, m_indiP_fanlvl);
   
   ///@}
   
   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_cooler * );
   
   int recordCooler(bool force = false);
   
   ///@}


};

koolanceCtrl::koolanceCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void koolanceCtrl::setupConfig()
{
   tty::usbDevice::setupConfig(config);
   
   dev::telemeter<koolanceCtrl>::setupConfig(config);
   
}

int koolanceCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   this->m_baudRate = B9600; //default for a Koolance controller.  Will be overridden by any config setting.
   
   int rv = tty::usbDevice::loadConfig(_config);

   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   return 0;
}

void koolanceCtrl::loadConfig()
{
   loadConfigImpl(config);
   
   dev::telemeter<koolanceCtrl>::loadConfig(config);
}

int koolanceCtrl::appStartup()
{
   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }
   
   // set up the  INDI properties
   REG_INDI_NEWPROP_NOCB(m_indiP_status, "status", pcf::IndiProperty::Number);
   m_indiP_status.add(pcf::IndiElement("liquid_temp"));
   m_indiP_status.add(pcf::IndiElement("flow_rate"));
   m_indiP_status.add(pcf::IndiElement("pump_rpm"));
   m_indiP_status.add(pcf::IndiElement("fan_rpm"));
   
   if(dev::telemeter<koolanceCtrl>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   state(stateCodes::NODEVICE);
   return 0;
}

int koolanceCtrl::appLogic()
{
   
   if( state() == stateCodes::NODEVICE )
   {
      int rv = tty::usbDevice::getDeviceName();
      if(rv < 0 && rv != TTY_E_DEVNOTFOUND && rv != TTY_E_NODEVNAMES)
      {
         state(stateCodes::FAILURE);
         if(!stateLogged())
         {
            log<software_critical>({__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
         }
         return -1;
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
         std::stringstream logs;
         logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
         log<text_log>(logs.str());
         
         state(stateCodes::NOTCONNECTED);
      }
   }
   
   if( state() == stateCodes::NOTCONNECTED )
   {
      int rv = 0;
      {
         elevatedPrivileges ep(this);
         rv = connect();
      }

      if(rv < 0)
      {
         int nrv = tty::usbDevice::getDeviceName();
         if(nrv < 0 && nrv != TTY_E_DEVNOTFOUND && nrv != TTY_E_NODEVNAMES)
         {
            state(stateCodes::FAILURE);
            if(!stateLogged()) log<software_critical>({__FILE__, __LINE__, nrv, tty::ttyErrorString(nrv)});
            return -1;
         }

         if(nrv == TTY_E_DEVNOTFOUND || nrv == TTY_E_NODEVNAMES)
         {
            state(stateCodes::NODEVICE);

            if(!stateLogged())
            {
               std::stringstream logs;
               logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " no longer found in udev";
               log<text_log>(logs.str());
            }
            return 0;
         }

         //if connect failed, and there is a device, then we have some other problem.
         sleep(1); //wait to see if power state updates 
         if(m_powerState == 0) return 0;
         
         //Ok we can't figure this out, die.
         state(stateCodes::FAILURE);
         if(!stateLogged()) log<software_error>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
         return -1;

      }
      else 
      {
         std::unique_lock<std::mutex> lock(m_indiMutex);
         if(initialConnect () == 0) 
         {
            state(stateCodes::CONNECTED);
         }
         else 
         {
            if(!stateLogged())
            {
               log<text_log>("no response from device");
            }
            return 0;
         }
      }
      
      if(state() == stateCodes::CONNECTED)
      {
         std::stringstream logs;
         logs << "Connected to koolance system on " << m_deviceName;
         log<text_log>(logs.str());
         state(stateCodes::READY);
         return 0;
      }

   }

   if( state() == stateCodes::READY )
   {
      { //mutex scope
         std::unique_lock<std::mutex> lock(m_indiMutex);
         if(getStatus() < 0)
         {
            log<software_error>({__FILE__,__LINE__});
         }
      }
      
      if(telemeter<koolanceCtrl>::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
      return 0;
   }
   
   return 0;
}

int koolanceCtrl::appShutdown()
{
   dev::telemeter<koolanceCtrl>::appShutdown();
   return 0;
}



int koolanceCtrl::initialConnect()
{
   int rv;
   
   std::vector<unsigned char> com;
   com.resize(3, '\0');
   com[0] = 0xCF;
   com[1] = 0x01;
   com[2] = 0x08;
   
   rv = write(m_fileDescrip, com.data(), com.size());
   
   if(rv < 0)
   {
      log<software_error>({__FILE__,__LINE__, errno, "error from write"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }

   std::vector<unsigned char> resp;
   resp.resize(51);
   
   mx::sys::milliSleep(1000); //Sleep for a long time to make sure device responds
   int readBytes;
   rv = tty::ttyReadRaw(resp, readBytes, m_fileDescrip, 1000); ///\todo needs to be iodevice
   
   if(rv < 0)
   {
      log<software_error>({__FILE__,__LINE__, errno, "error from read"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }
   
   if(readBytes == 43)
   {
      log<text_log>("found protocol 1 device");
      m_protocolChars = 43;
   }
   else if(readBytes == 51)
   {
      log<text_log>("found protocol 2 device");
      m_protocolChars = 51;
   }
   else
   {
      return -1;
   }
   
   if(!m_indiSetup)
   {
      if(readBytes == 43)
      {
          REG_INDI_NEWPROP_NOCB(m_indiP_pumplvl, "pump_level", pcf::IndiProperty::Number);
          m_indiP_pumplvl.add(pcf::IndiElement("current"));
          
          REG_INDI_NEWPROP_NOCB(m_indiP_fanlvl, "fan_level", pcf::IndiProperty::Number);
          m_indiP_fanlvl.add(pcf::IndiElement("current"));
      }
      else
      {
         createStandardIndiNumber<int>( m_indiP_pumplvl, "pump_level", 1, 10, 1, "%d", "Pump Level", "Lab");
         registerIndiPropertyNew( m_indiP_pumplvl, INDI_NEWCALLBACK(m_indiP_pumplvl));
   
         createStandardIndiNumber<int>( m_indiP_fanlvl, "fan_level", 0, 100, 1, "%d", "Fan Level", "Lab");
         registerIndiPropertyNew( m_indiP_fanlvl, INDI_NEWCALLBACK(m_indiP_fanlvl));
      }
      
      m_indiSetup = true;
   }
   
   return 0;
}

int koolanceCtrl::getStatus()
{
   int rv;
   
   std::vector<unsigned char> com;
   com.resize(3, '\0');
   com[0] = 0xCF;
   com[1] = 0x01;
   com[2] = 0x08;
   
   rv = write(m_fileDescrip, com.data(), com.size());
   
   if(rv < 0)
   {
      log<software_error>({__FILE__,__LINE__, errno, "error from write"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }

   std::string resp;
   
   rv = tty::ttyRead(resp, m_protocolChars, m_fileDescrip, 1000); ///\todo needs to be iodevice
   
   if(rv < 0)
   {
      log<software_error>({__FILE__,__LINE__, errno, "error from read"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }
   
   if(resp.size() == m_protocolChars)
   {
      m_liqTemp = ((float) (( (unsigned char)resp[2] << 8) + (unsigned char)resp[3]-2000)) / 10.0;
      m_flowRate = ( (float) ((unsigned char)resp[12] << 8) + (unsigned char)resp[13])  / 10.0;
      m_fanRPM = ((unsigned char)resp[8] << 8) + (unsigned char)resp[9];
      m_pumpRPM = ((unsigned char)resp[10] << 8) + (unsigned char)resp[11];
      m_fanLvl = (unsigned char)resp[15];
      m_pumpLvl = (unsigned char)resp[17];
      
      recordCooler();
//       std::cout << std::dec;
//       std::cout << "liq. temp: " << m_liqTemp << " C\n";
//       std::cout << "flow rate: " << m_flowRate << " LPM\n";
//       std::cout << "pump lvl:  " << m_pumpLvl << "\n";
//       std::cout << "pump speed:" << m_pumpRPM << " RPM\n";
//       std::cout << "fan lvl:  " << m_fanLvl << "\n";
//       std::cout << "fan speed:" << m_fanRPM << " RPM\n";
      
      updateIfChanged(m_indiP_status, "liquid_temp", m_liqTemp);
      updateIfChanged(m_indiP_status, "flow_rate", m_flowRate);
      updateIfChanged(m_indiP_status, "pump_rpm", m_pumpRPM);
      updateIfChanged(m_indiP_status, "fan_rpm", m_fanRPM);
      updateIfChanged(m_indiP_pumplvl, "current", m_pumpLvl);
      updateIfChanged(m_indiP_fanlvl, "current", m_fanLvl);
      
      return 0;
   }
   else
   {
      log<software_error>({__FILE__,__LINE__, std::string("wrong response size (") + std::to_string(resp.size()) + ") returned"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }
}

int koolanceCtrl::setPumpLvl( int lvl )
{
   if(m_protocolChars == 43) return 0;
   
   int rv;
   
   std::vector<unsigned char> com;
   com.resize(m_protocolChars, '\0');
   com[0] = 0xCF;
   com[1] = 0x04;

   com[15] = m_fanLvl;
   com[17] = lvl;
   
   //Disable most stuff.
   for(size_t n = 20; n <m_protocolChars-1; ++n) com[n] = 0xAA;
   
   //Preserve units
   if(m_protocolChars > 43)
   {
      com[44] = 0;
      com[45] = 0x0001;
      com[46] = 0;
      com[47] = 0;
      com[48] = 0;
      com[49] = 0;
   }
   
   int checksum = 0;
   for(size_t n = 0; n <m_protocolChars-1; ++n) checksum += com[n];
   com[m_protocolChars-1] = checksum % 0x64;
   
   rv = write(m_fileDescrip, com.data(), com.size());
   
   if(rv < 0)
   {
      log<software_error>({__FILE__,__LINE__, errno, "error from write"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }
   
   log<text_log>("set pump level to " + std::to_string(lvl));
   
   return 0;
}

int koolanceCtrl::setFanLvl( int lvl )
{
   if(m_protocolChars == 43) return 0;
   
   int rv;
   
   std::vector<unsigned char> com;
   com.resize(m_protocolChars, '\0');
   com[0] = 0xCF;
   com[1] = 0x04;

   com[15] = lvl;
   com[17] = m_pumpLvl;
   
   //Disable most stuff.
   for(size_t n = 20; n <m_protocolChars-1; ++n) com[n] = 0xAA;
   
   //Preserve units
   if(m_protocolChars > 43)
   {
      com[44] = 0;
      com[45] = 0x0001;
      com[46] = 0;
      com[47] = 0;
      com[48] = 0;
      com[49] = 0;
   }
   
   int checksum = 0;
   for(size_t n = 0; n <m_protocolChars-1; ++n) checksum += com[n];
   com[m_protocolChars-1] = checksum % 0x64;
   
   rv = write(m_fileDescrip, com.data(), com.size());
   
   if(rv < 0)
   {
      log<software_error>({__FILE__,__LINE__, errno, "error from write"});
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
      state(stateCodes::NOTCONNECTED);
      
      return -1;
   }
   
   log<text_log>("set fan level to " + std::to_string(lvl));
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(koolanceCtrl, m_indiP_pumplvl)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_pumplvl.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }

   int lvl = -1;
   
   if( ipRecv.find("current") )
   {
      lvl = ipRecv["current"].get<int>();
   }
   
   if( ipRecv.find("target") )
   {
      lvl = ipRecv["target"].get<int>();
   }

   if(lvl < 1 || lvl > 10)
   {
      log<software_error>({__FILE__,__LINE__, "Pump level out of range"});
      return 0;
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   updateIfChanged(m_indiP_pumplvl, "target", lvl);
   return setPumpLvl(lvl);
   

}

INDI_NEWCALLBACK_DEFN(koolanceCtrl, m_indiP_fanlvl)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_fanlvl.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }

   
   int lvl = -1;
   
   if( ipRecv.find("current") )
   {
      lvl = ipRecv["current"].get<int>();
   }
   
   if( ipRecv.find("target") )
   {
      lvl = ipRecv["target"].get<int>();
   }

   if(lvl < 0 || lvl > 100)
   {
      log<software_error>({__FILE__,__LINE__, "Fan level out of range"});
      return 0;
   }
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   updateIfChanged(m_indiP_fanlvl, "target", lvl);
   return setFanLvl(lvl);
   
   return 0;
}

inline
int koolanceCtrl::checkRecordTimes()
{
   return telemeter<koolanceCtrl>::checkRecordTimes(telem_cooler());
}
   
inline
int koolanceCtrl::recordTelem( const telem_cooler * )
{
   return recordCooler(true);
}
 
inline
int koolanceCtrl::recordCooler( bool force )
{
   static float last_liqTemp = std::numeric_limits<float>::max();
   static float last_flowRate = std::numeric_limits<float>::max();
   static int last_pumpLvl = std::numeric_limits<int>::max();
   static int last_pumpRPM = std::numeric_limits<int>::max();
   static int last_fanRPM = std::numeric_limits<int>::max();
   static int last_fanLvl = std::numeric_limits<int>::max();
   
   if( m_liqTemp != last_liqTemp || m_flowRate != last_flowRate || m_pumpLvl != last_pumpLvl ||
          m_pumpRPM != last_pumpRPM || m_fanRPM != last_fanRPM || m_fanLvl != last_fanLvl || force )
   {
      telem<telem_cooler>({m_liqTemp, m_flowRate, (uint8_t) m_pumpLvl, (uint16_t) m_pumpRPM, (uint8_t) m_fanLvl, (uint16_t) m_fanRPM});
      
      last_liqTemp = m_liqTemp;
      last_flowRate = m_flowRate;
      last_pumpLvl = m_pumpLvl;
      last_pumpRPM = m_pumpRPM;
      last_fanLvl = m_fanLvl;
      last_fanRPM = m_fanRPM;
   }
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //koolanceCtrl_hpp
