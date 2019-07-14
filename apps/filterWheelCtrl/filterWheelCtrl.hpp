/** \file filterWheelCtrl.hpp
  * \brief The MagAO-X Filter Wheel Controller
  *
  * \ingroup filterWheelCtrl_files
  */


#ifndef filterWheelCtrl_hpp
#define filterWheelCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


/** \defgroup filterWheelCtrl Filter Wheel Control
  * \brief Control of MagAO-X MCBL-based f/w.
  * 
  * \link page_module_filterWheelCtrl Application Documentation
  * 
  * \ingroup apps
  * 
  */

/** \defgroup filterWheelCtrl_files Filter Wheel Control Files
  * \ingroup filterWheelCtrl
  */

namespace MagAOX
{
namespace app
{

/** MagAO-X application to control a filter wheel.
  *
  * \todo add temperature monitoring
  * \todo add power monitoring
  * \todo add INDI props to md doc
  * \todo should move in least time direction, rather than always in the same direction.
  * \todo add tests
  * 
  * \ingroup filterWheelCtrl
  */
class filterWheelCtrl : public MagAOXApp<>, public tty::usbDevice
{

protected:

   /** \name Non-configurable parameters
     *@{
     */
   
   int m_motorType {2};
   
   ///@}
   
   /** \name Configurable Parameters
     * @{
     */

   int m_writeTimeOut {1000};  ///< The timeout for writing to the device [msec].
   int m_readTimeOut {1000}; ///< The timeout for reading from the device [msec].
   
   double m_acceleration {1000};
   double m_motorSpeed {1000};
   
   long m_circleSteps {0}; ///< The number of position counts in 1 360-degree revolution.
   long m_homeOffset {0}; ///< The number of position counts to offset from the home position
   
   bool m_powerOnHome {false}; ///< If true, then the motor is homed at startup (software or actual power on)
   
   std::vector<std::string> m_filterNames; ///< The names of each position in the wheel.
   std::vector<double> m_filterPositions; ///< The positions, in filter units, of each filter.  If 0, then the integer position number is used to calculate.
   
   ///@}
   
   /** \name Status
     * @{
     */

   bool m_switch{false}; ///< The home switch status
   int m_moving {0}; ///< Whether or not the wheel is moving.
   long m_rawPos {0}; ///< The position of the wheel in motor coutns.
   
   int m_homingState{0}; ///< The homing state, tracks the stages of homing.
   ///@}


public:

   /// Default c'tor.
   filterWheelCtrl();

   /// D'tor, declared and defined for noexcept.
   ~filterWheelCtrl() noexcept
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



protected:

   //declare our properties   
   
   ///The position of the wheel in counts
   pcf::IndiProperty m_indiP_counts;

   ///The position of the wheel in filters
   pcf::IndiProperty m_indiP_filters;

   ///The name of the nearest filter for this position   
   pcf::IndiProperty m_indiP_filterName;

   ///Command the wheel to home.  Any change in this property causes a home.
   pcf::IndiProperty m_indiP_req_home;
   
   ///Command the wheel to halt.  Any change in this property causes an immediate halt.
   pcf::IndiProperty m_indiP_req_halt;
   
public:
   INDI_NEWCALLBACK_DECL(filterWheelCtrl, m_indiP_counts);
   INDI_NEWCALLBACK_DECL(filterWheelCtrl, m_indiP_filters);
   INDI_NEWCALLBACK_DECL(filterWheelCtrl, m_indiP_filterName);
   
   INDI_NEWCALLBACK_DECL(filterWheelCtrl, m_indiP_req_home);
   INDI_NEWCALLBACK_DECL(filterWheelCtrl, m_indiP_req_halt);

protected:
   //Each of these should have m_indiMutex locked before being called.
   
   /// Set up the MCBL controller, called after each power-on/connection
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */ 
   int onPowerOnConnect();
   
   /// Get the home switch status, sets m_switch to true or false.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int getSwitch();

   /// Get the moving-state of the wheel, sets m_moving to true or false.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int getMoving();
   
   /// Get the current position of the wheel, sets m_rawPos to the current motor counts.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int getPos();
   
   /// Start a homing sequence.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int home();
   
   /// Halt the wheel motion immediately.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int halt();
   
   /// Move to an absolute position in raw counts.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveToRaw( const long & counts /**< [in] The new position in absolute motor counts*/);
   
   /// Move to a new position relative to current, in raw counts.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveToRawRelative( const long & counts /**< [in] The new position in relative motor counts*/ );
   
   /// Move to an absolute position in filter units.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveTo( const double & filters /**< [in] The new position in absolute filter units*/ );

   /// Move to a new position relative to current, filter units.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveToRelative( const double & filters /**< [in] The new position in relative filter units*/ );
   
   /// Move to a new position, based on precendence of each possible source of position.
   /** The precedence order is:
     *  # absolute filters
     *  # relative filters
     *  # absolute counts
     *  # relative counts
     *
     * That is, it moves based on the first valid position in the list.  To be valid, an absolute position must not 
     * be -1, and a relative position must not be 0.
     * 
     * This primarily intended for processing the req_position INDI property.
     * 
     * \returns 0 on success
     * \returns -1 on error
     * 
     */
   int moveTo( const double & filters,          ///< [in] The new position in absolute filters.  Only valid if not -1.
               const double & filters_relative, ///< [in] The new position in relative filters.  Only valid if not 0.
               const double & counts,           ///< [in] The new position in absolute counts.  Only valid if not -1.
               const double & counts_relative   ///< [in] The new position in relative counts.  Only valid if not 0.
             );
   
   /// Move to a new filter by name.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int moveTo( const std::string & name /**< [in] The name of the filter to move to*/);
   
};

inline
filterWheelCtrl::filterWheelCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

inline
void filterWheelCtrl::setupConfig()
{
   
   tty::usbDevice::setupConfig(config);
   
   config.add("timeouts.write", "", "timeouts.write", argType::Required, "timeouts", "write", false, "int", "The timeout for writing to the device [msec]. Default = 1000");
   config.add("timeouts.read", "", "timeouts.read", argType::Required, "timeouts", "read", false, "int", "The timeout for reading the device [msec]. Default = 1000");
   
   config.add("motor.acceleration", "", "motor.acceleration", argType::Required, "motor", "acceleration", false, "real", "The motor acceleration parameter. Default=1000.");
   config.add("motor.speed", "", "motor.speed", argType::Required, "motor", "speeed", false, "real", "The motor speed parameter.  Default=1000.");
   config.add("motor.circleSteps", "", "motor.circleSteps", argType::Required, "motor", "circleSteps", false, "long", "The number of steps in 1 revolution.");
   config.add("motor.homeOffset", "", "motor.homeOffset", argType::Required, "motor", "homeOffset", false, "long", "The homing offset in motor counts.");
   config.add("motor.powerOnHome", "", "motor.powerOnHome", argType::Required, "motor", "powerOnHome", false, "bool", "If true, home at startup/power-on.  Default=false.");
   
   config.add("filters.names", "", "filters.names",  argType::Required, "filters", "names", false, "vector<string>", "The names of the filters.");
   config.add("filters.positions", "", "filters.positions",  argType::Required, "filters", "positions", false, "vector<double>", "The positions of the filters.  If omitted or 0 then order is used.");
}

inline
void filterWheelCtrl::loadConfig()
{
   this->m_baudRate = B9600; //default for MCBL controller.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(config);
   
   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND) //Ignore error if not plugged in
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   config(m_writeTimeOut, "timeouts.write");
   config(m_readTimeOut, "timeouts.read");
   
   config(m_acceleration, "motor.acceleration");
   config(m_motorSpeed, "motor.speed");
   config(m_circleSteps, "motor.circleSteps");
   config(m_homeOffset, "motor.homeOffset");
   config(m_powerOnHome, "motor.powerOnHome");
   
   config(m_filterNames, "filters.names");
   m_filterPositions.resize(m_filterNames.size(), 0);
   for(size_t n=0;n<m_filterPositions.size();++n) m_filterPositions[n] = n+1;
   config(m_filterPositions, "filters.positions");
   for(size_t n=0;n<m_filterPositions.size();++n) if(m_filterPositions[n] == 0) m_filterPositions[n] = n+1;
   
}

inline
int filterWheelCtrl::appStartup()
{
   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_counts, "counts", pcf::IndiProperty::Number);
   m_indiP_counts.add (pcf::IndiElement("current"));
   m_indiP_counts.add (pcf::IndiElement("target"));
   m_indiP_counts.add (pcf::IndiElement("target_rel"));
   m_indiP_counts["current"].set(-1);
   m_indiP_counts["target"].set(-1);
   m_indiP_counts["target_rel"].set(0);
   
   REG_INDI_NEWPROP(m_indiP_filters, "filters", pcf::IndiProperty::Number);
   m_indiP_filters.add (pcf::IndiElement("current"));
   m_indiP_filters.add (pcf::IndiElement("target"));
   m_indiP_filters.add (pcf::IndiElement("target_rel"));
   m_indiP_filters["current"].set(-1);
   m_indiP_filters["target"].set(-1);
   m_indiP_filters["target_rel"].set(0);
   
   REG_INDI_NEWPROP(m_indiP_filterName, "filterName", pcf::IndiProperty::Number);
   m_indiP_filterName.add (pcf::IndiElement("current"));
   m_indiP_filterName.add (pcf::IndiElement("target"));
   
   
   REG_INDI_NEWPROP(m_indiP_req_home, "req_home", pcf::IndiProperty::Number);
   m_indiP_req_home.add (pcf::IndiElement("home"));
   
   REG_INDI_NEWPROP(m_indiP_req_halt, "req_halt", pcf::IndiProperty::Number);
   m_indiP_req_halt.add (pcf::IndiElement("halt"));
   
   //Get the USB device if it's in udev
   if(m_deviceName == "") state(stateCodes::NODEVICE);
   else
   {
      state(stateCodes::NOTCONNECTED);
      std::stringstream logs;
      logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
      log<text_log>(logs.str());
   }
   
   return 0;
}

inline
int filterWheelCtrl::appLogic()
{
   if( state() == stateCodes::INITIALIZED )
   {
      log<text_log>( "In appLogic but in state INITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }

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
         state(stateCodes::NOTCONNECTED);
         if(!stateLogged())
         {
            std::stringstream logs;
            logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
            log<text_log>(logs.str());
         }
      }
   }
   
   if( state() == stateCodes::NOTCONNECTED )
   {
      euidCalled();
      int rv = connect();
      euidReal();
      
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
         state(stateCodes::FAILURE);
         if(!stateLogged()) log<software_error>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
         return -1;
                           
      }
         
      
      
      if( getPos() == 0 ) state(stateCodes::CONNECTED);
      else
      {
         return 0;
      }

      if(state() == stateCodes::CONNECTED)
      {
         std::stringstream logs;
         logs << "Connected to filter wheel on " << m_deviceName;
         log<text_log>(logs.str());
      }

   }
   
   if( state() == stateCodes::CONNECTED )
   {
      int rv = onPowerOnConnect();
      
      if(rv < 0)
      {
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         
         return 0;
      }
         
      std::string landfill;
      tty::ttyRead(landfill, "\r", m_fileDescrip, m_readTimeOut); //read to timeout to kill any missed chars.
      
      if(m_powerOnHome)
      {
         std::lock_guard<std::mutex> guard(m_indiMutex);
      
         m_homingState = 1;
         
         if(home()<0)
         {
            state(stateCodes::ERROR);
            log<software_error>({__FILE__,__LINE__});
            return 0;
         }
      }
      else state(stateCodes::READY);
      
      
   }
   
   if( state() == stateCodes::READY || state() == stateCodes::OPERATING || state() == stateCodes::HOMING)
   {
      { //mutex scope
         //Make sure we have exclusive attention of the device
         std::lock_guard<std::mutex> guard(m_indiMutex);  //Lock the mutex before conducting any communications.

         int rv = getSwitch(); 
      
         if(rv  != 0 )
         {
            state(stateCodes::NOTCONNECTED);
            return 0;
         }
         
         rv = getMoving(); 
      
         if(rv  != 0 )
         {
            state(stateCodes::NOTCONNECTED);
            return 0;
         }
         
         rv = getPos(); 
      
         if(rv  != 0 )
         {
            state(stateCodes::NOTCONNECTED);
            return 0;
         }
      }
      
      if(m_moving)
      {
         //Started moving but we don't know yet.
         if(state() == stateCodes::READY) state(stateCodes::OPERATING);
      }
      else
      {
         if(state() == stateCodes::OPERATING) state(stateCodes::READY); //stopped moving but was just changing pos
         else if (state() == stateCodes::HOMING) //stopped moving but was in the homing sequence
         {
            if(m_homingState == 1)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex); 
               if(moveToRawRelative(-50000) < 0)
               {
                  state(stateCodes::ERROR);
                  return log<software_error,0>({__FILE__,__LINE__});
               }
               
               m_homingState=2;
            }
            else if (m_homingState == 2)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               if(home()<0)
               {
                  state(stateCodes::ERROR);
                  return log<software_error,0>({__FILE__,__LINE__});
               }
               m_homingState = 3;
            }
            else if(m_homingState == 3)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex); 
               if(moveToRaw(m_homeOffset)<0)
               {
                  state(stateCodes::ERROR);
                  return log<software_error,0>({__FILE__,__LINE__});
               }
               m_homingState=4;
            }
            else
            {
               m_homingState = 0;
               state(stateCodes::READY);
            }
         }
      }
      
      
      updateIfChanged(m_indiP_counts, "current", m_rawPos);
      
      double filPos = ((double) m_rawPos-m_homeOffset)/m_circleSteps*m_filterNames.size() + 1.0;
      updateIfChanged(m_indiP_filters, "current", filPos);
      
      int nfilPos = fmod(filPos-1+0.5, m_filterNames.size()) ;
      if(nfilPos > (long) m_filterNames.size()) nfilPos -= m_filterNames.size();
      if(nfilPos < 0) nfilPos += m_filterNames.size();
      
      updateIfChanged(m_indiP_filterName, "current", m_filterNames[nfilPos]);
   
      return 0;
   }
   
   if(state() == stateCodes::ERROR)
   {
      return log<software_error,-1>({__FILE__,__LINE__, "In state ERROR but no recovery implemented.  Terminating."});
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "appLogic fell through.  Terminating."});

}



inline
int filterWheelCtrl::appShutdown()
{
   //don't bother
   return 0;
}

INDI_NEWCALLBACK_DEFN(filterWheelCtrl, m_indiP_counts)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_counts.getName())
   {
      std::cerr << "counts\n";
      double counts = -1;
      double target_abs = -1;
      double target_rel = 0;
      
      if(ipRecv.find("current"))
      {
         counts = ipRecv["current"].get<double>();
      }
      
      if(ipRecv.find("target"))
      {
         target_abs = ipRecv["target"].get<double>();
      }

      if(target_abs == -1) target_abs = counts;
      
      if(ipRecv.find("target_rel"))
      {
         target_rel = ipRecv["target_rel"].get<double>();
      }
      
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      updateIfChanged(m_indiP_counts, "target", target_abs);
      updateIfChanged(m_indiP_counts, "target_rel", target_rel);
      
      return moveTo( -1, 0, target_abs, target_rel );
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(filterWheelCtrl, m_indiP_filters)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_filters.getName())
   {
      double filters = -1;
      double target_abs = -1;
      double target_rel = 0;
      
      if(ipRecv.find("current"))
      {
         filters = ipRecv["current"].get<double>();
      }
      
      if(ipRecv.find("target"))
      {
         target_abs = ipRecv["target"].get<double>();
      }

      if(target_abs == -1) target_abs = filters;
      
      if(ipRecv.find("target_rel"))
      {
         target_rel = ipRecv["target_rel"].get<double>();
      }
      
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      updateIfChanged(m_indiP_filters, "target", target_abs);
      updateIfChanged(m_indiP_filters, "target_rel", target_rel);
      
      return moveTo( target_abs, target_rel, -1, 0 );
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(filterWheelCtrl, m_indiP_filterName)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_filterName.getName())
   {
      std::string name;
      std::string target;
      
      if(ipRecv.find("current"))
      {
         name = ipRecv["current"].get();
      }
      
      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get();
      }

      if(target == "") target = name;
      
      
      if(target == "") return 0;
      
      size_t n;
      for(n=0; n< m_filterNames.size(); ++n) if( m_filterNames[n] == target ) break;
      
      if(n >= m_filterNames.size()) return -1;
      
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      updateIfChanged(m_indiP_filterName, "target", target);
      
      return moveTo(n+1);
      
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(filterWheelCtrl, m_indiP_req_home)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_req_home.getName())
   {
      
      if(state() == stateCodes::HOMING) return 0; //Don't restart while homing already.
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      m_homingState = 1;
      return home();
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(filterWheelCtrl, m_indiP_req_halt)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_req_halt.getName())
   {
      halt(); //Try immediately without locking
      
      //Now lock to make sure we get uninterrupted attention
      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      return halt();
   }
   return -1;
}

int filterWheelCtrl::onPowerOnConnect()
{
   std::string com;
   
   int rv;
   
   std::lock_guard<std::mutex> guard(m_indiMutex);
   
   rv = tty::ttyWrite( "ANSW0\r", m_fileDescrip, m_writeTimeOut); //turn off replies and asynchronous comms.
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   //Send motor type 
   com = "MOTTYP" + std::to_string(m_motorType) + "\r";
   rv = tty::ttyWrite( com, m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   //Set up acceleration and speed.
   com = "AC" + std::to_string(m_acceleration) + "\r";
   rv = tty::ttyWrite( com, m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   com = "SP" + std::to_string(m_motorSpeed) + "\r";
   rv = tty::ttyWrite( com, m_fileDescrip,m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   return 0;
}

int filterWheelCtrl::getSwitch()
{
   int rv;
   std::string resp;
   
   rv = tty::ttyWriteRead( resp, "GAST\r", "\r\n", false, m_fileDescrip, m_writeTimeOut, m_readTimeOut);

   if(rv == 0) 
   {

      if(resp == "1011") m_switch=true;
      else m_switch=false;
      
      return 0;
   }

   
   return rv;
   
}

int filterWheelCtrl::getMoving()
{
   int rv;
   std::string resp;
   
   rv = tty::ttyWriteRead( resp, "GN\r", "\r\n", false, m_fileDescrip, m_writeTimeOut, m_readTimeOut);

   if(rv == 0) 
   {
      int speed;
      try{ speed = std::stol(resp.c_str());}
      catch(...){speed=0;}
      
      if(fabs(speed) > 0.1*m_motorSpeed) m_moving = true;
      else m_moving = false;
      
      return 0;
   }

   
   return rv;
   
}

int filterWheelCtrl::getPos()
{
   int rv;
   std::string resp;
   
   rv = tty::ttyWriteRead( resp, "POS\r", "\r\n", false, m_fileDescrip, m_writeTimeOut, m_readTimeOut);

   if(rv == 0) 
   {
      try{ m_rawPos = std::stol(resp.c_str());}
      catch(...){m_rawPos=0;}
   }

   
   return rv;
   
}

int filterWheelCtrl::home()
{
   state(stateCodes::HOMING);

   int rv;
      
   rv = tty::ttyWrite( "EN\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "HA4\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "HL4\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "CAHOSEQ\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "HP0\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   std::string com = "HOSP" + std::to_string(m_motorSpeed) + "\r";
   rv = tty::ttyWrite( com, m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "GOHOSEQ\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   

   return 0;
}

int filterWheelCtrl::halt()
{
   m_homingState = 0;
   int rv = tty::ttyWrite( "DI\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   return 0;
}

int filterWheelCtrl::moveToRaw( const long & counts )
{

   std::string com;
   int rv;
   
   rv = tty::ttyWrite( "EN\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   com = "LA" + std::to_string(counts) + "\r";
   rv = tty::ttyWrite( com, m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "M\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   
   return 0;
}

int filterWheelCtrl::moveToRawRelative( const long & counts_relative )
{
      
   std::string com;
   int rv;
   
   rv = tty::ttyWrite( "EN\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   com = "LR" + std::to_string(counts_relative) +"\r";
   rv = tty::ttyWrite( com, m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   
   rv = tty::ttyWrite( "M\r", m_fileDescrip, m_writeTimeOut);
   if(rv < 0) return log<software_error,-1>({__FILE__,__LINE__,rv, tty::ttyErrorString(rv)});
   

   
   return 0;
}

int filterWheelCtrl::moveTo( const double & filters )
{
   long counts;
   
   if(m_circleSteps ==0 || m_filterNames.size() == 0) counts = filters;
   else counts = m_homeOffset + m_circleSteps/m_filterNames.size() * (filters-1);
   
   return moveToRaw(counts);

}

int filterWheelCtrl::moveToRelative( const double & filters_relative )
{
   long counts_relative;
   
   if(m_circleSteps ==0 || m_filterNames.size() == 0) counts_relative = filters_relative;
   else counts_relative = m_circleSteps/m_filterNames.size() * filters_relative;
   
   return moveToRawRelative(counts_relative);
 
}

int filterWheelCtrl::moveTo( const double & filters,
                             const double & filters_relative,
                             const double & counts,
                             const double & counts_relative
                           )
{
   if( filters != -1 )
   {
      return moveTo( filters );
   }
   
   if( counts != -1 )
   {
      return moveToRaw( counts );
   }
   
   if( filters_relative != 0)
   {
      return moveToRelative( filters_relative );
   }
   
   if( counts_relative != 0)
   {
      return moveToRawRelative( counts_relative );
   }
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //filterWheelCtrl_hpp
