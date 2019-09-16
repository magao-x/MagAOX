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
  * <a href="../handbook/apps/filterWheelCtrl.html">Application Documentation</a>
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

/** MagAO-X application to control a Faulhaber MCBL controlled filter wheel.
  *
  * \todo add temperature monitoring
  * \todo add INDI props to md doc
  * \todo should move in least time direction, rather than always in the same direction.
  * \todo add tests
  * \todo should be an iodevice
  *
  * \ingroup filterWheelCtrl
  */
class filterWheelCtrl : public MagAOXApp<>, public tty::usbDevice, public dev::stdMotionStage<filterWheelCtrl>
{

   friend class dev::stdMotionStage<filterWheelCtrl>;
   
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

   

   ///@}

   /** \name Status
     * @{
     */

   bool m_switch{false}; ///< The home switch status
   
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

public:
   INDI_NEWCALLBACK_DECL(filterWheelCtrl, m_indiP_counts);

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

};

inline
filterWheelCtrl::filterWheelCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_presetNotation = "filter"; //sets the name of the configs, etc.
   
   m_powerMgtEnabled = true;
   
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
   config.add("stage.homeOffset", "", "stage.homeOffset", argType::Required, "stage", "homeOffset", false, "long", "The homing offset in motor counts.");
   
   dev::stdMotionStage<filterWheelCtrl>::setupConfig(config);
   
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
   config(m_homeOffset, "stage.homeOffset");
   

   dev::stdMotionStage<filterWheelCtrl>::loadConfig(config);

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
   createStandardIndiNumber<long>( m_indiP_counts, "counts", std::numeric_limits<long>::lowest(), std::numeric_limits<long>::max(), 0.0, "%ld");
   registerIndiPropertyNew( m_indiP_counts, INDI_NEWCALLBACK(m_indiP_counts)) ;

   
   dev::stdMotionStage<filterWheelCtrl>::appStartup();
   
   

   
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

   if( state() == stateCodes::POWERON )
   {
      if(m_deviceName == "") state(stateCodes::NODEVICE);
      else
      {
         state(stateCodes::NOTCONNECTED);
         std::stringstream logs;
         logs << "USB Device " << m_idVendor << ":" << m_idProduct << ":" << m_serial << " found in udev as " << m_deviceName;
         log<text_log>(logs.str());
      }
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
         sleep(1); //wait to see if power state updates 
         if(m_powerState == 0) return 0;
         
         //Ok we can't figure this out, die.
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

         if(startHoming()<0)
         {
            state(stateCodes::ERROR);
            log<software_error>({__FILE__,__LINE__});
            return 0;
         }
      }
      else state(stateCodes::NOTHOMED);


   }

   if( state() == stateCodes::NOTHOMED || state() == stateCodes::READY || state() == stateCodes::OPERATING || state() == stateCodes::HOMING)
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
         if(state() == stateCodes::OPERATING) 
         {
            if(m_movingState == 1)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               if(moveToRawRelative(-20000) < 0)
               {
                  sleep(1);
                  if(m_powerState == 0) return 0;
                  
                  state(stateCodes::ERROR);
                  return log<software_error,0>({__FILE__,__LINE__});
               }

               m_movingState=2;
            }
            else if(m_movingState == 2)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               if(moveTo(m_preset_target) < 0)
               {
                  sleep(1);
                  if(m_powerState == 0) return 0;
                  
                  state(stateCodes::ERROR);
                  return log<software_error,0>({__FILE__,__LINE__});
               }

               m_movingState=3;
            }
            else
            {
               m_movingState = 0;
               state(stateCodes::READY); //stopped moving but was just changing pos
            }
            
         }
         else if (state() == stateCodes::HOMING) //stopped moving but was in the homing sequence
         {
            if(m_homingState == 1)
            {
               std::lock_guard<std::mutex> guard(m_indiMutex);
               if(moveToRawRelative(-50000) < 0)
               {
                  sleep(1);
                  if(m_powerState == 0) return 0;
                  
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
                  sleep(1);
                  if(m_powerState == 0) return 0;
                  
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
                  sleep(1);
                  if(m_powerState == 0) return 0;
                  
                  state(stateCodes::ERROR);
                  return log<software_error,0>({__FILE__,__LINE__});
               }
               m_homingState=4;
            }
            else
            {
               m_homingState = 0;
               state(stateCodes::READY);
               
               m_preset_target = ((double) m_rawPos-m_homeOffset)/m_circleSteps*m_presetNames.size() + 1.0;
            }
         }
      }

      std::lock_guard<std::mutex> guard(m_indiMutex);
      
      if(m_moving)
      {
         updateIfChanged(m_indiP_counts, "current", m_rawPos, INDI_BUSY);
      }
      else
      {
         updateIfChanged(m_indiP_counts, "current", m_rawPos, INDI_IDLE);
      }
      
      m_preset = ((double) m_rawPos-m_homeOffset)/m_circleSteps*m_presetNames.size() + 1.0;
      
      stdMotionStage<filterWheelCtrl>::updateINDI();
      
      return 0;
   }

   if(state() == stateCodes::ERROR)
   {
      sleep(1);
      if(m_powerState == 0) return 0;
                  
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
      
      
      double counts = -1;
      double target_abs = -1;

      if(ipRecv.find("current"))
      {
         counts = ipRecv["current"].get<double>();
      }

      if(ipRecv.find("target"))
      {
         target_abs = ipRecv["target"].get<double>();
      }

      if(target_abs == -1) target_abs = counts;
      
      
      m_preset_target = ((double) target_abs - m_homeOffset)/m_circleSteps*m_presetNames.size() + 1.0;

      std::lock_guard<std::mutex> guard(m_indiMutex);
      return moveToRaw(target_abs);
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

int filterWheelCtrl::startHoming()
{
   m_homingState = 1;
   updateSwitchIfChanged(m_indiP_home, "request", pcf::IndiElement::Off, INDI_IDLE);
   return home();
}

int filterWheelCtrl::presetNumber()
{
   //First we calculate current preset name
   int n = floor(m_preset + 0.5) - 1;
   if(n < 0)
   {
      while(n < 0) n += m_presetNames.size();
   }
   if( n > (long) m_presetNames.size()-1 )
   {
      while( n > (long) m_presetNames.size()-1 ) n -= m_presetNames.size();
   }
   
   if( n < 0)
   {
      log<software_error>({__FILE__,__LINE__, "error calculating " + m_presetNotation + " index, n < 0"});
      return -1;
   }
      
   return n;
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

int filterWheelCtrl::stop()
{
   m_homingState = 0;
   //First try without locking
   tty::ttyWrite( "DI\r", m_fileDescrip, m_writeTimeOut);

   //Now make sure it goes through
   std::lock_guard<std::mutex> guard(m_indiMutex);
   int rv = tty::ttyWrite( "DI\r", m_fileDescrip, m_writeTimeOut);

   updateSwitchIfChanged(m_indiP_stop, "request", pcf::IndiElement::Off, INDI_IDLE);
   
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
   
   updateIfChanged(m_indiP_counts, "target", counts, pcf::IndiProperty::Busy);
    
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

   if(m_circleSteps ==0 || m_presetNames.size() == 0) counts = filters;
   else counts = m_homeOffset + m_circleSteps/m_presetNames.size() * (filters-1);

   return moveToRaw(counts);

}

} //namespace app
} //namespace MagAOX

#endif //filterWheelCtrl_hpp
