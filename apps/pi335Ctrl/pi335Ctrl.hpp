/** \file pi335Ctrl.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup pi335Ctrl_files
  */

#ifndef pi335Ctrl_hpp
#define pi335Ctrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup pi335Ctrl 
  * \brief The XXXXXXX application to do YYYYYYY
  *
  * <a href="..//apps_html/page_module_pi335Ctrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup pi335Ctrl_files
  * \ingroup pi335Ctrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup pi335Ctrl
  */
class pi335Ctrl : public MagAOXApp<true> , public tty::usbDevice, public dev::ioDevice, public dev::dm<pi335Ctrl,float>,  public dev::shmimMonitor<pi335Ctrl>
{

   //Give the test harness access.
   friend class pi335Ctrl_test;

   friend class dev::dm<pi335Ctrl,float>;
   
   friend class dev::shmimMonitor<pi335Ctrl>;
   
protected:

   /** \name Configurable Parameters
     *@{
     */  
   float m_homePos1 {17.5}; ///< Home position of axis 1.  Default is 17.5
   float m_homePos2 {17.5}; ///< Home position of axis 2.  Default is 17.5
   
   ///@}

   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.

   double m_homingStart {0};
   int m_homingState {0};
   
   int m_servoState {0};

   float m_pos1 {0};
   float m_pos2 {0};
   
public:
   /// Default c'tor.
   pi335Ctrl();

   /// D'tor, declared and defined for noexcept.
   ~pi335Ctrl() noexcept
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

   /// Implementation of the FSM for pi335Ctrl.
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

   /// Test the connection to the device
   /** Uses the *IDN? query
     * 
     * \returns 0 if the E227 is found
     * \returns -1 on an error
     */ 
   int testConnection();
   
   int initDM();
   
   int home();
   
   int home_1();
   
   int home_2();
   
   int finishInit();
   
    /// Zero all commands on the DM
   /** This does not update the shared memory buffer.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */
   int zeroDM() { return 0;}
   
   /// Send a command to the DM
   /** This is called by the shmim monitoring thread in response to a semaphore trigger.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */
   int commandDM(void * curr_src) 
   { 
      static_cast<void>(curr_src);
      return 0;
      
   }
   
   /// Release the DM, making it safe to turn off power.
   /** The application will be state READY at the conclusion of this.
     *  
     * \returns 0 on success 
     * \returns -1 on error
     */
   int releaseDM();
   
   int setCom( const std::string & com );
   
   int setCom( const std::string & com,
               int axis
             );
   
   int setCom( const std::string & com,
               int axis,
               const std::string & arg
             );
   
   int getCom( std::string & resp,
               const std::string & com,
               int axis
             );
   
   int getPos( float & pos,
               int n 
             );
   
   int getMov( float & mov,
               int n
             );
   
   int move_1( float absPos );
   
   int move_2( float absPos );
  
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_pos1;
   pcf::IndiProperty m_indiP_pos2;

public:
   INDI_NEWCALLBACK_DECL(pi335Ctrl, m_indiP_pos1);
   INDI_NEWCALLBACK_DECL(pi335Ctrl, m_indiP_pos2);
   
};

pi335Ctrl::pi335Ctrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   
   return;
}

void pi335Ctrl::setupConfig()
{
   dev::ioDevice::setupConfig(config);
   tty::usbDevice::setupConfig(config);
   dev::dm<pi335Ctrl,float>::setupConfig(config);
   
   config.add("stage.homePos1", "", "stage.homePos1", argType::Required, "stage", "homePos1", false, "float", "Home position of axis 1.  Default is 17.5.");
   config.add("stage.homePos2", "", "stage.homePos2", argType::Required, "stage", "homePos2", false, "float", "Home position of axis 2.  Default is 17.5.");
}

int pi335Ctrl::loadConfigImpl( mx::app::appConfigurator & _config )
{

   this->m_baudRate = B115200; //default for E727 controller.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(_config);
   
   if(rv != 0 && rv != TTY_E_NODEVNAMES && rv != TTY_E_DEVNOTFOUND )
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   dev::ioDevice::loadConfig(_config);
   
   dev::dm<pi335Ctrl,float>::loadConfig(_config);
   
   config(m_homePos1, "stage.homePos1");
   config(m_homePos2, "stage.homePos2");
   
   return 0;
}

void pi335Ctrl::loadConfig()
{
   loadConfigImpl(config);
}

int pi335Ctrl::appStartup()
{
   if( state() == stateCodes::UNINITIALIZED )
   {
      log<text_log>( "In appStartup but in state UNINITIALIZED.", logPrio::LOG_CRITICAL );
      return -1;
   }
   
   ///\todo promote usbDevice to dev:: and make this part of its appStartup
   //Get the USB device if it's in udev
   if(m_deviceName != "") 
   {
      log<text_log>(std::string("USB Device ") + m_idVendor + ":" + m_idProduct + ":" + m_serial + " found in udev as " + m_deviceName);
   }
   
   ///\todo error checks here
   dev::dm<pi335Ctrl,float>::appStartup();
   shmimMonitor<pi335Ctrl>::appStartup();
   
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_pos1, "pos_1", pcf::IndiProperty::Number);
   m_indiP_pos1.add (pcf::IndiElement("current"));
   m_indiP_pos1.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP(m_indiP_pos2, "pos_2", pcf::IndiProperty::Number);
   m_indiP_pos2.add (pcf::IndiElement("current"));
   m_indiP_pos2.add (pcf::IndiElement("target"));
   
   //std::cerr << "appStartup complete \n";
   return 0;
}

int pi335Ctrl::appLogic()
{
   dev::dm<pi335Ctrl,float>::appLogic();
   shmimMonitor<pi335Ctrl>::appLogic();
   
   if(state() == stateCodes::POWERON)
   {
      if(!powerOnWaitElapsed()) 
      {
         return 0;
      }
      else
      {
         ///\todo promote usbDevice to dev:: and make this part of its appStartup
         //Get the USB device if it's in udev
         if(m_deviceName == "") 
         {
            state(stateCodes::NODEVICE);
         }
         else
         {
            state(stateCodes::NOTCONNECTED);
         }
      }  
   }
      
   ///\todo promote usbDevice to dev:: and make this part of its appLogic
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
            log<text_log>(std::string("USB Device ") + m_idVendor + ":" + m_idProduct + ":" + m_serial + " not found in udev");
         }
         return 0;
      }
      else
      {
         state(stateCodes::NOTCONNECTED);
         if(!stateLogged())
         {
            log<text_log>(std::string("USB Device ") + m_idVendor + ":" + m_idProduct + ":" + m_serial + " found in udev as " + m_deviceName);
         }
      }
   }
   
   if( state() == stateCodes::NOTCONNECTED)
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
      
      if( testConnection() == 0 ) 
      {
         state(stateCodes::CONNECTED);
         log<text_log>(std::string("Connected to filter wheel on ") + m_deviceName);
      }
      else
      {
         return 0;
      }
   }
   
   if(state() == stateCodes::CONNECTED)
   {
      state(stateCodes::NOTHOMED);
   }
   
   if(state() == stateCodes::HOMING)
   {
      //std::cerr << "Homing state: " << m_homingState << " " << mx::get_curr_time() - m_homingStart << "\n";
      if(mx::get_curr_time() - m_homingStart > 20)
      {
         ++m_homingState;
         
         if(m_homingState == 1) //x complete
         {
            home_2();
         }
         else if(m_homingState == 2) //y complete
         {
            finishInit();            
         }
      }
   }
   
   if(state() == stateCodes::READY)
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;

      float pos1;
      if(getPos(pos1, 1) < 0)
      {
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      m_pos1 = pos1;
      
      //std::cerr << "m_pos1: " << m_pos1 << "\n";
      float mov1;
      if(getMov(mov1, 1) < 0)
      {
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }   
      
      //std::cerr << "mov1: " << mov1 << "\n";
      
      if(mov1 != m_pos1)
      {
         updateIfChanged(m_indiP_pos1, "current", m_pos1, INDI_BUSY);
      }
      else
      {
         updateIfChanged(m_indiP_pos1, "current", m_pos1, INDI_IDLE);
      }
      
      float pos2;
      if(getPos(pos2, 2) < 0)
      {
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      m_pos2 = pos2;
      //std::cerr << "m_pos2: " << m_pos2 << "\n";
      
      float mov2;
      if(getMov(mov2, 2) < 0)
      {
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }   
      //std::cerr << "mov2: " << mov2 << "\n";
      
      if(mov2 != m_pos2)
      {
         updateIfChanged(m_indiP_pos2, "current", m_pos2, INDI_BUSY);
      }
      else
      {
         updateIfChanged(m_indiP_pos2, "current", m_pos2, INDI_IDLE);
      }
      
   }
   return 0;
}

int pi335Ctrl::appShutdown()
{
   dev::dm<pi335Ctrl,float>::appShutdown();
   shmimMonitor<pi335Ctrl>::appShutdown();
   
   return 0;
}

int pi335Ctrl::testConnection()
{
   int rv;
   std::string resp;
   
   rv = tty::ttyWriteRead( resp, "*IDN?\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   //std::cerr << "idn response: " << resp << "\n";
   
   if(resp.find("E-727.3SDA") != std::string::npos) return 0;
   
   return -1;
   
}

int pi335Ctrl::initDM()
{
   int rv;
   std::string resp;
   
   
   //get open-loop position of axis 1 (should be zero)
   //std::cerr << "Sending: SVA? 1\n";
   rv = tty::ttyWriteRead( resp, "SVA? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   //std::cerr << "response: " << resp << "\n";
   
   //get open-loop position of axis 2 (should be zero)
  // std::cerr << "Sending: SVA? 2\n";
   rv = tty::ttyWriteRead( resp, "SVA? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   //std::cerr << "response: " << resp << "\n";
   
   
   //make sure axis 1 has servo off
   //std::cerr << "Sending: SVO 1 0\n";
   rv = tty::ttyWrite("SVO 1 0\n",  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   //make sure axis 2 has servo off
   //std::cerr << "Sending: SVO 2 0\n";
   rv = tty::ttyWrite("SVA 2 0\n",  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   m_servoState = 0;
   
   log<text_log>("servos off", logPrio::LOG_NOTICE);
   
   return home();
}
   

int pi335Ctrl::home()
{
   if(m_servoState != 0)
   {
      log<text_log>("home requested but servos are not off", logPrio::LOG_ERROR);
      return -1;
   }
   
   m_homingStart = 0;
   m_homingState = 0;

   state(stateCodes::HOMING);   
   
   
   return home_1();
   
}
   
int pi335Ctrl::home_1()
{
   int rv;
   
   if(m_servoState != 0)
   {
      log<text_log>("home_1 requested but servos are not off", logPrio::LOG_ERROR);
      return -1;std::string com = "MOV 1 " + std::to_string(m_homePos1) + "\n";
      //std::cerr << "Sending: " << com;
   }
   
   if(m_homingState != 0)
   {
      log<text_log>("home_1 requested but not in correct homing state", logPrio::LOG_ERROR);
      return -1;
   }
   
   //zero range found in axis 1 (NOTE this moves mirror full range) TAKES 1min 
   //std::cerr << "Sending: ATZ 1 NaN\n";
   rv = tty::ttyWrite("ATZ 1 NaN\n",  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }

   m_homingStart = mx::get_curr_time();
   m_homingState = 0;
   log<text_log>("commenced homing x");
   
   return 0;
}

int pi335Ctrl::home_2()
{
   int rv;
   
   if(m_servoState != 0)
   {
      log<text_log>("home_2 requested but servos are not off", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(m_homingState != 1)
   {
      log<text_log>("home_2 requested but not in correct homing state", logPrio::LOG_ERROR);
      return -1;
   }
   
   //zero range found in axis 2 (NOTE this moves mirror full range) TAKES 1min 
   //std::cerr << "Sending: ATZ 2 NaN\n";
   rv = tty::ttyWrite("ATZ 2 NaN\n", m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }

   m_homingStart = mx::get_curr_time();
   log<text_log>("commenced homing y");
   
   return 0;
}
 
int pi335Ctrl::finishInit()
{
   int rv;
   std::string resp;
   
   if(m_servoState != 0)
   {
      log<text_log>("finishInit requested but servos are not off", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(m_homingState != 2)
   {
      log<text_log>("finishInit requested but not in correct homing state", logPrio::LOG_ERROR);
      return -1;
   }
   
   
   //goto openloop pos zero (0 V) axis 1
   //std::cerr << "Sending: SVA 1 0.0\n";
   rv = tty::ttyWrite("SVA 1 0.0\n", m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   mx::milliSleep(2000);
   
   //goto openloop pos zero (0 V) axis 2
   //std::cerr << "Sending: SVA 2 0.0\n";
   rv = tty::ttyWrite("SVA 2 0.0\n", m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   mx::milliSleep(2000);

    //Get the real position of axis 1 (should be 0mrad st start) 
   //std::cerr << "Sending: SVA? 1\n";
   rv = tty::ttyWriteRead( resp, "SVA? 1\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   //std::cerr << "response: " << resp << "\n";
   
   //Get the real position of axis 2 (should be 0mrad st start) 
   //std::cerr << "Sending: SVA? 2\n";
   rv = tty::ttyWriteRead( resp, "SVA? 2\n", "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   //std::cerr << "response: " << resp << "\n";

   //now safe to engage servos    
   //(IMPORTANT:    NEVER EVER enable servos on axis 3 -- will damage S-335) 
   
   
   
   //turn on servo to axis 1 (green servo LED goes on 727) 
   //std::cerr << "Sending: SVO 1 1\n";
   rv = tty::ttyWrite("SVO 1 1\n",  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }

   mx::milliSleep(250);
   
   //turn on servo to axis 1 (green servo LED goes on 727) 
   //std::cerr << "Sending: SVO 2 1\n";
   rv = tty::ttyWrite("SVO 2 1\n", m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }

   m_servoState = 1;
   log<text_log>("servos engaged", logPrio::LOG_NOTICE);

   mx::milliSleep(1000);
   
   //now safe for closed loop moves 
   //center axis 1 (to 17.5 mrad)
   
   std::string com = "MOV 1 " + std::to_string(m_homePos1) + "\n";
   //std::cerr << "Sending: " << com;
   rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   //center axis 1 (to 17.5 mrad)
   com = "MOV 2 " + std::to_string(m_homePos2) + "\n";
   //std::cerr << "Sending: " << com;
   rv = tty::ttyWrite(com,  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv) } );
   }
   
   state(stateCodes::READY);
   
   return 0;
}

int pi335Ctrl::releaseDM()
{
   int rv;
   
   //std::cerr << "Sending: MOV 1 0\n";
   rv = tty::ttyWrite("MOV 1 0\n", m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   //std::cerr << "Sending: MOV 2 0\n";
   rv = tty::ttyWrite("MOV 2 0\n",  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv) } );
   }
   
   //std::cerr << "Sending: SVO 1 0\n";
   rv = tty::ttyWrite("SVO 1 0\n", m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   //std::cerr << "Sending: SVO 2 0\n";
   rv = tty::ttyWrite("SVO 2 0\n",  m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv) } );
   }

   m_servoState = 0;

   log<text_log>("servos off", logPrio::LOG_NOTICE);

   state(stateCodes::NOTHOMED);
   
   return 0;
}

int pi335Ctrl::getCom( std::string & resp,
                       const std::string & com,
                       int axis
                     )
{
   std::string sendcom = com;
   if(axis == 1 || axis == 2)
   {
      sendcom += " ";
      sendcom += std::to_string(axis);
   }
   
   sendcom += "\n";

   //std::cerr << "sending: " << sendcom;
   int rv = tty::ttyWriteRead( resp, sendcom, "\n", false, m_fileDescrip, m_writeTimeout, m_readTimeout);
   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
      return -1;
   }

   //std::cerr << "response: " << resp << "\n";
   
   return 0;
}

int pi335Ctrl::getPos( float & pos,
                       int n 
                     )
{
   std::string resp;   
   if(getCom(resp, "POS?", n)  < 0)
   {
      log<software_error>( {__FILE__, __LINE__});
   }


   ///\todo this should be a separate unit-tested parser
   size_t st = resp.find('=');
   if(st == std::string::npos || st > resp.size()-2)
   {
      log<software_error>( {__FILE__, __LINE__, "error parsing response"});
      return -1;
   }
   st += 1;
   pos = mx::ioutils::convertFromString<double>(resp.substr(st));
   
   return 0;
}

int pi335Ctrl::getMov( float & mov,
                       int n 
                     )
{
   std::string resp;   
   if(getCom(resp, "MOV?", n)  < 0)
   {
      log<software_error>( {__FILE__, __LINE__});
   }
   
   ///\todo this should be a separate unit-tested parser
   size_t st = resp.find('=');
   if(st == std::string::npos || st > resp.size()-2)
   {
      log<software_error>( {__FILE__, __LINE__, "error parsing response"});
      return -1;
   }
   st += 1;
   mov = mx::ioutils::convertFromString<double>(resp.substr(st));
   
   return 0;
}

int pi335Ctrl::move_1( float absPos )
{
   int rv;
   
   if(absPos < 0 || absPos > 35)
   {
      log<text_log>("request move on azis 1 out of range", logPrio::LOG_ERROR);
      return -1;
   }
   
   std::string com = "MOV 1 " + std::to_string(absPos) + "\n";
   
   //std::cerr << "Sending: " << com;
   rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   return 0;
}
   
int pi335Ctrl::move_2( float absPos )
{
   int rv;
   
   if(absPos < 0 || absPos > 35)
   {
      log<text_log>("request move on azis 2 out of range", logPrio::LOG_ERROR);
      return -1;
   }
   
   std::string com = "MOV 2 " + std::to_string(absPos) + "\n";
   
   //std::cerr << "Sending: " << com;
   rv = tty::ttyWrite(com, m_fileDescrip, m_writeTimeout);

   if(rv < 0)
   {
      log<software_error>( {__FILE__, __LINE__, rv, tty::ttyErrorString(rv)});
   }
   
   return 0;
}
   
INDI_NEWCALLBACK_DEFN(pi335Ctrl, m_indiP_pos1)(const pcf::IndiProperty &ipRecv)
{
   //if(MagAOXAppT::m_powerState == 0) return 0;
   
   if (ipRecv.getName() == m_indiP_pos1.getName())
   {
      float current = -1, target = -1;

      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<float>();
      }

      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<float>();
      }
      
      if(target == -1) target = current;
      
      if(target == -1) return 0;
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_pos1, "target", target);
      
      return move_1(target);
      
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(pi335Ctrl, m_indiP_pos2)(const pcf::IndiProperty &ipRecv)
{
   //if(MagAOXAppT::m_powerState == 0) return 0;
   
   if (ipRecv.getName() == m_indiP_pos2.getName())
   {
      float current = -1, target = -1;

      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<float>();
      }

      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<float>();
      }
      
      if(target == -1) target = current;
      
      if(target == -1) return 0;
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_pos2, "target", target);
      
      return move_2(target);
      
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //pi335Ctrl_hpp
