/** \file flipperCtrl.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup flipperCtrl_files
  */

#ifndef flipperCtrl_hpp
#define flipperCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup flipperCtrl
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup flipperCtrl_files
  * \ingroup flipperCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup flipperCtrl
  */
class flipperCtrl : public MagAOXApp<true>, public tty::usbDevice, public dev::ioDevice
{

   //Give the test harness access.
   friend class flipperCtrl_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}




public:
   /// Default c'tor.
   flipperCtrl();

   /// D'tor, declared and defined for noexcept.
   ~flipperCtrl() noexcept
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

   /// Implementation of the FSM for flipperCtrl.
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

   
   int getPos();

};

flipperCtrl::flipperCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void flipperCtrl::setupConfig()
{
   tty::usbDevice::setupConfig(config);
   dev::ioDevice::setupConfig(config);
}

int flipperCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   this->m_baudRate = B115200; //default for MCBL controller.  Will be overridden by any config setting.

   int rv = tty::usbDevice::loadConfig(_config);
   
   if(rv < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   dev::ioDevice::loadConfig(_config);
   
   return 0;
}

void flipperCtrl::loadConfig()
{
   if(loadConfigImpl(config)<0)
   {
      log<software_critical>({__FILE__, __LINE__});
      m_shutdown = 1;
      return;
   }
}

int flipperCtrl::appStartup()
{

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

int flipperCtrl::appLogic()
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
      elevatedPrivileges ep(this);
      int rv = connect();
      ep.restore();
      
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
      }
      
      state(stateCodes::CONNECTED);
   }
        
   if( state() == stateCodes::CONNECTED )
   {
      /*std::string header(6,'\0');
      header[0] = 0x23;
      header[1] = 0x02;
      header[2] = 0x00;
      header[3] = 0x00;
      header[4] = 0x50;
      header[5] = 0x01;
      
      tty::ttyWrite( header, m_fileDescrip, m_writeTimeout);*/
      
      getPos();
      
      std::string header(6,'\0');
      
      static int move = 0;
      
      header[0] = 0x6A;
      header[1] = 0x04;
      header[2] = 0x00;
      if(move == 0)
      {
         header[3] = 0x01;
         move = 1;
      }
      else if(move == 5)
      {
         header[3] = 0x02;
         move = 6;
      }
      else
      {
         ++move;
         
         if(move > 10) move = 0;
         return 0;
      }
      
      header[4] = 0x50;
      header[5] = 0x01;
      
      tty::ttyWrite( header, m_fileDescrip, m_writeTimeout);
      
      //sleep(2);
   }
   
   return 0;
}

int flipperCtrl::appShutdown()
{
   return 0;
}


int flipperCtrl::getPos()
{
   std::string header(6,'\0');
      
   header[0] = 0x80;
   header[1] = 0x04;
   header[2] = 0x00;
   header[3] = 0x00;
   header[4] = 0x50;
   header[5] = 0x01;
      
   tty::ttyWrite( header, m_fileDescrip, m_writeTimeout);
      
   std::string response;
   int rv = tty::ttyRead(response, 20, m_fileDescrip, m_readTimeout);
      
   std::cout << "Read: " << response.size() << " bytes\n";
      
//    if(rv > 0)
//    {
//       for(int i=0; i < rv; ++i)
//       {
//          std::cerr << i << " " << (int) response[i] << "\n";
//       }
//    }
//    else
//    {
//       return 0;
//    }
   
   
   if(response[16] == 1)
   {
      std::cerr << "position 1\n";
   }
   else
   {
      std::cerr << "position 2\n";
   }
}

} //namespace app
} //namespace MagAOX

#endif //flipperCtrl_hpp
