/** \file acesxeCtrl.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup acesxeCtrl_files
  */

#ifndef acesxeCtrl_hpp
#define acesxeCtrl_hpp

extern "C"
{
   #include "ArcusPerformaxDriver.h"
}

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup acesxeCtrl
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup acesxeCtrl_files
  * \ingroup acesxeCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup acesxeCtrl
  */
class acesxeCtrl : public MagAOXApp<true>
{

   //Give the test harness access.
   friend class acesxeCtrl_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}


   AR_HANDLE m_handle {nullptr}; //usb handle


   float m_windSpeed {10};
   bool m_forward {true};

   int m_lspd {150}; //This sets a lower limit of 0.9 m/s.
   int m_hspd {300};

public:
   /// Default c'tor.
   acesxeCtrl();

   /// D'tor, declared and defined for noexcept.
   ~acesxeCtrl() noexcept
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

   /// Implementation of the FSM for acesxeCtrl.
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


   int sendRecv( std::string & sout,
                 const std::string & com
               );

   
   float hspd();

   int hspd(int hspd);

   float windSpeed();

   int windSpeed(float ws);


   int start();

   int stop();

   int abort();

   pcf::IndiProperty m_indiP_windspeed;

   pcf::IndiProperty m_indiP_start;
   pcf::IndiProperty m_indiP_stop;
   pcf::IndiProperty m_indiP_abort;

   INDI_NEWCALLBACK_DECL(acesxeCtrl, m_indiP_windspeed);
   INDI_NEWCALLBACK_DECL(acesxeCtrl, m_indiP_start);
   INDI_NEWCALLBACK_DECL(acesxeCtrl, m_indiP_stop);
   INDI_NEWCALLBACK_DECL(acesxeCtrl, m_indiP_abort);

};

acesxeCtrl::acesxeCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void acesxeCtrl::setupConfig()
{
}

int acesxeCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{

   static_cast<void>(_config);

   return 0;
}

void acesxeCtrl::loadConfig()
{
   loadConfigImpl(config);
}

int acesxeCtrl::appStartup()
{
   
   //These can be made config parametrs
   if(!fnPerformaxComSetTimeouts(5000,5000))
	{
      log<software_error>({__FILE__, __LINE__, "error setting timeouts"});
		return -1;
	}

   createStandardIndiNumber<float>( m_indiP_windspeed, "windspeed", -60, 60, 0.0, "%f");
   registerIndiPropertyNew( m_indiP_windspeed, INDI_NEWCALLBACK(m_indiP_windspeed)) ;
   m_indiP_windspeed["target"].set<float>(0);
   m_indiP_windspeed["current"].set<float>(0);

   createStandardIndiRequestSw( m_indiP_start, "start", "Start", "Turb Sim Controls");
   registerIndiPropertyNew( m_indiP_start, INDI_NEWCALLBACK(m_indiP_start) );

   createStandardIndiRequestSw( m_indiP_stop, "stop", "Stop", "Turb Sim Controls");
   registerIndiPropertyNew( m_indiP_stop, INDI_NEWCALLBACK(m_indiP_stop) );

   createStandardIndiRequestSw( m_indiP_abort, "abort", "Abort", "Turb Sim Controls");
   registerIndiPropertyNew( m_indiP_abort, INDI_NEWCALLBACK(m_indiP_abort) );

   m_powerState = 1; 
   m_powerTargetState = 1;


   state(stateCodes::NODEVICE);
   return 0;
}

int acesxeCtrl::appLogic()
{
   if(state() == stateCodes::POWERON)
   {
      state(stateCodes::NODEVICE);
   }

   if(state() == stateCodes::NODEVICE)
   {
      AR_DWORD num;

      std::lock_guard<std::mutex> guard(m_indiMutex);

      if(!fnPerformaxComGetNumDevices(&num))
	   {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
		   log<software_error>({__FILE__, __LINE__, "error in fnPerformaxComGetNumDevices"});
		   return 0;
	   }

	   if(num<1)
	   {
         if(!stateLogged())
         {
            log<text_log>("ACE-SXE not found");
         }
		   return 0;
	   }

      if(num>1)
	   {
         log<text_log>("Too many ACE-SXEs found.  I can't handle this.", logPrio::LOG_CRITICAL);
		   return -1;
	   }

      char 	lpDeviceString[PERFORMAX_MAX_DEVICE_STRLEN];

      elevatedPrivileges elPriv(this);

      if( !fnPerformaxComGetProductString(0, lpDeviceString, PERFORMAX_RETURN_SERIAL_NUMBER) )
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__, __LINE__, "error acquiring product serial number"});
		   return 0;
      }
      lpDeviceString[sizeof(lpDeviceString) - 1] = '\0'; //don't trust too much

		std::string serial = lpDeviceString;

      if( !fnPerformaxComGetProductString(0, lpDeviceString, PERFORMAX_RETURN_DESCRIPTION) )
   	{
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
		   log<software_error>({__FILE__, __LINE__, "errovr acquiring product description"});
		   return 0;
	   }
      lpDeviceString[sizeof(lpDeviceString) - 1] = '\0'; //don't trust too much
 
      std::string descrip = lpDeviceString;

      log<text_log>("found ACE-SXE: " + descrip + " " + serial);

      state(stateCodes::NOTCONNECTED);
   }

   if(state() == stateCodes::NOTCONNECTED)
   {
      //setup the connection
	
      std::lock_guard<std::mutex> guard(m_indiMutex);

      if(m_handle)
      {
         if(!fnPerformaxComClose(m_handle))
	      {
            if(m_powerState != 1 || m_powerTargetState != 1) return 0;
		      log<software_error>({__FILE__, __LINE__, "error closing existing handle"});
	      }
         m_handle = nullptr;
      }


      {
      elevatedPrivileges elPriv(this);

	   if(!fnPerformaxComOpen(0,&m_handle))
	   {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;  
         log<software_error>({__FILE__, __LINE__, "error closing existing handle"});
         state(stateCodes::ERROR);
		   return 0;
	   }
      }

      if(!fnPerformaxComFlush(m_handle))
	   {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;  
         log<software_error>({__FILE__, __LINE__, "error flushing"});
         state(stateCodes::ERROR);
		   return 0;
	   }

      char		out[64];
	   char		in[64];

      strncpy(out, "ID", sizeof(out)); 
	   if( !fnPerformaxComSendRecv(m_handle, out, sizeof(out), sizeof(in), in) )
	   {
		   if(m_powerState != 1 || m_powerTargetState != 1) return 0;  
         log<software_error>({__FILE__, __LINE__, "error getting ID"});
         state(stateCodes::ERROR);
		   return 0;   
   	}
      in[sizeof(in)-1] = '\0';
      std::string id = in;

	   strncpy(out, "DN", sizeof(out)); //read current
	   if(!fnPerformaxComSendRecv(m_handle, out, sizeof(out), sizeof(in), in))
	   {
		   if(m_powerState != 1 || m_powerTargetState != 1) return 0;  
         log<software_error>({__FILE__, __LINE__, "error getting DN"});
         state(stateCodes::ERROR);
		   return 0;   
   	}

      std::string dn = in;

      state(stateCodes::CONNECTED);
      log<text_log>("connected to ACE-SXE " + id + " " + dn);
      
	}
	

   if(state() == stateCodes::CONNECTED)
   {
      std::string resp;

      std::lock_guard<std::mutex> guard(m_indiMutex);

      //Check the parameters
      if(sendRecv(resp, "EDIO") != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      log<text_log>( "EDIO=" + resp);

      if(sendRecv(resp, "POL") != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      log<text_log>( "POL=" + resp);

      if(sendRecv(resp, "ACC") != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      log<text_log>( "ACC=" + resp);      

      if(sendRecv(resp, "LSPD=" + std::to_string(m_lspd)) != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      if(resp != "OK")
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__, __LINE__, "not OK from LSPD="});
         state(stateCodes::ERROR);
         return 0;
      }
      log<text_log>("set LSPD=" + std::to_string(m_lspd) );

      if( windSpeed(m_windSpeed) != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }


      if(sendRecv(resp, "MST") != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      int mst = stoi(resp);

      if(mst & 1 || mst & 2 || mst & 4) state(stateCodes::OPERATING);
      else state(stateCodes::READY);

   }

   if(state() == stateCodes::ERROR)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;  

      if(m_handle)
      {
         fnPerformaxComClose(m_handle);
	      m_handle = nullptr;
      }

      state(stateCodes::NODEVICE);

      return 0;

   }

   if(state() == stateCodes::READY || state() == stateCodes::OPERATING)
   {
      std::string smst;
      if(sendRecv(smst, "MST") != 0)
      {
         if(m_powerState != 1 || m_powerTargetState != 1) return 0;
         log<software_error>({__FILE__,__LINE__});
         state(stateCodes::ERROR);
         return 0;
      }
      
      int mst = stoi(smst);

      if(mst & 1 || mst & 2 || mst & 4) state(stateCodes::OPERATING);
      else state(stateCodes::READY);

      std::lock_guard<std::mutex> guard(m_indiMutex);

      updateIfChanged(m_indiP_windspeed, "current", windSpeed(), INDI_OK);
   }
   else
   {
      if(state() == stateCodes::POWEROFF || state() == stateCodes::POWERON) return 0;
      log<software_error>({__FILE__,__LINE__, "bad state"});
   }

   return 0;
}

int acesxeCtrl::appShutdown()
{
   return 0;
}

int acesxeCtrl::sendRecv( std::string & sout, const std::string & com)
{
   char		out[64];
	char		in[64];

   //Check the parameters
   strncpy(out, com.c_str(), sizeof(out)-1); 
	if( !fnPerformaxComSendRecv(m_handle, out, sizeof(out), sizeof(in), in) )
	{
	   if(m_powerState != 1 || m_powerTargetState != 1) return -1; //error, but don't log  
      log<software_error>({__FILE__, __LINE__, std::string("error getting ") + out});
      state(stateCodes::ERROR);
		return -1;   
   }
   in[sizeof(in)-1] = '\0'; //trust but verify
   sout = in;

   return 0;
}

float acesxeCtrl::hspd()
{
   return m_hspd;
}

int acesxeCtrl::hspd(int hspd)
{
   std::string resp;

   //Check the parameters
   if(sendRecv(resp, "HSPD="+std::to_string(hspd)) != 0)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__,__LINE__});
      state(stateCodes::ERROR);
      return -1;
   }
   if(resp != "OK")
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__, __LINE__, "not OK from HSPD="});
      state(stateCodes::ERROR);
      return -1;
   }

   if(sendRecv(resp, "HSPD") != 0)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__,__LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   m_hspd = std::stoi(resp);

   log<text_log>("set HSPD to " + std::to_string(m_hspd));

   return 0;

}

float acesxeCtrl::windSpeed()
{
   return (1. - 2.*(m_forward-1)) * m_hspd/9600.*60. * (12.7/12.0); //(12.7 m/s per 12 rpm)

}

int acesxeCtrl::windSpeed(float ws)
{
   if(ws < 0) m_forward = false;
   else m_forward = true;

   if(ws < -60) 
   {
      ws = -60;
      log<text_log>("wind speed limited to -60 m/s", logPrio::LOG_NOTICE);
   }
   if(ws > 60) 
   {
      ws = 60;
      log<text_log>("wind speed limited to 60 m/s", logPrio::LOG_NOTICE);
   }

   float ll = m_lspd/9600.*60. * (12.7/12.0); 
   if(ws <= 0 && ws > -ll)
   {
      log<text_log>("wind speed below minimum of 0.9 m/s", logPrio::LOG_NOTICE);
   }

   if(ws >= 0 && ws < ll)
   {
      log<text_log>("wind speed below minimum of 0.9 m/s", logPrio::LOG_NOTICE);
   }


   int h = (12.0/12.7) / 60. * 9600 * fabs(ws);

   if(hspd(h) != 0)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return -1;
      log<software_error>({__FILE__,__LINE__});
      state(stateCodes::ERROR);
      return -1;
   }

   log<text_log>("set wind speed to " + std::to_string(windSpeed()));

   updateIfChanged(m_indiP_windspeed, "target", windSpeed(), INDI_OK);
   updateIfChanged(m_indiP_windspeed, "current", windSpeed(), INDI_OK);

   return 0;

}

int acesxeCtrl::start()
{
   std::string resp;

   std::string com = "J";
   if(m_forward) com += "+";
   else com += "-";

   if(sendRecv(resp, com) != 0)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__,__LINE__});
      state(stateCodes::ERROR);
      return -1;
   }
   if(resp != "OK")
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__, __LINE__, "not OK from " + com});
      state(stateCodes::ERROR);
      return -1;
   }

   log<text_log>("started spinning turbulence simulator", logPrio::LOG_NOTICE);
   return 0;
}

int acesxeCtrl::stop()
{
   std::string resp;

   //Check the parameters
   if(sendRecv(resp, "STOP") != 0)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__,__LINE__});
      state(stateCodes::ERROR);
      return -1;
   }
   if(resp != "OK")
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__, __LINE__, "not OK from STOP"});
      state(stateCodes::ERROR);
      return -1;
   }

   log<text_log>("stopped spinning turbulence simulator", logPrio::LOG_NOTICE);
   return 0;
}

int acesxeCtrl::abort()
{
   std::string resp;

   //Check the parameters
   if(sendRecv(resp, "ABORT") != 0)
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__,__LINE__});
      state(stateCodes::ERROR);
      return -1;
   }
   if(resp != "OK")
   {
      if(m_powerState != 1 || m_powerTargetState != 1) return 0;
      log<software_error>({__FILE__, __LINE__, "not OK from ABORT"});
      state(stateCodes::ERROR);
      return -1;
   }

   log<text_log>("aborted spinning turbulence simulator", logPrio::LOG_NOTICE);
   return 0;
}

INDI_NEWCALLBACK_DEFN(acesxeCtrl, m_indiP_windspeed)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_windspeed.getName())
   {
      float ws = 0;

      if(ipRecv.find("current"))
      {
         ws = ipRecv["current"].get<double>();
      }

      if(ipRecv.find("target"))
      {
         ws = ipRecv["target"].get<double>();
      }

      if(ws == 0) return 0;
      
      std::lock_guard<std::mutex> guard(m_indiMutex);

      updateIfChanged(m_indiP_windspeed, "target", ws, INDI_OK);

      return windSpeed(ws);
   }

   return -1;
}

INDI_NEWCALLBACK_DEFN(acesxeCtrl, m_indiP_start )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_start.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
      
   if(!ipRecv.find("request")) return 0;
           
   std::unique_lock<std::mutex> lock(m_indiMutex);
      
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      return start();
   }   
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(acesxeCtrl, m_indiP_stop )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_stop.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
      
   if(!ipRecv.find("request")) return 0;
           
   std::unique_lock<std::mutex> lock(m_indiMutex);
      
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      return stop();
   }   
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(acesxeCtrl, m_indiP_abort )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_abort.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
      
   if(!ipRecv.find("request")) return 0;
           
   std::unique_lock<std::mutex> lock(m_indiMutex);
      
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      return abort();
   }   
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //acesxeCtrl_hpp
