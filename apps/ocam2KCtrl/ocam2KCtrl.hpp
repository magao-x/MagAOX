/** \file ocam2KCtrl.hpp
  * \brief The MagAO-X OCAM2K EMCCD camera controller.
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup ocam2KCtrl_files
  */

#ifndef ocam2KCtrl_hpp
#define ocam2KCtrl_hpp


#include <edtinc.h>
#include <ImageStruct.h>
#include <ImageStreamIO.h>



#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT; //This needs to be before pdvUtils.hpp for logging to work.

#include "fli/ocam2_sdk.h"
#include "ocamUtils.hpp"

namespace MagAOX
{
namespace app
{

#define CAMCTRL_E_NOCONFIGS (-10)
   
struct cameraConfig 
{
   std::string configFile;
   std::string serialCommand;
   unsigned binning {0};
   unsigned sizeX {0};
   unsigned sizeY {0};
   float maxFPS {0};
};

typedef std::unordered_map<std::string, cameraConfig> cameraConfigMap;

inline
int loadCameraConfig( cameraConfigMap & ccmap,
                      mx::app::appConfigurator & config 
                    )
{
   std::vector<std::string> sections;

   config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      return CAMCTRL_E_NOCONFIGS;
   }
   
   for(size_t i=0; i< sections.size(); ++i)
   {
      bool fileset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "configFile" ));
      /*bool binset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "binning" ));
      bool sizeXset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      bool sizeYset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      bool maxfpsset = config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      */
      
      //The configuration file tells us most things for EDT, so it's our current requirement. 
      if( !fileset ) continue;
      
      std::string configFile;
      config.configUnused(configFile, mx::app::iniFile::makeKey(sections[i], "configFile" ));
      
      std::string serialCommand;
      config.configUnused(serialCommand, mx::app::iniFile::makeKey(sections[i], "serialCommand" ));
      
      unsigned binning = 0;
      config.configUnused(binning, mx::app::iniFile::makeKey(sections[i], "binning" ));
      
      unsigned sizeX = 0;
      config.configUnused(sizeX, mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      
      unsigned sizeY = 0;
      config.configUnused(sizeY, mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      
      float maxFPS = 0;
      config.configUnused(maxFPS, mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      
      ccmap[sections[i]] = cameraConfig({configFile, serialCommand, binning, sizeX, sizeY, maxFPS});
   }
   
   return 0;
}

/** \defgroup ocam2KCtrl OCAM2K EMCCD Camera
  * \brief Control of the OCAM2K EMCCD Camera.
  *
  * \link page_module_ocam2KCtrl Application Documentation
  *
  * \ingroup apps
  *
  */

/** \defgroup ocam2KCtrl_files OCAM2K EMCCD Camera Files
  * \ingroup ocam2KCtrl
  */

/** MagAO-X application to control the OCAM 2K EMCCD
  *
  * \ingroup ocam2KCtrl
  */
class ocam2KCtrl : public MagAOXApp<>, public dev::ioDevice
{

protected:

   

   /** \name configurable parameters 
     *@{
     */ 
   int m_unit {0};
   int m_channel {0};

   unsigned long m_powerOnWait {10000000000}; ///< Time in nsec to wait for camera boot after power on.

   
   cameraConfigMap m_cameraConfigs; ///< Map holding the possible camera mode configurations
   
   float m_startupTemp {20.0}; ///< The temperature to set after a power-on.
   
   std::string m_startupMode; ///< The camera mode to load during first init after a power-on.
   
   std::string m_ocamDescrambleFile; ///< Path the OCAM 2K pixel descrambling file, relative to MagAO-X config directory.

   ///@}
   
   PdvDev * m_pdv {nullptr}; ///< The EDT PDV device handle

   float m_fpsSet {0}; ///< The commanded fps, as returned by the camera

   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.
   
   int m_width {0}; ///< The width of the image according to the framegrabber
   int m_height {0}; ///< The height of the image frame according the framegrabber
   int m_depth {0}; ///< The pixel bit depth according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
    
      
public:

   ///Default c'tor
   ocam2KCtrl();

   ///Destructor
   ~ocam2KCtrl() noexcept;

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


   int pdvConfig(std::string & cfgname);
   
   
   int getTemps();
   int setTemp(float temp);
   
   int getFPS();
   int setFPS(float fps);
   
protected:
   
   int m_fgThreadPrio {1}; ///< Priority of the framegrabber thread, should normally be > 00.

   std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

   ///Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
   static void _fgThreadStart( ocam2KCtrl * o /**< [in] a pointer to an ocam2KCtrl instance (normally this) */);

   /// Start the log capture.
   int fgThreadStart();

   /// Execute the log capture.
   void fgThreadExec();

   
   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_ccdtemp;
   pcf::IndiProperty m_indiP_temps;
   pcf::IndiProperty m_indiP_mode;
   pcf::IndiProperty m_indiP_fps;

public:
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_ccdtemp);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_mode);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_fps);

};

inline
ocam2KCtrl::ocam2KCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   ///\todo if power management is not fully corectly specified (e.g. outlet instead of channel is used as keyword), things just silently hang. Fix implemented, test needed.
   m_powerMgtEnabled = true;
   
   return;
}

inline
ocam2KCtrl::~ocam2KCtrl() noexcept
{
   if(m_pdv) pdv_close(m_pdv);

   return;
}

inline
void ocam2KCtrl::setupConfig()
{
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the fraemgrabber thread.");
   
   
   config.add("camera.startupTemp", "", "camera.startupTemp", argType::Required, "camera", "startupTemp", false, "float", "The temperature setpoint to set after a power-on [C].  Default is 20 C.");
   
   config.add("camera.startupMode", "", "camera.startupMode", argType::Required, "camera", "startupMode", false, "string", "The name of the configuration to set at startup.");
   
   config.add("camera.startupMode", "", "camera.startupMode", argType::Required, "camera", "startupMode", false, "string", "The name of the configuration to set at startup.");
   
   config.add("camera.ocamDescrambleFile", "", "camera.ocamDescrambleFile", argType::Required, "camera", "ocamDescrambleFile", false, "string", "The path of the OCAM descramble file, relative to MagAOX/config.");
   
   dev::ioDevice::setupConfig(config);
}



inline
void ocam2KCtrl::loadConfig()
{
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   
   config(m_startupMode, "camera.startupMode");
   config(m_ocamDescrambleFile, "camera.ocamDescrambleFile");
   
   int rv = loadCameraConfig( m_cameraConfigs, config);
   
   if(rv < 0)
   {
      if(rv == CAMCTRL_E_NOCONFIGS)
      {
         log<text_log>("No camera configurations found.", logPrio::LOG_CRITICAL);
      }
      
      m_shutdown = true;
   }
   
   m_readTimeout = 1000;
   m_writeTimeout = 1000;
   dev::ioDevice::loadConfig(config);
}

#define MAGAOX_PDV_SERBUFSIZE 512

int pdvSerialWriteRead( std::string & response,
                        PdvDev * pdv,
                        const std::string & command,
                        int timeout ///< [in] timeout in milliseconds
                      )
{
   char    buf[MAGAOX_PDV_SERBUFSIZE+1];

   // Flush the channel first.
   // This does not indicate errors, so no checks possible.
   pdv_serial_read(pdv, buf, MAGAOX_PDV_SERBUFSIZE);

   if( pdv_serial_command(pdv, command.c_str()) < 0)
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__, "PDV: error sending serial command"});
      return -1;
   }

   int ret;

   ret = pdv_serial_wait(pdv, timeout, 1);

   if(ret == 0)
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__, "PDV: timeout, no serial response"});
      return -1;
   }

   u_char  lastbyte, waitc;

   response.clear();

   do
   {
      ret = pdv_serial_read(pdv, buf, MAGAOX_PDV_SERBUFSIZE);

      if(ret > 0) response += buf;

      //Check for last char, wait for more otherwise.
      if (*buf) lastbyte = (u_char)buf[strlen(buf)-1];

      if (pdv_get_waitchar(pdv, &waitc) && (lastbyte == waitc))
          break;
      else ret = pdv_serial_wait(pdv, timeout/2, 1);
   }
   while(ret > 0);

   if(ret == 0 && pdv_get_waitchar(pdv, &waitc))
   {
      MagAOXAppT::log<software_error>({__FILE__, __LINE__, "PDV: timeout in serial response"});
      return -1;
   }

   return 0;
}

/* Todo:


-- Image loop:
  -- flags for stop, reconfig, shutdown 
  -- measure fps 
  --
  
-- Need to load pdv config on mode change.
   -- but from in imaging loop 
   
-- mutex m_pdv, along with serial communications, but not framegrabbing.

-- need non-mutex way to check for consistency in f.g.-ing, or a way to wait until that loop exits.

-- need way for f.g. loop to communicate errors.

-- INDI props for measured fps.
-- INDI prop for timestamp of last frame skip.  Maybe total frameskips?
-- Configs to add:
  -- pdv unit number
  -- startup temp command
  -- startup delay
  -- circ buff size for pdv 
  
-- add ImageStreamIO
  -- config: filename
  -- buffer length (ser. buffer size)
  */



inline
int ocam2KCtrl::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_ccdtemp, "ccdtemp", pcf::IndiProperty::Number);
   m_indiP_ccdtemp.add (pcf::IndiElement("current"));
   m_indiP_ccdtemp["current"].set(0);
   m_indiP_ccdtemp.add (pcf::IndiElement("target"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_temps, "temps", pcf::IndiProperty::Number);
   m_indiP_temps.add (pcf::IndiElement("cpu"));
   m_indiP_temps["cpu"].set(0);
   m_indiP_temps.add (pcf::IndiElement("power"));
   m_indiP_temps["power"].set(0);
   m_indiP_temps.add (pcf::IndiElement("bias"));
   m_indiP_temps["bias"].set(0);
   m_indiP_temps.add (pcf::IndiElement("water"));
   m_indiP_temps["water"].set(0);
   m_indiP_temps.add (pcf::IndiElement("left"));
   m_indiP_temps["left"].set(0);
   m_indiP_temps.add (pcf::IndiElement("right"));
   m_indiP_temps["right"].set(0);
   m_indiP_temps.add (pcf::IndiElement("cooling"));
   m_indiP_temps["cooling"].set(0);

   REG_INDI_NEWPROP(m_indiP_mode, "mode", pcf::IndiProperty::Text);
   m_indiP_mode.add (pcf::IndiElement("current"));
   m_indiP_mode.add (pcf::IndiElement("target"));

   REG_INDI_NEWPROP(m_indiP_fps, "fps", pcf::IndiProperty::Number);
   m_indiP_fps.add (pcf::IndiElement("current"));
   m_indiP_fps["current"].set(0);
   m_indiP_fps.add (pcf::IndiElement("target"));

   if(pdvConfig(m_startupMode) < 0) 
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   if(fgThreadStart() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}



inline
int ocam2KCtrl::appLogic()
{
   //first do a join check to see if other threads have exited.
   if(pthread_tryjoin_np(m_fgThread.native_handle(),0) == 0)
   {
      log<software_error>({__FILE__, __LINE__, "framegrabber thread has exited"});
      
      return -1;
   }
   
   //Handle the case where we enter this loop already powered on.
   static bool firstCall = 1;
   
   if( state() == stateCodes::POWERON )
   {
      if(m_powerOnCounter*m_loopPause > m_powerOnWait || firstCall)
      {
         state(stateCodes::NOTCONNECTED);
         m_powerOnCounter = 0;
      }
      else
      {
         ++m_powerOnCounter;
         return 0;
      }
   }

   firstCall = 0;
   
   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      std::string response;

      //Might have gotten here because of a power off.
      if(m_powerState == 0) return 0;
      
      int ret = pdvSerialWriteRead( response, m_pdv, "fps", m_readTimeout);
      if( ret == 0)
      {
         state(stateCodes::CONNECTED);
      }
      else
      {
         sleep(1);
         return 0;
      }
   }

   if( state() == stateCodes::CONNECTED )
   {
      //Get a lock
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      if( getFPS() == 0 )
      {
         if(m_fpsSet == 0) state(stateCodes::READY);
         else state(stateCodes::OPERATING);
         
         if(setTemp(m_startupTemp) < 0)
         {
            return log<software_error,0>({__FILE__,__LINE__});
         }
      }
      else
      {
         state(stateCodes::ERROR);
         return log<software_error,0>({__FILE__,__LINE__});
      }
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //Get a lock if we can
      std::unique_lock<std::mutex> lock(m_indiMutex, std::try_to_lock);

      //but don't wait for it, just go back around.
      if(!lock.owns_lock()) return 0;
      
      if(getTemps() < 0)
      {
         if(m_powerState == 0) return 0;
         
         state(stateCodes::ERROR);
         return 0;
      }

      if(getFPS() < 0)
      {
         if(m_powerState == 0) return 0;
         
         state(stateCodes::ERROR);
         return 0;
      }
   }

   //Fall through check?

   return 0;

}

inline
int ocam2KCtrl::onPowerOff()
{
   m_powerOnCounter = 0;
   
   std::lock_guard<std::mutex> lock(m_indiMutex);
   
   updateIfChanged(m_indiP_ccdtemp, "current", -999);
   updateIfChanged(m_indiP_ccdtemp, "target", -999);
   
   updateIfChanged(m_indiP_temps, "cpu", -999);
   updateIfChanged(m_indiP_temps, "power", -999);
   updateIfChanged(m_indiP_temps, "bias", -999);
   updateIfChanged(m_indiP_temps, "water", -999);
   updateIfChanged(m_indiP_temps, "left", -999);
   updateIfChanged(m_indiP_temps, "right", -999);
   updateIfChanged(m_indiP_temps, "cooling", -999);
   
   updateIfChanged(m_indiP_mode, "current", std::string(""));
   updateIfChanged(m_indiP_mode, "target", std::string(""));

   updateIfChanged(m_indiP_fps, "current", 0);
   updateIfChanged(m_indiP_fps, "target", 0);
      
   return 0;
}

inline
int ocam2KCtrl::whilePowerOff()
{
   return 0;
}

inline
int ocam2KCtrl::appShutdown()
{
   //don't bother
   return 0;
}

inline
int ocam2KCtrl::pdvConfig(std::string & modeName)
{
   Dependent *dd_p;
   EdtDev *edt_p = NULL;
   Edtinfo edtinfo;
   
   //Preliminaries
   if(m_pdv)
   {
      pdv_close(m_pdv);
      m_pdv = nullptr;
   }

      
   //----- NEED TO resolve to absolute cfgname path.
   if(m_cameraConfigs.count(modeName) != 1)
   {
      return log<text_log, -1>("No mode named " + modeName + " found.", logPrio::LOG_ERROR);
   }
   
   std::string configFile = m_configDir + "/" + m_cameraConfigs[modeName].configFile;
   
   log<text_log>("Loading EDT PDV config file: " + configFile);
      
   if ((dd_p = pdv_alloc_dependent()) == NULL)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "EDT PDV alloc_dependent FAILED"});      
   }
   
   if (pdv_readcfg(configFile.c_str(), dd_p, &edtinfo) != 0)
   {
      free(dd_p);
      return log<software_error, -1>({__FILE__, __LINE__, "EDT PDV readcfg FAILED"});
      
   }
   
   char edt_devname[128];
   strncpy(edt_devname, EDT_INTERFACE, sizeof(edt_devname));
   
   if ((edt_p = edt_open_channel(edt_devname, m_unit, m_channel)) == NULL)
   {
      char errstr[256];
      edt_perror(errstr);
      free(dd_p);
      return log<software_error, -1>({__FILE__, __LINE__, std::string("EDT PDV edt_open_channel FAILED: ") + errstr});
   }
   
   char bitdir[1];
   bitdir[0] = '\0';
    
   int pdv_debug = 0;
   if(m_log.logLevel() > logPrio::LOG_INFO) pdv_debug = 2;
   
   if (pdv_initcam(edt_p, dd_p, m_unit, &edtinfo, configFile.c_str(), bitdir, pdv_debug) != 0)
   {
      edt_close(edt_p);
      free(dd_p);
      return log<software_error, -1>({__FILE__, __LINE__, "initcam failed. Run with '--logLevel=DBG' to see complete debugging output."});
   }

   edt_close(edt_p);
   free(dd_p);
   
   
   //Now open the PDV device handle for talking to the camera via the EDT board.
   if ((m_pdv = pdv_open_channel(edt_devname, m_unit, m_channel)) == NULL)
   {
      std::string errstr = std::string("pdv_open_channel(") + edt_devname + std::to_string(m_unit) + "_" + std::to_string(m_channel) + ")";

      log<software_error>({__FILE__, __LINE__, errstr});
      log<software_error>({__FILE__, __LINE__, errno});

      return -1;
   }

   pdv_flush_fifo(m_pdv);

   pdv_serial_read_enable(m_pdv); //This is undocumented, don't know if it's really needed.
   
   m_width = pdv_get_width(m_pdv);
   m_height = pdv_get_height(m_pdv);
   m_depth = pdv_get_depth(m_pdv);
   m_cameraType = pdv_get_cameratype(m_pdv);

   log<text_log>("Initialized camera: " + m_cameraType);
   log<text_log>("WxHxD: " + std::to_string(m_width) + " X " + std::to_string(m_height) + " X " + std::to_string(m_depth));

   return 0;

}

inline
int ocam2KCtrl::getTemps()
{
   std::string response;

   if( pdvSerialWriteRead( response, m_pdv, "temp", m_readTimeout) == 0)
   {
      ocamTemps temps;

      if(parseTemps( temps, response ) < 0) 
      {
         if(m_powerState == 0) return -1;
         return log<software_error, -1>({__FILE__, __LINE__, "Temp. parse error"});
      }
      
      log<ocam_temps>({temps.CCD, temps.CPU, temps.POWER, temps.BIAS, temps.WATER, temps.LEFT, temps.RIGHT, temps.COOLING_POWER});

      updateIfChanged(m_indiP_ccdtemp, "current", temps.CCD);
      updateIfChanged(m_indiP_ccdtemp, "target", temps.SET);
      
      updateIfChanged(m_indiP_temps, "cpu", temps.CPU);
      updateIfChanged(m_indiP_temps, "power", temps.POWER);
      updateIfChanged(m_indiP_temps, "bias", temps.BIAS);
      updateIfChanged(m_indiP_temps, "water", temps.WATER);
      updateIfChanged(m_indiP_temps, "left", temps.LEFT);
      updateIfChanged(m_indiP_temps, "right", temps.RIGHT);
      updateIfChanged(m_indiP_temps, "cooling", temps.COOLING_POWER);
      return 0;

   }
   else return log<software_error,-1>({__FILE__, __LINE__});

}

inline
int ocam2KCtrl::setTemp(float temp)
{
   std::string response;

   std::string tempStr = std::to_string(temp);
   
   ///\todo make more configurable
   if(temp >= 30 || temp < -40) 
   {
      return log<text_log,-1>({"attempt to set temperature outside valid range: " + tempStr}, logPrio::LOG_ERROR);
   }
   
   if( pdvSerialWriteRead( response, m_pdv, "temp " + tempStr, m_readTimeout) == 0)
   {
      ///\todo check response
      return log<text_log,0>({"set temperature: " + tempStr});
   }
   else return log<software_error,-1>({__FILE__, __LINE__});

}

inline
int ocam2KCtrl::getFPS()
{
   std::string response;

   if( pdvSerialWriteRead( response, m_pdv, "fps", m_readTimeout) == 0)
   {
      float fps;
      if(parseFPS( fps, response ) < 0) 
      {
         if(m_powerState == 0) return -1;
         return log<software_error, -1>({__FILE__, __LINE__, "fps parse error"});
      }
      m_fpsSet = fps;

      updateIfChanged(m_indiP_fps, "current", m_fpsSet);

      return 0;

   }
   else return log<software_error,-1>({__FILE__, __LINE__});

}

inline
int ocam2KCtrl::setFPS(float fps)
{
   std::string response;

   ///\todo should we have fps range checks or let camera deal with it?
   
   std::string fpsStr= std::to_string(fps);
   if( pdvSerialWriteRead( response, m_pdv, "fps " + fpsStr, m_readTimeout) == 0)
   {
      ///\todo check response
      return log<text_log,0>({"set fps: " + fpsStr});
   }
   else return log<software_error,-1>({__FILE__, __LINE__});

}

inline
void ocam2KCtrl::_fgThreadStart( ocam2KCtrl * o)
{
   o->fgThreadExec();
}

inline
int ocam2KCtrl::fgThreadStart()
{
   try
   {
      m_fgThread  = std::thread( _fgThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on framegrabber thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on framegrabber thread start"});
      return -1;
   }

   if(!m_fgThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "framegrabber thread did not start"});
      return -1;
   }

   sched_param sp;
   sp.sched_priority = m_fgThreadPrio;

   int rv = pthread_setschedparam( m_fgThread.native_handle(), SCHED_OTHER, &sp);

   if(rv != 0)
   {
      log<software_error>({__FILE__, __LINE__, rv, "Error setting framegrabber thread params."});
      return -1;
   }

   return 0;

}

inline
void ocam2KCtrl::fgThreadExec()
{

   while(m_shutdown == 0)
   {
      while(m_pdv == nullptr && !m_shutdown)
      {
         sleep(1);
      }
      
      //set up like in EDTocam 
      //start reading, continue as long as m_shutdown == 0 and m_reconfig = 0
      // loop . . .
      
      // if m_reconfig then stop grabbing, then call pdfConfig and pdvInit
      // -- then loop back to the top
      
      
      /* Initialize the OCAM2 SDK
       */

      ocam2_rc rc;
      ocam2_id id;
      ocam2_mode mode;
      unsigned number;

      int OCAM_SZ;
      if(m_height == 121)
      {
         mode = OCAM2_NORMAL;
         OCAM_SZ = 240;
      }
      else if (m_height == 62)
      {
         mode = OCAM2_BINNING;
         OCAM_SZ = 120;
      }
      else
      {
         log<text_log>("Unrecognized OCAM2 mode.", logPrio::LOG_ERROR);
         return;
      }

      std::string ocamDescrambleFile = m_configDir + "/" + m_ocamDescrambleFile;
      
      std::cerr << "ocamDescrambleFile: " << ocamDescrambleFile << "\n";
      rc=ocam2_init(mode, ocamDescrambleFile.c_str(), &id);
      if (rc != OCAM2_OK)
      {
         log<text_log>("ocam2_init error. Failed to initialize OCAM SDK with descramble file: " + ocamDescrambleFile, logPrio::LOG_ERROR);
         return;
      }
      

      log<text_log>("OCAM2K initialized. id: " + std::to_string(id));
      log<text_log>(std::string("OCAM2K mode is:") + ocam2_modeStr(ocam2_getMode(id)));
      
      /* Initialize ImageStreamIO
       */
      //#define SNAME "ocam2ksem"
      IMAGE * imarray;
      //sem_t * sem;
      imarray = (IMAGE *) malloc(sizeof(IMAGE)*100);
      uint32_t imsize[3];
      imsize[0] = OCAM_SZ;
      imsize[1] = OCAM_SZ;
      imsize[2] = 1;
      ImageStreamIO_createIm(&imarray[0], "ocam2k",2, imsize, _DATATYPE_INT16, 1, 0);

      /*
       * allocate four buffers for optimal pdv ring buffer pipeline (reduce if
       * memory is at a premium)
       */
      int numbufs=16;
      pdv_multibuf(m_pdv, numbufs);


      pdv_start_images(m_pdv, numbufs);
      //started = numbufs;
      uint64_t imno = 0;

      //pthread_t fpsThread;
      //pthread_create(&fpsThread,NULL, fps_thread, &imno);
      

      
      int overruns = 0;
      int last_timeouts = 0;
      int recovering_timeout = FALSE;

      while(!m_shutdown)
      {
         u_char *image_p;
      
         /*
          * get the image and immediately start the next one (if not the last
          * time through the loop). Processing (saving to a file in this case)
          * can then occur in parallel with the next acquisition
          */
         image_p = pdv_wait_image(m_pdv);

         imarray[0].md[0].write=1;
         ocam2_descramble(id, &number, imarray[0].array.SI16, (short int *) image_p);
         imarray[0].md[0].cnt0++;
         imarray[0].md[0].cnt1++;
         imarray[0].md[0].write=0;
         ImageStreamIO_sempost(&imarray[0],-1);
         ++imno;

         if ( edt_reg_read(m_pdv, PDV_STAT) & PDV_OVERRUN) ++overruns;

         pdv_start_image(m_pdv);
         int timeouts = pdv_timeouts(m_pdv);

         /*
          * check for timeouts or data overruns -- timeouts occur when data
          * is lost, camera isn't hooked up, etc, and application programs
          * should always check for them. data overruns usually occur as a
          * result of a timeout but should be checked for separately since
          * ROI can sometimes mask timeouts
          */
         if (timeouts > last_timeouts)
         {
            /*
             * pdv_timeout_cleanup helps recover gracefully after a timeout,
             * particularly if multiple buffers were prestarted
             */
            pdv_timeout_restart(m_pdv, TRUE);
            last_timeouts = timeouts;
            recovering_timeout = TRUE;
            log<text_log>("timeout", logPrio::LOG_ERROR);
            printf("\ntimeout....\n");
         } 
         else if (recovering_timeout)
         {
            pdv_timeout_restart(m_pdv, TRUE);
            recovering_timeout = FALSE;
            log<text_log>("restarted", logPrio::LOG_INFO);
            printf("\nrestarted....\n");
         }
      }
    
      
      while(!m_shutdown) sleep(1);
   }

}



INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_ccdtemp)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_ccdtemp.getName())
   {
      float current = 99, target = 99;

      try
      {
         current = ipRecv["current"].get<float>();
      }
      catch(...){}

      try
      {
         target = ipRecv["target"].get<float>();
      }
      catch(...){}

      
      //Check if target is empty
      if( target == 99 ) target = current;
      
      //Now check if it's valid?
      ///\todo implement more configurable max-set-able temperature
      if( target > 30 ) return 0;
      
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      updateIfChanged(m_indiP_ccdtemp, "target", target);
      
      return setTemp(target);
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_mode)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_mode.getName())
   {
      std::cerr << "New mode\n";
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_fps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_fps.getName())
   {
      float current = -99, target = -99;

      try
      {
         current = ipRecv["current"].get<float>();
      }
      catch(...){}
      
      try
      {
         target = ipRecv["target"].get<float>();
      }
      catch(...){}
      
      if(target == -99) target = current;
      
      if(target <= 0) return 0;
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_fps, "target", target);
      
      return setFPS(target);
      
   }
   return -1;
}
}//namespace app
} //namespace MagAOX
#endif
