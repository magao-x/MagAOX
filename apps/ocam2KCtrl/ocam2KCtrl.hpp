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
   
///\todo craete cameraConfig in libMagAOX
struct cameraConfig 
{
   std::string m_configFile;
   std::string m_serialCommand;
   unsigned m_binning {0};
   unsigned m_sizeX {0};
   unsigned m_sizeY {0};
   float m_maxFPS {0};
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
  *  <a href="../apps_html/page_module_ocam2KCtrl.html">Application Documentation</a>
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
  * 
  * \todo INDI prop for timestamp of last frame skip.  Maybe total frameskips?
  * \todo Config item for ImageStreamIO name filename
  * \todo implement ImageStreamIO circular buffer, with config setting
  * \todo calculate frames skipped due to timeouts.
  */
class ocam2KCtrl : public MagAOXApp<>, public dev::ioDevice
{

protected:

   /** \name configurable parameters 
     *@{
     */ 
   //Framegrabber:
   int m_unit {0}; ///< EDT PDV board unit number
   int m_channel {0}; ///< EDT PDV board channel number
   int m_numBuffs {4}; ///< EDT PDV DMA buffer size, indicating number of images.
   
   int m_fgThreadPrio {1}; ///< Priority of the framegrabber thread, should normally be > 00.

   std::string m_shmimName {"ocam2k"}; ///< The name of the shared memory image, is used in `/tmp/<shmimName>.im.shm`.  Default is `ocam2k`.
   
   int m_shmemCubeSz {1}; ///< The size of the shared memory image cube.  Default is 1.

   //Camera:
   unsigned long m_powerOnWait {10}; ///< Time in sec to wait for camera boot after power on.

   cameraConfigMap m_cameraModes; ///< Map holding the possible camera mode configurations
   
   float m_startupTemp {20.0}; ///< The temperature to set after a power-on.
   
   std::string m_startupMode; ///< The camera mode to load during first init after a power-on.
   
   std::string m_ocamDescrambleFile; ///< Path the OCAM 2K pixel descrambling file, relative to MagAO-X config directory.

   ///@}
   
   PdvDev * m_pdv {nullptr}; ///< The EDT PDV device handle

   float m_fpsSet {0}; ///< The commanded fps, as returned by the camera

   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.
   
   std::string m_modeName;
   std::string m_nextMode;
   
   int m_width {0}; ///< The width of the image according to the framegrabber, not necessarily the true image width.
   int m_height {0}; ///< The height of the image frame according the framegrabber, not necessarily the true image height.
   int m_depth {0}; ///< The pixel bit depth according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
    
      
   long m_currImageNumber {-1};
   double m_currImageTimestamp {0};
   double m_currImageDMATimestamp {0};
      
   long m_lastImageNumber {-1};
      
   long m_firstGoodImageNumber {-1};
   double m_firstGoodImageTimestamp {0};
   double m_firstGoodImageDMATimestamp {0};
   
   long m_framesSkipped = 0;
      
   bool m_resetFPS {false};
   bool m_reconfig {false};
   
   unsigned m_emGain {1};
   
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
   
   int getEMGain();
   
   int setEMGain( unsigned emg );
   
protected:
   

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
   pcf::IndiProperty m_indiP_emGain;

public:
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_ccdtemp);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_mode);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_fps);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_emGain);

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
   config.add("framegrabber.pdv_unit", "", "framegrabber.pdv_unit", argType::Required, "framegrabber", "pdv_unit", false, "int", "The EDT PDV framegrabber unit number.  Default is 0.");
   config.add("framegrabber.pdv_channel", "", "framegrabber.pdv_channel", argType::Required, "framegrabber", "pdv_channel", false, "int", "The EDT PDV framegrabber channel number.  Default is 0.");
   config.add("framegrabber.numBuffs", "", "framegrabber.numBuffs", argType::Required, "framegrabber", "numBuffs", false, "int", "The EDT PDV framegrabber DMA buffer size [images].  Default is 4.");
   config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the fraemgrabber thread.");
   
   config.add("framegrabber.shmimName", "", "framegrabber.shmimName", argType::Required, "framegrabber", "shmimName", false, "string", "The name of the ImageStreamIO shared memory image. Will be used as /tmp/<shmimName>.im.shm.");
   
   config.add("framegrabber.shmemCubeSz", "", "framegrabber.shmemCubeSz", argType::Required, "framegrabber", "shmemCubeSz", false, "int", "The cube size (number of images) in the shared memory buffer.");
   
   config.add("camera.powerOnWait", "", "camera.powerOnWait", argType::Required, "camera", "powerOnWait", false, "int", "Time after power-on to begin attempting connections [sec].  Default is 10 sec.");
   
   config.add("camera.startupTemp", "", "camera.startupTemp", argType::Required, "camera", "startupTemp", false, "float", "The temperature setpoint to set after a power-on [C].  Default is 20 C.");
   
   config.add("camera.startupMode", "", "camera.startupMode", argType::Required, "camera", "startupMode", false, "string", "The name of the configuration to set at startup.");
   
   config.add("camera.ocamDescrambleFile", "", "camera.ocamDescrambleFile", argType::Required, "camera", "ocamDescrambleFile", false, "string", "The path of the OCAM descramble file, relative to MagAOX/config.");
   
   dev::ioDevice::setupConfig(config);
}



inline
void ocam2KCtrl::loadConfig()
{
   config(m_unit, "framegrabber.pdv_unit");
   config(m_channel, "framegrabber.pdv_channel");
   config(m_numBuffs, "framegrabber.numBuffs");
   config(m_fgThreadPrio, "framegrabber.threadPrio");
   config(m_shmimName, "framegrabber.shmimName");
   config(m_shmemCubeSz, "framegrabber.shmemCubeSz");
   
   config(m_powerOnWait, "camera.powerOnWait");
   config(m_startupTemp, "camera.startupTemp");
   config(m_startupMode, "camera.startupMode");
   config(m_ocamDescrambleFile, "camera.ocamDescrambleFile");
   
   int rv = loadCameraConfig(m_cameraModes, config);
   
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
   m_indiP_fps.add (pcf::IndiElement("measured"));

   REG_INDI_NEWPROP(m_indiP_emGain, "emgain", pcf::IndiProperty::Number);
   m_indiP_emGain.add (pcf::IndiElement("current"));
   m_indiP_emGain["current"].set(m_emGain);
   m_indiP_emGain.add (pcf::IndiElement("target"));
   
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
   
   if( state() == stateCodes::POWERON )
   {
      if(m_powerOnCounter*m_loopPause > ((double) m_powerOnWait)*1e9)
      {
         state(stateCodes::NOTCONNECTED);
         m_reconfig = true; //Trigger a f.g. thread reconfig.
         m_powerOnCounter = 0;
      }
      else
      {
         ++m_powerOnCounter;
         return 0;
      }
   }

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
   updateIfChanged(m_indiP_fps, "measured", 0);
   
   updateIfChanged(m_indiP_emGain, "current", 0);
   updateIfChanged(m_indiP_emGain, "target", 0);
   
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
   if(m_fgThread.joinable())
   {
      m_fgThread.join();
   }
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
      
   m_modeName = modeName;
   if(m_indiDriver)
   {
      updateIfChanged(m_indiP_mode, "target", m_modeName);
   }
   
   if(m_cameraModes.count(modeName) != 1)
   {
      return log<text_log, -1>("No mode named " + modeName + " found.", logPrio::LOG_ERROR);
   }
   
   std::string configFile = m_configDir + "/" +m_cameraModes[modeName].m_configFile;
   
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

   log<text_log>("Initialized framegrabber: " + m_cameraType);
   log<text_log>("WxHxD: " + std::to_string(m_width) + " X " + std::to_string(m_height) + " X " + std::to_string(m_depth));

   
  
   
   log<text_log>("camera configured with: " +m_cameraModes[modeName].m_serialCommand);
   
   
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

      double fpsMeas;
      
      if(m_currImageNumber > m_firstGoodImageNumber)
      {
         fpsMeas = (m_currImageNumber - m_firstGoodImageNumber)/(m_currImageTimestamp-m_firstGoodImageTimestamp);
      }
      else
      {
         fpsMeas = ( std::numeric_limits<unsigned int>::max() - m_firstGoodImageNumber + m_currImageNumber)/(m_currImageTimestamp-m_firstGoodImageTimestamp);
      }
      
      updateIfChanged(m_indiP_fps, "measured", fpsMeas);
      
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
      log<text_log>({"set fps: " + fpsStr});
      
      m_resetFPS = true;
      
      return 0;
   }
   else return log<software_error,-1>({__FILE__, __LINE__});

}

inline
int ocam2KCtrl::getEMGain()
{
   std::string response;

   if( pdvSerialWriteRead( response, m_pdv, "gain", m_readTimeout) == 0)
   {
      unsigned emGain;
      if(parseEMGain( emGain, response ) < 0) 
      {
         if(m_powerState == 0) return -1;
         return log<software_error, -1>({__FILE__, __LINE__, "EM Gain parse error"});
      }
      m_emGain = emGain;

      updateIfChanged(m_indiP_emGain, "current", m_emGain);
      
      return 0;

   }
   else return log<software_error,-1>({__FILE__, __LINE__});
}
   
inline
int ocam2KCtrl::setEMGain( unsigned emg )
{
   return 0;
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

   //Now set the RT priority.
   
   int prio=m_fgThreadPrio;
   if(prio < 0) prio = 0;
   if(prio > 99) prio = 99;

   sched_param sp;
   sp.sched_priority = prio;

   //Get the maximum privileges available
   if( euidCalled() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to called failed."});
      return -1;
   }
   
   //We set return value based on result from sched_setscheduler
   //But we make sure to restore privileges no matter what happens.
   errno = 0;
   int rv = 0;
   if(prio > 0) rv = pthread_setschedparam(m_fgThread.native_handle(), MAGAOX_RT_SCHED_POLICY, &sp);
   else rv = pthread_setschedparam(m_fgThread.native_handle(), SCHED_OTHER, &sp);
   
   //Go back to regular privileges
   if( euidReal() < 0 )
   {
      log<software_error>({__FILE__, __LINE__, "Setting euid to real failed."});
   }
   
   if(rv < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__, errno, "Setting F.G. thread scheduler priority to " + std::to_string(prio) + " failed."});
   }
   else
   {
      return log<text_log,0>("F.G. thread scheduler priority (framegrabber.threadPrio) set to " + std::to_string(prio));
   }
   

}

inline
void ocam2KCtrl::fgThreadExec()
{

   //Create and initialize the ImageStreamIO buffer.
   //It will only be recreated if needed.
   IMAGE imageStream;
   uint32_t imsize[3];
   imsize[0] = 120;
   imsize[1] = 120;
   imsize[2] = 1;
   ImageStreamIO_createIm(&imageStream, m_shmimName.c_str(), 2, imsize, _DATATYPE_INT16, 1, 0);
      
   while(m_shutdown == 0)
   {
      while(!m_shutdown && (!( state() == stateCodes::READY || state() == stateCodes::OPERATING) || m_powerState <= 0 || m_pdv == nullptr ) )
      {
         sleep(1);
      }
      
      if(m_shutdown) continue;
      else //This gives a nice scope for the mutex
      {
         std::unique_lock<std::mutex> lock(m_indiMutex);
      
         //Send command to camera to place it in the correct mode
         std::string response;
         if( pdvSerialWriteRead( response, m_pdv, m_cameraModes[m_modeName].m_serialCommand, m_readTimeout) != 0)
         {
            log<software_error>({__FILE__, __LINE__, "Error sending command to set mode"});
            sleep(1);
            continue;
         }
         
         if(m_fpsSet > 0) setFPS(m_fpsSet);
         
         log<text_log>("Send command to set mode: " + m_cameraModes[m_modeName].m_serialCommand);
         log<text_log>("Response was: " + response);
         
         updateIfChanged(m_indiP_mode, "current", m_modeName);
         updateIfChanged(m_indiP_mode, "target", std::string(""));
      }
   
      ///\todo check response
   
   
      /* Initialize the OCAM2 SDK
       */

      ocam2_rc rc;
      ocam2_id id;
      ocam2_mode mode;
   
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
      if(OCAM_SZ != imageStream.md[0].size[0] || OCAM_SZ != imageStream.md[0].size[1])
      {
         std::cerr << "recreating ImageStream\n";
         imsize[0] = OCAM_SZ;
         imsize[1] = OCAM_SZ;
         imsize[2] = 1;
         ImageStreamIO_createIm(&imageStream, m_shmimName.c_str(), 2, imsize, _DATATYPE_INT16, 1, 0);
      }
      else
      {
         std::cerr << "not recreating\n";
      }
      
      /*
       * allocate four buffers for optimal pdv ring buffer pipeline (reduce if
       * memory is at a premium)
       */
      pdv_multibuf(m_pdv, m_numBuffs); ///\todo this can be done in the main config fxn.
      pdv_start_images(m_pdv, m_numBuffs);
      
      //This completes the reconfiguration.
      m_reconfig = false;
                  
      //Trigger an FPS reset.
      m_lastImageNumber = -1;
      
      //This is the main image grabbing loop.
      // - It waits for a new image DMA to complete
      // - Then checks for timeouts or overrun
      // - Then checks image number for consistency 
      // - Then loads the image into the stream 
      while(!m_shutdown && !m_reconfig && m_powerState > 0)
      {
         if( state() != stateCodes::OPERATING || m_powerState <= 0)
         {
            sleep(1);
            continue;
         }
         
         u_char *image_p;
         bool overrun = false;
         long framesSkipped = 0;
         
         int last_timeouts = 0;

         unsigned currImageNumber = 0;
      
         /*
          * get the image and immediately start the next one (if not the last
          * time through the loop). Processing (saving to a file in this case)
          * can then occur in parallel with the next acquisition
          */
         uint dmaTimeStamp[2];
         timespec timeStamp;
         image_p = pdv_wait_last_image_timed(m_pdv, dmaTimeStamp);
         pdv_start_image(m_pdv);


         /* Removed all pdv timeout and overrun checking, since we can rely on frame number from the camera
            to detect missed and corrupted frames.
         
            See ef0dd24 for last version with full checks in it.
         */
        
         //Get the image number to see if this is valid.
         //This is how it is in the ocam2_sdk:
         currImageNumber = ((int *)image_p)[OCAM2_IMAGE_NB_OFFSET/4]; /* int offset */
         
         //We want to do arithmetic in signed long, big enough to hold the unsigned counter from OCAM2
         m_currImageNumber = currImageNumber;
         
         //For the first loop after a restart
         if(m_lastImageNumber == -1 || m_resetFPS) 
         {
            m_lastImageNumber = m_currImageNumber - 1;
            
            //Update this for fps meter
            m_firstGoodImageNumber = m_currImageNumber;
            m_firstGoodImageTimestamp = m_currImageTimestamp;
            m_firstGoodImageDMATimestamp = m_currImageDMATimestamp;
            
            m_framesSkipped = 0;
            
            m_resetFPS = false;
         }
            
      
         if(m_currImageNumber - m_lastImageNumber != 1)
         {
            //Detect exact condition of a wraparound on the unsigned int.
            // Yes, this can only happen once every 13.72 days at 3622 fps 
            // But just in case . . .
            if(m_lastImageNumber != std::numeric_limits<unsigned int>::max() && m_currImageNumber != 0)
            {
               //The far more likely case is a problem...
         
               //If a reasonably small number of frames skipped, then we trust the image number
               if(m_currImageNumber - m_lastImageNumber > 1 && m_currImageNumber - m_lastImageNumber < 100)
               { 
                  //This we handle as a non-timeout -- report how many frames were skipped
                  framesSkipped = m_currImageNumber - m_lastImageNumber;
                  //and don't `continue` to top of loop
                  
                  log<text_log>("frames skipped: " + std::to_string(framesSkipped) + ", total since restart: " + std::to_string(m_framesSkipped), logPrio::LOG_ERROR);
                  
                  m_nextMode = m_modeName;
                  m_reconfig = 1;
            
                  /*pdv_timeout_restart(m_pdv, FALSE);
                  pdv_multibuf(m_pdv, m_numBuffs);
                  pdv_start_images(m_pdv, m_numBuffs);
                  m_lastImageNumber = -1;
                  */
                  
                  continue;
                  
               }
               else //but if it's any bigger or < 0, it's probably garbage
               {
                  //This we handle as corruption, therefore a timeout
//                   pdv_timeout_restart(m_pdv, FALSE);
//                   pdv_multibuf(m_pdv, m_numBuffs);
//                   pdv_start_images(m_pdv, m_numBuffs);
//                   last_timeouts = timeouts;
               
                  ///\todo need frame corrupt log type
                  log<text_log>("frame number possibly corrupt: " + std::to_string(m_currImageNumber) + " - " + std::to_string(m_lastImageNumber), logPrio::LOG_ERROR);
                  
                  m_nextMode = m_modeName;
                  m_reconfig = 1;
            
                  //Reset the counters.
                  m_lastImageNumber = -1;
                  
                  continue;
               
               }
            }
         }

         
         
         //Ok, no timeout, so we process the image and publish it.
         imageStream.md[0].write=1; ///\todo make sure rtimv skips image if write=1
         ocam2_descramble(id, &currImageNumber, imageStream.array.SI16, (short int *) image_p);
         
         imageStream.md[0].atime.tv_sec = dmaTimeStamp[0];
         imageStream.md[0].atime.tv_nsec = dmaTimeStamp[1];
         clock_gettime(CLOCK_REALTIME, &timeStamp); 
         ///\todo need to update the write time in imagestruct with this timeStamp
         imageStream.md[0].cnt0++;
         imageStream.md[0].cnt1++; ///\todo this is wrong, needs to be 0 if needed, and needs to be done *before* image write for cube handling.
         imageStream.md[0].write=0;
         ImageStreamIO_sempost(&imageStream,-1);
 
         
         
         if(imageStream.md[0].cnt0 % 2000 == 0)
         {
            std::cerr << ( (double) timeStamp.tv_sec + ((double) timeStamp.tv_nsec)/1e9) - ( (double) dmaTimeStamp[0] + ((double) dmaTimeStamp[1])/1e9) << "\n";
         }
         
         if(framesSkipped)
         {
            m_framesSkipped += framesSkipped;
            
            ///\todo need frame skip log type
            log<text_log>("frames skipped: " + std::to_string(framesSkipped) + ", total since restart: " + std::to_string(m_framesSkipped), logPrio::LOG_ERROR);
         }
         
         m_lastImageNumber = m_currImageNumber;
      }
    
      if(m_reconfig && !m_shutdown)
      {
         //lock mutex
         std::unique_lock<std::mutex> lock(m_indiMutex);
         
         if(pdvConfig(m_nextMode) < 0)
         {
            log<text_log>("error trying to re-configure with " + m_nextMode, logPrio::LOG_ERROR);
            sleep(1);
         }
         else
         {
            m_nextMode = "";
            
         }
      }

   } //outer loop, will exit if m_shutdown==true

   ImageStreamIO_destroyIm( &imageStream );
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
      std::string current;
      std::string target;
      try 
      {
         current = ipRecv["current"].get();
      }
      catch(...)
      {
         current = "";
      }
      
      try 
      {
         target = ipRecv["target"].get();
      }
      catch(...)
      {
         target = "";
      }
      
      if(target == "") target = current;
      
      if(m_cameraModes.count(target) == 0 )
      {
         return log<text_log, -1>("Unrecognized mode requested: " + target, logPrio::LOG_ERROR);
      }
      
      updateIfChanged(m_indiP_mode, "target", target);
      
      //Now signal the f.g. thread to reconfigure
      m_nextMode = target;
      m_reconfig = true;
      
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

INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_emGain)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_emGain.getName())
   {
      unsigned current = 0, target = 0;

      if(ipRecv.find("current"))
      {
         current = ipRecv["current"].get<unsigned>();
      }

      if(ipRecv.find("target"))
      {
         target = ipRecv["target"].get<unsigned>();
      }
      
      if(target == 0) target = current;
      
      if(target == 0) return 0;
      
      //Lock the mutex, waiting if necessary
      std::unique_lock<std::mutex> lock(m_indiMutex);

      updateIfChanged(m_indiP_emGain, "target", target);
      
      return setEMGain(target);
      
   }
   return -1;
}

}//namespace app
} //namespace MagAOX
#endif
