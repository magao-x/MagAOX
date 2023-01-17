/** \file edtCamera.hpp
  * \brief EDT framegrabber interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef edtCamera_hpp
#define edtCamera_hpp

#ifndef MAGAOX_NOEDT 

#include <edtinc.h>

#include "../../common/paths.hpp"

#include "ioDevice.hpp"

#include "stdCamera.hpp"

namespace MagAOX
{
namespace app
{
namespace dev 
{
   
   


/// MagAO-X EDT framegrabber interface
/** Implements an interface to the EDT PDV SDK
  * 
  * The derived class `derivedT` must be a MagAOXApp\<true\>, and must declare this class a friend like so: 
  * \code
  * friend class dev::dssShutter<derivedT>;
  * \endcode
  *
  * A static configuration variable must be defined in derivedT as
  * \code
  * static constexpr bool c_edtCamera_relativeConfigPath = true; //or: false
  * \endcode
  * which determines whether or not the EDT config path is relative to the MagAO-X config path (true) or absolute (false).
  *
  * In addition, `derivedT` should be `dev::frameGrabber` or equivalent, with an `m_reconfig` member, and calls
  * to this class's `pdvStartAcquisition`, `pdvAcquire`, and `pdvReconfig` in the relevant `dev::frameGrabber`
  * implementation functions.
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`
  * `onPowerOff`, and `whilePowerOff`,  must be placed in the derived class's functions of the same name.
  *
  * \ingroup appdev
  */
template<class derivedT>
class edtCamera : public ioDevice
{

public:
   
   
protected:

   /** \name Configurable Parameters
    * @{
    */
   //Framegrabber:
   int m_unit {0}; ///< EDT PDV board unit number
   int m_channel {0}; ///< EDT PDV board channel number
   int m_numBuffs {4}; ///< EDT PDV DMA buffer size, indicating number of images.
    
   //cameraConfigMap derived().m_cameraModes; ///< Map holding the possible camera mode configurations
   
   //std::string derived().m_startupMode; ///< The camera mode to load during first init after a power-on.
   
   ///@}
   
   PdvDev * m_pdv {nullptr}; ///< The EDT PDV device handle
   
   u_char * m_image_p {nullptr}; ///< The image data grabbed

   //std::string m_modeName; ///< The current mode name
   
   //std::string derived().m_nextMode; ///< The mode to be set by the next reconfiguration
   
   int m_raw_height {0}; ///< The height of the frame, according to the framegrabber
   int m_raw_width {0}; ///< The width of the frame, according to the framegrabber
   int m_raw_depth {0}; ///< The bit-depth of the frame, according to the framegrabber
   std::string m_cameraType; ///< The camera type according to the framegrabber
   
public:

   ///C'tor, sets up stdCamera
   edtCamera();
   
   ///Destructor, destroys the PdvDev structure
   ~edtCamera() noexcept;
   
   /// Send a serial command over cameralink and retrieve the response
   int pdvSerialWriteRead( std::string & response,     ///< [out] the response to the command from the device
                           const std::string & command ///< [in] the command to send to the device
                         );

   /// Configure the EDT framegrabber
   int pdvConfig( std::string & cfgname /**< [in] The configuration name for the mode to set */);

   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       edtCamera<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       edtCamera<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       edtCamera<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// Application logic 
   /** Checks the edtCamera thread
     * 
     * This should be called from the derived's appLogic() as in
     * \code
       edtCamera<derivedT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// Actions on power off
   /**
     * This should be called from the derived's onPowerOff() as in
     * \code
       edtCamera<derivedT>::onPowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int onPowerOff();

   /// Actions while powered off
   /**
     * This should be called from the derived's whilePowerOff() as in
     * \code
       edtCamera<derivedT>::whilePowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int whilePowerOff();
   
   /// Application the shutdown 
   /** Shuts down the edtCamera thread
     * 
     * \code
       edtCamera<derivedT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
   
   int pdvStartAcquisition();
   
   int pdvAcquire( timespec & currImageTimestamp );
   
   int pdvReconfig();
   
protected:
   
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
public:

   /// The static callback function to be registered for the channel properties.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
  // static int st_newCallBack_mode( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
  //                                   const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
  //                               );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   //int newCallBack_mode( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
   
private:
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT>
edtCamera<derivedT>::edtCamera()
{
}

   
template<class derivedT>
edtCamera<derivedT>::~edtCamera() noexcept
{
   if(m_pdv) pdv_close(m_pdv);

   return;
}


#define MAGAOX_PDV_SERBUFSIZE 512

template<class derivedT>
int edtCamera<derivedT>::pdvSerialWriteRead( std::string & response,
                                             const std::string & command
                                           )
{
   char    buf[MAGAOX_PDV_SERBUFSIZE+1];

   // Flush the channel first.
   // This does not indicate errors, so no checks possible.
   pdv_serial_read(m_pdv, buf, MAGAOX_PDV_SERBUFSIZE);

   if( pdv_serial_command(m_pdv, command.c_str()) < 0)
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "PDV: error sending serial command"});
      return -1;
   }

   int ret;

   ret = pdv_serial_wait(m_pdv, m_readTimeout, 1);

   if(ret == 0)
   {
      if(derived().powerState() != 1 || derived().powerStateTarget() != 1) return -1;

      derivedT::template log<software_error>({__FILE__, __LINE__, "PDV: timeout, no serial response"});
      return -1;
   }

   u_char  lastbyte, waitc;

   response.clear();

   do
   {
      ret = pdv_serial_read(m_pdv, buf, MAGAOX_PDV_SERBUFSIZE);

      if(ret > 0) response += buf;

      //Check for last char, wait for more otherwise.
      if (*buf) lastbyte = (u_char)buf[strnlen(buf, sizeof(buf))-1];

      if (pdv_get_waitchar(m_pdv, &waitc) && (lastbyte == waitc))
          break;
      else ret = pdv_serial_wait(m_pdv, m_readTimeout/2, 1);
   }
   while(ret > 0);

   if(ret == 0 && pdv_get_waitchar(m_pdv, &waitc))
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "PDV: timeout in serial response"});
      return -1;
   }

   return 0;
}

template<class derivedT>
int edtCamera<derivedT>::pdvConfig(std::string & modeName)
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
      
   derived().m_modeName = modeName;
   
   if(modeName == "")
   {
      return derivedT::template log<text_log, -1>("Empty modeName passed to pdvConfig", logPrio::LOG_ERROR);
   }
   
   if(derived().m_cameraModes.count(modeName) != 1)
   {
      return derivedT::template log<text_log, -1>("No mode named " + modeName + " found.", logPrio::LOG_ERROR);
   }
   
   //Construct config file, adding relative path if configured that way
   std::string configFile;
   
   if(derivedT::c_edtCamera_relativeConfigPath)
   {
      configFile = derived().configDir() + "/";
   }
   
   configFile += derived().m_cameraModes[modeName].m_configFile;
   
   
   derivedT::template log<text_log>("Loading EDT PDV config file: " + configFile);
      
   if ((dd_p = pdv_alloc_dependent()) == NULL)
   {
      return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "EDT PDV alloc_dependent FAILED"});      
   }
   
   if (pdv_readcfg(configFile.c_str(), dd_p, &edtinfo) != 0)
   {
      free(dd_p);
      return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "EDT PDV readcfg FAILED"});
      
   }
   
   char edt_devname[128];
   strncpy(edt_devname, EDT_INTERFACE, sizeof(edt_devname));
   
   if ((edt_p = edt_open_channel(edt_devname, m_unit, m_channel)) == NULL)
   {
      char errstr[256];
      edt_perror(errstr);
      free(dd_p);
      return derivedT::template log<software_error, -1>({__FILE__, __LINE__, std::string("EDT PDV edt_open_channel FAILED: ") + errstr});
   }
   
   char bitdir[1];
   bitdir[0] = '\0';
    
   int pdv_debug = 0;
   
   if(derived().m_log.logLevel() > logPrio::LOG_INFO) pdv_debug = 2;
   
   if (pdv_initcam(edt_p, dd_p, m_unit, &edtinfo, configFile.c_str(), bitdir, pdv_debug) != 0)
   {
      edt_close(edt_p);
      free(dd_p);
      return derivedT::template log<software_error, -1>({__FILE__, __LINE__, "initcam failed. Run with '--logLevel=DBG' to see complete debugging output."});
   }

   edt_close(edt_p);
   free(dd_p);
   
   //Now open the PDV device handle for talking to the camera via the EDT board.
   if ((m_pdv = pdv_open_channel(edt_devname, m_unit, m_channel)) == NULL)
   {
      std::string errstr = std::string("pdv_open_channel(") + edt_devname + std::to_string(m_unit) + "_" + std::to_string(m_channel) + ")";

      derivedT::template log<software_error>({__FILE__, __LINE__, errstr});
      derivedT::template log<software_error>({__FILE__, __LINE__, errno});

      return -1;
   }

   pdv_flush_fifo(m_pdv);

   pdv_serial_read_enable(m_pdv); //This is undocumented, don't know if it's really needed.
   
   m_raw_width = pdv_get_width(m_pdv);
   m_raw_height = pdv_get_height(m_pdv);
   m_raw_depth = pdv_get_depth(m_pdv);
   m_cameraType = pdv_get_cameratype(m_pdv);

   derivedT::template log<text_log>("Initialized framegrabber: " + m_cameraType);
   derivedT::template log<text_log>("WxHxD: " + std::to_string(m_raw_width) + " X " + std::to_string(m_raw_height) + " X " + std::to_string(m_raw_depth));

   /*
    * allocate four buffers for optimal pdv ring buffer pipeline (reduce if
    * memory is at a premium)
    */
   pdv_multibuf(m_pdv, m_numBuffs);
   derivedT::template log<text_log>("allocated " + std::to_string(m_numBuffs) + " buffers");
  
   
   
   return 0;

}

template<class derivedT>
void edtCamera<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("framegrabber.pdv_unit", "", "framegrabber.pdv_unit", argType::Required, "framegrabber", "pdv_unit", false, "int", "The EDT PDV framegrabber unit number.  Default is 0.");
   config.add("framegrabber.pdv_channel", "", "framegrabber.pdv_channel", argType::Required, "framegrabber", "pdv_channel", false, "int", "The EDT PDV framegrabber channel number.  Default is 0.");
   config.add("framegrabber.numBuffs", "", "framegrabber.numBuffs", argType::Required, "framegrabber", "numBuffs", false, "int", "The EDT PDV framegrabber DMA buffer size [images].  Default is 4.");
   
   dev::ioDevice::setupConfig(config);
}

template<class derivedT>
void edtCamera<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_unit, "framegrabber.pdv_unit");
   config(m_channel, "framegrabber.pdv_channel");
   config(m_numBuffs, "framegrabber.numBuffs");
   
   m_readTimeout = 1000;
   m_writeTimeout = 1000;
   dev::ioDevice::loadConfig(config);

}
   

template<class derivedT>
int edtCamera<derivedT>::appStartup()
{
   if(pdvConfig(derived().m_startupMode) < 0) 
   {
      derivedT::template log<software_error>({__FILE__, __LINE__});
      return -1;
   }
   
   return 0;

}

template<class derivedT>
int edtCamera<derivedT>::appLogic()
{
   return 0;

}

template<class derivedT>
int edtCamera<derivedT>::onPowerOff()
{
   return 0;
}

template<class derivedT>
int edtCamera<derivedT>::whilePowerOff()
{
   return 0;
}

template<class derivedT>
int edtCamera<derivedT>::appShutdown()
{
   return 0;
}

template<class derivedT>
int edtCamera<derivedT>::pdvStartAcquisition()
{
   pdv_start_images(m_pdv, m_numBuffs);
   
   return 0;
}

template<class derivedT>
int edtCamera<derivedT>::pdvAcquire( timespec & currImageTimestamp )
{
   
   uint dmaTimeStamp[2];
   m_image_p = pdv_wait_last_image_timed(m_pdv, dmaTimeStamp);
   //m_image_p = pdv_wait_image_timed(m_pdv, dmaTimeStamp);
   pdv_start_image(m_pdv);

   currImageTimestamp.tv_sec = dmaTimeStamp[0];
   currImageTimestamp.tv_nsec = dmaTimeStamp[1];
   
   
   return 0;
}

template<class derivedT>
int edtCamera<derivedT>::pdvReconfig( )
{
   
   if(pdvConfig(derived().m_nextMode) < 0)
   {
      derivedT::template log<text_log>("error trying to re-configure with " + derived().m_nextMode, logPrio::LOG_ERROR);
      sleep(1);
   }
   else
   {
      derived().m_nextMode = "";
   }
   
   return 0;
}


   
template<class derivedT>
int edtCamera<derivedT>::updateINDI()
{
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //MAGAOX_NOEDT
#endif //edtCamera_hpp
