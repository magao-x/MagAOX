/** \file dmMode.hpp
  * \brief The MagAO-X DM mode command header file
  *
  * \ingroup dmMode_files
  */

#ifndef dmMode_hpp
#define dmMode_hpp

#include <mx/improc/eigenCube.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/stringUtils.hpp>
#include <mx/sys/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup dmMode
  * \brief The DM mode command app, places modes on a DM channel
  * \todo update md doc
  * \todo the current_amps/target_amps thing is dumb.  Should consider mode00.target, mode00.current maybe.
  *
  * <a href="../handbook/operating/software/apps/dmMode.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup dmMode_files
  * \ingroup dmMode
  */


namespace MagAOX
{
namespace app
{

/// The MagAO-X DM mode commander
/** 
  * \ingroup dmMode
  */
class dmMode : public MagAOXApp<true>, public dev::telemeter<dmMode>
{

   typedef float realT;
   
   typedef dev::telemeter<dmMode> telemeterT;

   friend class dev::telemeter<dmMode>;
   friend class dmMode_test;

protected:

   /** \name Configurable Parameters
     *@{
     */

   std::string m_modeCube;

   int m_maxModes {50};
   
   std::string m_dmName;
   
   std::string m_dmChannelName;
   
   ///@}

   mx::improc::eigenCube<realT> m_modes;
   
   std::vector<realT> m_amps;
   
   mx::improc::eigenImage<realT> m_shape;
   
   IMAGE m_imageStream; 
   uint32_t m_width {0}; ///< The width of the image
   uint32_t m_height {0}; ///< The height of the image.
   
   uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
   size_t m_typeSize {0}; ///< The size of the type, in bytes.  
   
   bool m_opened {true};
   bool m_restart {false};
   
public:
   /// Default c'tor.
   dmMode();

   /// D'tor, declared and defined for noexcept.
   ~dmMode() noexcept
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

   /// Implementation of the FSM for dmMode.
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


   int sendCommand();
   
   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_dm;
   pcf::IndiProperty m_indiP_currAmps;
   pcf::IndiProperty m_indiP_tgtAmps;

   std::vector<std::string> m_elNames;
public:
   INDI_NEWCALLBACK_DECL(dmMode, m_indiP_currAmps);
   INDI_NEWCALLBACK_DECL(dmMode, m_indiP_tgtAmps);

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_dmmodes * );
   
   int recordDmModes( bool force = false );
   ///@}


};

dmMode::dmMode() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void dmMode::setupConfig()
{
   config.add("dm.modeCube", "", "dm.modeCube", argType::Required, "dm", "modeCube", false, "string", "Full path to the FITS file containing the modes for this DM.");
   config.add("dm.maxModes", "", "dm.maxModes", argType::Required, "dm", "maxModes", false, "int", "The maximum number of modes to use (truncates the cube).  If <=0 all modes in cube are used.");
   config.add("dm.name", "", "dm.name", argType::Required, "dm", "name", false, "string", "The descriptive name of this dm. Default is the channel name.");
   config.add("dm.channelName", "", "dm.channelName", argType::Required, "dm", "channelName", false, "string", "The name of the DM channel to write to.");
   config.add("dm.maxModes", "", "dm.maxModes", argType::Required, "dm", "maxModes", false, "int", "The maximum number of modes to use (truncates the cube).");

   telemeterT::setupConfig(config);
}

int dmMode::loadConfigImpl( mx::app::appConfigurator & _config )
{

   _config(m_modeCube, "dm.modeCube");
   _config(m_maxModes, "dm.maxModes");
   _config(m_dmChannelName, "dm.channelName");
   
   m_dmName = m_dmChannelName;
   _config(m_dmName, "dm.name");
   
   if(telemeterT::loadConfig(_config) < 0)
   {
      log<text_log>("Error during telemeter config", logPrio::LOG_CRITICAL);
      m_shutdown = true;
   }

   return 0;
}

void dmMode::loadConfig()
{
   loadConfigImpl(config);
}

int dmMode::appStartup()
{
   mx::fits::fitsFile<realT> ff;
   
   if(ff.read(m_modes, m_modeCube) < 0) 
   {
      return log<text_log,-1>("Could not open mode cube file", logPrio::LOG_ERROR);
   }

   if(m_maxModes > 0 && m_maxModes < m_modes.planes())
   {
      mx::improc::eigenCube<realT> modes;
      //This probably just works as a realloc in eigenCube but I haven't looked.
      modes.resize(m_modes.rows(), m_modes.cols(), m_maxModes);
      for(int p =0; p < modes.planes(); ++p) modes.image(p) = m_modes.image(p);
      m_modes.resize(m_modes.rows(), m_modes.cols(), m_maxModes);
      for(int p =0; p < modes.planes(); ++p) m_modes.image(p) = modes.image(p);
   }


   
   m_amps.resize(m_modes.planes(), 0);
   m_shape.resize(m_modes.rows(), m_modes.cols());
   
   REG_INDI_NEWPROP_NOCB(m_indiP_dm, "dm", pcf::IndiProperty::Text);
   m_indiP_dm.add(pcf::IndiElement("name"));
   m_indiP_dm["name"] = m_dmName;
   m_indiP_dm.add(pcf::IndiElement("channel"));
   m_indiP_dm["channel"] = m_dmChannelName;
   
   REG_INDI_NEWPROP(m_indiP_currAmps, "current_amps", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP(m_indiP_tgtAmps, "target_amps", pcf::IndiProperty::Number);
   
   m_elNames.resize(m_amps.size());
   
   for(size_t n=0; n < m_amps.size(); ++n)
   {
      //std::string el = std::to_string(n);
      m_elNames[n] = mx::ioutils::convertToString<size_t, 4, '0'>(n);
      
      m_indiP_currAmps.add( pcf::IndiElement(m_elNames[n]) );
      m_indiP_currAmps[m_elNames[n]].set(0);
   
      m_indiP_tgtAmps.add( pcf::IndiElement(m_elNames[n]) );
   }
   
   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   state(stateCodes::NOTCONNECTED);
   
   
   
   return 0;
}

int dmMode::appLogic()
{
   if(state() == stateCodes::NOTCONNECTED)
   {
      m_opened = false;
      m_restart = false; //Set this up front, since we're about to restart.
      
      if( ImageStreamIO_openIm(&m_imageStream, m_dmChannelName.c_str()) == 0)
      {
         if(m_imageStream.md[0].sem < 10) ///<\todo this is hardcoded in ImageStreamIO.c -- should be a define
         {
            ImageStreamIO_closeIm(&m_imageStream);
         }
         else
         {
            m_opened = true;
         }
      }
      
      if(m_opened)
      {
         state(stateCodes::CONNECTED);
      }
   }
   
   if(state() == stateCodes::CONNECTED)
   {
      m_dataType = m_imageStream.md[0].datatype;
      m_typeSize = ImageStreamIO_typesize(m_dataType);
      m_width = m_imageStream.md[0].size[0];
      m_height = m_imageStream.md[0].size[1];
   
   
      if(m_dataType != _DATATYPE_FLOAT )
      {
         return log<text_log,-1>("Data type of DM channel is not float.", logPrio::LOG_CRITICAL);
      }
   
      if(m_typeSize != sizeof(realT))
      {
         return log<text_log,-1>("Type-size mismatch, realT is not float.", logPrio::LOG_CRITICAL);
      }
   
      if(m_width != m_modes.rows())
      {
         return log<text_log,-1>("Size mismatch between DM and modes (rows)", logPrio::LOG_CRITICAL);
      }
      
      if(m_height != m_modes.cols())
      {
         return log<text_log,-1>("Size mismatch between DM and modes (cols)", logPrio::LOG_CRITICAL);
      }
      
      for(size_t n=0; n < m_amps.size(); ++n) m_amps[n] = 0;
      sendCommand();
      
      state(stateCodes::READY);
   }
   
   if(state() == stateCodes::READY)
   {
      if(telemeterT::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
   }

   return 0;
}

int dmMode::appShutdown()
{
   telemeterT::appShutdown();

   return 0;
}

int dmMode::sendCommand()
{
   if(!m_opened)
   {
      log<text_log>("not connected to DM channel.", logPrio::LOG_WARNING);
      return 0;
   }
   
   m_shape = m_amps[0]*m_modes.image(0);
   
   for(size_t n = 1; n<m_amps.size(); ++n)
   {
      m_shape += m_amps[n]*m_modes.image(n);
   }
   
   if(m_imageStream.md[0].write)
   {
      while(m_imageStream.md[0].write) mx::sys::microSleep(10);
   }
   
   recordDmModes(true);
   m_imageStream.md[0].write = 1;
   
   uint32_t curr_image;
   if(m_imageStream.md[0].size[2] > 0) ///\todo change to naxis?
   {
      curr_image = m_imageStream.md[0].cnt1;
   }
   else curr_image = 0;
   
   char* next_dest = (char *) m_imageStream.array.raw + curr_image*m_width*m_height*m_typeSize;
   
   memcpy(next_dest, m_shape.data(), m_width*m_height*m_typeSize);
      
   m_imageStream.md[0].cnt0++;
   
   m_imageStream.md->write=0;
   ImageStreamIO_sempost(&m_imageStream,-1);
   
   recordDmModes(true);
   
   for(size_t n = 0; n<m_amps.size(); ++n)
   {
      m_indiP_currAmps[m_elNames[n]] = m_amps[n];
   }
   m_indiP_currAmps.setState (pcf::IndiProperty::Ok);
   m_indiDriver->sendSetProperty (m_indiP_currAmps);

   return 0;
   
}

INDI_NEWCALLBACK_DEFN(dmMode, m_indiP_currAmps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_currAmps.getName())
   {
      size_t found = 0;
      for(size_t n=0; n < m_amps.size(); ++n)
      {
         if(ipRecv.find(m_elNames[n]))
         {
            realT amp = ipRecv[m_elNames[n]].get<realT>();
            
            ///\todo add bounds checks here
            
            m_amps[n] = amp;
            ++found;
         }
      }
      
      if(found) 
      {
         return sendCommand();
      }
      
      return 0;
      
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "invalid indi property name"});
}

INDI_NEWCALLBACK_DEFN(dmMode, m_indiP_tgtAmps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_tgtAmps.getName())
   {
      size_t found = 0;
      for(size_t n=0; n < m_amps.size(); ++n)
      {
         if(ipRecv.find(m_elNames[n]))
         {
            realT amp = ipRecv[m_elNames[n]].get<realT>();
            
            ///\todo add bounds checks here
            
            m_amps[n] = amp;
            ++found;
         }
      }
      
      if(found) 
      {
         return sendCommand();
      }
      
      return 0;
      
   }
   
   return log<software_error,-1>({__FILE__,__LINE__, "invalid indi property name"});
}

int dmMode::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_dmmodes());
}
   
int dmMode::recordTelem( const telem_dmmodes * )
{
   return recordDmModes(true);
}

int dmMode::recordDmModes( bool force )
{
   static std::vector<float> lastamps(m_amps.size(), std::numeric_limits<float>::max());
   
   bool changed = false;
   for(size_t p=0; p < m_amps.size(); ++p)
   {
      if(m_amps[p] != lastamps[p]) changed = true;
   }
   
   if( changed || force )
   {
      for(size_t p=0; p < m_amps.size(); ++p)
      {
         lastamps[p] = m_amps[p];
      }
   
      telem<telem_dmmodes>(telem_dmmodes::messageT(lastamps));
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //dmMode_hpp
