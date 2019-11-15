/** \file usbtempMon.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup usbtempMon_files
  */

#ifndef usbtempMon_hpp
#define usbtempMon_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

extern "C"
{
#include "usbtemp.h"
}

/** \defgroup usbtempMon
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup usbtempMon_files
  * \ingroup usbtempMon
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup usbtempMon
  */
class usbtempMon : public MagAOXApp<true>, public tty::usbDevice
{

   //Give the test harness access.
   friend class usbtempMon_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}


   struct probe
   {
      std::string m_location;
      std::string m_serial;
      int m_fd {0};
      std::string m_devName;
      
      float m_temperature;
   };
   
   std::vector<probe> m_probes;
   
   
   pcf::IndiProperty m_indiP_temps;
   
public:
   /// Default c'tor.
   usbtempMon();

   /// D'tor, declared and defined for noexcept.
   ~usbtempMon() noexcept
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

   /// Implementation of the FSM for usbtempMon.
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

   
   int checkConnections();

};

usbtempMon::usbtempMon() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void usbtempMon::setupConfig()
{
   config.add("usb.idVendor", "", "usb.idVendor", argType::Required, "usb", "idVendor", false, "string", "USB vendor id, 4 digits");
   config.add("usb.idProduct", "", "usb.idProduct", argType::Required, "usb", "idProduct", false, "string", "USB product id, 4 digits");
}

int usbtempMon::loadConfigImpl( mx::app::appConfigurator & _config )
{

   m_idVendor = "067b";
   _config(m_idVendor, "usb.idVendor");
   m_idProduct = "2303";
   _config(m_idProduct, "usb.idProduct");
   
   
   std::vector<std::string> sections;

   _config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      log<text_log>("No temperature probes found in config.", logPrio::LOG_CRITICAL);
      m_shutdown = 1;
      return -1;
   }
   
   for(size_t i=0; i< sections.size(); ++i)
   {
      if(config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "serial" )))
      {
         m_probes.emplace_back();
         config.configUnused(m_probes.back().m_serial, mx::app::iniFile::makeKey(sections[i], "serial" ));
         m_probes.back().m_serial = mx::ioutils::toLower(m_probes.back().m_serial);
         
         m_probes.back().m_location = sections[i];
         config.configUnused(m_probes.back().m_location, mx::app::iniFile::makeKey(sections[i], "location" ));
      }
   }
   
   return 0;
}

void usbtempMon::loadConfig()
{
   loadConfigImpl(config);
}

int usbtempMon::appStartup()
{
   createROIndiNumber( m_indiP_temps, "temperature", "Temperature [C]");
   for(size_t n =0; n < m_probes.size(); ++n)
   {
      indi::addNumberElement<double>( m_indiP_temps, m_probes[n].m_location, -20., 120., 0, "%0.2f");
      m_indiP_temps[m_probes[n].m_location] = -999;
   }
   registerIndiPropertyReadOnly(m_indiP_temps);
   
   
   return checkConnections();
}

int usbtempMon::appLogic()
{
   bool checkConn = false;
   for(size_t n =0; n < m_probes.size(); ++n)
   {
      if(m_probes[n].m_fd > 0)
      {
         float temperature = -1e37;
         if (DS18B20_acquire(m_probes[n].m_fd, &temperature) < 0) 
         {
            log<software_error>({__FILE__, __LINE__, "Did not get temp from " + m_probes[n].m_location + ". Error: " +  DS18B20_errmsg()});
            DS18B20_close(m_probes[n].m_fd);
            m_probes[n].m_fd = 0;
            checkConn = true;
            continue;
         }
         
         updateIfChanged(m_indiP_temps, m_probes[n].m_location, temperature);
         //std::cout << m_probes[n].m_location << " " << temperature << " C\n";
         
         if (DS18B20_measure(m_probes[n].m_fd) < 0) 
         {
            log<software_error>({__FILE__, __LINE__, "Error from " + m_probes[n].m_location + ". Error: " +  DS18B20_errmsg()});
            DS18B20_close(m_probes[n].m_fd);
            m_probes[n].m_fd = 0;
            updateIfChanged(m_indiP_temps, m_probes[n].m_location, -999);
            checkConn = true;
            continue;
         }
      }
      else
      {
         updateIfChanged(m_indiP_temps, m_probes[n].m_location, -999);
         checkConn = true;
      }
      
      
   }
   
   if(checkConn)
   {
      checkConnections();
   }
   return 0;
}

int usbtempMon::appShutdown()
{
   return 0;
}

int usbtempMon::checkConnections()
{
   std::vector<std::string> devNames;
   
   tty::ttyUSBDevNames(devNames, m_idVendor, m_idProduct);
   
   int nconn = 0;
   for(size_t i=0; i< devNames.size(); ++i)
   {
      elevatedPrivileges ep(this);
      
      int fd = DS18B20_open(devNames[i].c_str());
      if (!is_fd_valid(fd)) 
      {
         log<software_error>({__FILE__,__LINE__, std::string(DS18B20_errmsg())});
         DS18B20_close(fd);
         continue;
      }

      ep.restore();
      
      unsigned char rom[DS18X20_ROM_SIZE];
      char romstr[2*DS18X20_ROM_SIZE+1];
      if (DS18B20_rom(fd, rom) < 0) 
      {
         log<software_error>({__FILE__,__LINE__, std::string(DS18B20_errmsg())});
         DS18B20_close(fd);
         continue;        
      }
      
      for (size_t i = 0; i < DS18X20_ROM_SIZE; i++) 
      {
        snprintf(romstr + 2*i, 3, "%02x", rom[i]);
      }
      
      for(size_t j=0;j< m_probes.size();++j)
      {
         if( m_probes[j].m_serial == romstr)
         {
            if(m_probes[j].m_fd > 0 ) //Check if we're already connected to this one
            {
               if( m_probes[j].m_devName == devNames[i]) //This means we are still good
               {
                  ++nconn;
                  DS18B20_close(fd);
                  fd = 0;
                  break;
               }

               //If here, something changed, and so we close the existing fd and reconnect.
               DS18B20_close(m_probes[j].m_fd);
            }
            
            m_probes[j].m_fd = fd;
            fd = 0;
            m_probes[j].m_devName=devNames[i];
            
            log<text_log>("Found " + m_probes[j].m_location + " ["  + m_probes[j].m_serial + "] as " +  m_probes[j].m_devName);
            
            ++nconn;
            if (DS18B20_measure(m_probes[j].m_fd) < 0) 
            {
               log<software_error>({__FILE__, __LINE__, "Error from " + m_probes[j].m_location + ". Error: " +  DS18B20_errmsg()});
               
               DS18B20_close(m_probes[j].m_fd);
               m_probes[j].m_fd = 0;
            }
            break;
         }
      }
      
      if(fd != 0)
      {
         log<text_log>("no match for " + devNames[i]);
         DS18B20_close(fd);
      }
   }
   
   if(nconn) state(stateCodes::CONNECTED);
   else state(stateCodes::NOTCONNECTED);
   
   return 0;
}


} //namespace app
} //namespace MagAOX

#endif //usbtempMon_hpp
