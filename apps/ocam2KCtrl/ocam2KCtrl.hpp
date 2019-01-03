#ifndef ocam2KCtrl_hpp
#define ocam2KCtrl_hpp


#include <edtinc.h>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"

typedef MagAOX::app::MagAOXApp<true> MagAOXAppT; //This needs to be before pdvUtils.hpp for logging to work.

#include "ocamUtils.hpp"

namespace MagAOX
{
namespace app
{

/** MagAO-X application to control the OCAM 2K EMCCD
  *
  */
class ocam2KCtrl : public MagAOXApp<>
{

protected:

   PdvDev * m_pdv {nullptr};

   int m_unit {0};
   int m_channel {0};

   unsigned long m_powerOnWait {6000000000}; ///< Time in nsec to wait for camera boot after power on.


   float m_fpsSet {0};


   int m_powerOnCounter {0}; ///< Counts numer of loops after power on, implements delay for camera bootup.




public:

   ocam2KCtrl();

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


   int pdvInit();

   int getTemps();
   int getFPS();

   //INDI:
protected:
   //declare our properties
   pcf::IndiProperty m_indiP_temps;
   pcf::IndiProperty m_indiP_binning;
   pcf::IndiProperty m_indiP_fps;

public:
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_temps);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_binning);
   INDI_NEWCALLBACK_DECL(ocam2KCtrl, m_indiP_fps);

};

inline
ocam2KCtrl::ocam2KCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
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
}

inline
void ocam2KCtrl::loadConfig()
{
}

#define MAGAOX_PDV_SERBUFSIZE 512

int pdvSerialWriteRead( std::string & response,
                        PdvDev * pdv,
                        const std::string & command,
                        int timeout
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
-- mutex m_pdv, along with serial communications, but not framegrabbing.
-- need non-mutex way to check for consistency in f.g.-ing, or a way to wait until that loop exits.
-- need way for f.g. loop to communicate errors.
-- separate fps and temp functions.
-- Need startup wait capability.
-- INDI props for fps, temp, both curr and req.  Measured for fps.
   --- Maybe make set-temp separate since it is r/w
-- INDI prop for timestamp of last frame skip.  Maybe total frameskips?
-- INDI props for binning mode
-- Configs to add:
  -- pdv unit number
  -- serial comm timeout.
  -- startup temp command
  -- shutdown temp command

-- add ImageStreamIO
  -- config: filename
  -- buffer length (ser. buffer size)
  */



inline
int ocam2KCtrl::appStartup()
{
   // set up the  INDI properties
   REG_INDI_NEWPROP(m_indiP_temps, "temps", pcf::IndiProperty::Number);
   m_indiP_temps.add (pcf::IndiElement("ccd"));
   m_indiP_temps["ccd"].set(0);
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
   m_indiP_temps.add (pcf::IndiElement("set"));
   m_indiP_temps["set"].set(0);
   m_indiP_temps.add (pcf::IndiElement("cooling"));
   m_indiP_temps["cooling"].set(0);

   REG_INDI_NEWPROP(m_indiP_binning, "binning", pcf::IndiProperty::Number);
   m_indiP_binning.add (pcf::IndiElement("binning"));
   m_indiP_binning["binning"].set(0);

   REG_INDI_NEWPROP(m_indiP_fps, "fps", pcf::IndiProperty::Number);
   m_indiP_fps.add (pcf::IndiElement("value"));
   m_indiP_fps["value"].set(0);

   if(pdvInit() < 0) return -1;

   return 0;

}



inline
int ocam2KCtrl::appLogic()
{

   if( state() == stateCodes::POWERON )
   {
      if(m_powerOnCounter*loopPause > m_powerOnWait)
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

   if( state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      std::string response;

      int ret = pdvSerialWriteRead( response, m_pdv, "fps", 1000);
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
      std::string response;

      if( getFPS() == 0 )
      {
         if(m_fpsSet == 0) state(stateCodes::READY);
         else state(stateCodes::OPERATING);
      }
      else
      {
         state(stateCodes::ERROR);
         return log<software_error,-1>({__FILE__,__LINE__});
      }
   }

   if( state() == stateCodes::READY || state() == stateCodes::OPERATING )
   {
      //INDI mutex here?
      if(getTemps() < 0)
      {
         state(stateCodes::ERROR);
         return 0;
      }

      if(getFPS() < 0)
      {
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
   return 0;
}

inline
int ocam2KCtrl::whilePowerOff()
{
   return onPowerOff();
}

inline
int ocam2KCtrl::appShutdown()
{
   //don't bother
   return 0;
}

inline
int ocam2KCtrl::pdvInit()
{
   if(m_pdv)
   {
      pdv_close(m_pdv);
      m_pdv = nullptr;
   }

   char edt_devname[128];
   strncpy(edt_devname, EDT_INTERFACE, sizeof(edt_devname));

   if ((m_pdv = pdv_open_channel(edt_devname, m_unit, m_channel)) == NULL)
   {
      std::string errstr = std::string("pdv_open_channel(") + edt_devname + std::to_string(m_unit) + "_" + std::to_string(m_channel) + ")";

      log<software_error>({__FILE__, __LINE__, errstr});
      log<software_error>({__FILE__, __LINE__, errno});

      return -1;
   }

   pdv_flush_fifo(m_pdv);

   pdv_serial_read_enable(m_pdv); //This is undocumented, don't know if it's really needed.

   return 0;

}

inline
int ocam2KCtrl::getTemps()
{
   std::string response;

   if( pdvSerialWriteRead( response, m_pdv, "temp", 1000) == 0)
   {
      ocamTemps temps;

      if(parseTemps( temps, response ) < 0) return log<software_error, -1>({__FILE__, __LINE__, "Temp. parse error"});

      log<ocam_temps>({temps.CCD, temps.CPU, temps.POWER, temps.BIAS, temps.WATER, temps.LEFT, temps.RIGHT, temps.COOLING_POWER});

      updateIfChanged(m_indiP_temps, "ccd", temps.CCD);
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
int ocam2KCtrl::getFPS()
{
   std::string response;

   if( pdvSerialWriteRead( response, m_pdv, "fps", 1000) == 0)
   {
      float fps;
      if(parseFPS( fps, response ) < 0) return log<software_error, -1>({__FILE__, __LINE__, "fps parse error"});

      m_fpsSet = fps;

      updateIfChanged(m_indiP_fps, "value", m_fpsSet);

      return 0;

   }
   else return log<software_error,-1>({__FILE__, __LINE__});

}

INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_temps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_temps.getName())
   {
      std::cerr << "New temp\n";
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_binning)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_binning.getName())
   {
      std::cerr << "New binning\n";
      return 0;
   }
   return -1;
}

INDI_NEWCALLBACK_DEFN(ocam2KCtrl, m_indiP_fps)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_fps.getName())
   {
      std::cerr << "New fps\n";
      return 0;
   }
   return -1;
}
}//namespace app
} //namespace MagAOX
#endif
