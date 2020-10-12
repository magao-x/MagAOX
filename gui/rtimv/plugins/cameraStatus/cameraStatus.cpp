
#include "cameraStatus.hpp"

cameraStatus::cameraStatus() : rtimvOverlayInterface()
{
}

cameraStatus::~cameraStatus()
{
}

int cameraStatus::attachOverlay( rtimvGraphicsView* gv, 
                                 dictionaryT * dict,
                                 mx::app::appConfigurator & config
                               )
{   
   m_gv = gv;
   
   m_dict = dict;
   
   config.configUnused(m_deviceName, mx::app::iniFile::makeKey("camera", "name"));
   
   if(m_deviceName == "")
   {
      m_enableable = false;
      disableOverlay();
      return 0;
   }
   
   m_enableable = true;
   m_enabled = true;
   std::cerr << "cameraStatus deviceName: " << m_deviceName << "\n";
   
   config.configUnused(m_filterDeviceName, mx::app::iniFile::makeKey("camera", "filterDevice"));
   
   
   if(m_enabled) enableOverlay();
   else disableOverlay();
   
   return 0;
}
      
int cameraStatus::updateOverlay()
{
   if(!m_enabled) return 0;
   
   if(m_dict == nullptr) return 0;
   
   if(m_gv == nullptr) return 0;
   
   size_t n = 0;
   char * str;
   char tstr[128];
   std::string sstr;
   
   if( m_dict->count(m_deviceName + ".temp_ccd.current") > 0)
   {
      str = (char *)(*m_dict)[m_deviceName + ".temp_ccd.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "%0.1f C", strtod(str,0));
      m_gv->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_gv->statusTextNo()-1) return 0;
   
   if( m_dict->count(m_deviceName + ".exptime.current") > 0)
   {
      str = (char *)(*m_dict)[m_deviceName + ".exptime.current"].m_blob;
      double et = strtod(str,0);
      if(et >= 100)
      {
        snprintf(tstr, sizeof(tstr), "%0.1f s", et);
      }
      else if(et >= 10)
      {
         snprintf(tstr, sizeof(tstr), "%0.2f s", et);
      }
      else if(et >= 0.1)
      {
         snprintf(tstr, sizeof(tstr), "%0.3f s", et);
      }
      else
      {
         snprintf(tstr, sizeof(tstr), "%0.3e s", et);
      }
      m_gv->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_gv->statusTextNo()-1) return 0;
   
   if( m_dict->count(m_deviceName + ".fps.current") > 0)
   {
      str = (char *)(*m_dict)[m_deviceName + ".fps.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "%0.1f FPS", strtod(str,0));
      m_gv->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_gv->statusTextNo()-1) return 0;
   
   if( m_dict->count(m_deviceName + ".emgain.current") > 0)
   {
      str = (char *)(*m_dict)[m_deviceName + ".emgain.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "EMG: %d", atoi(str));
      m_gv->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_gv->statusTextNo()-1) return 0;
   
   if( m_dict->count(m_deviceName + ".shutter_status.status") > 0)
   {
      sstr = std::string((char *)(*m_dict)[m_deviceName + ".shutter_status.status"].m_blob);
      
      if(sstr == "READY")
      {
         if( m_dict->count(m_deviceName + ".shutter.toggle") > 0)
         {
            sstr = std::string((char *)(*m_dict)[m_deviceName + ".shutter.toggle"].m_blob);
            if(sstr == "on") sstr = "SHUT";
            else sstr = "OPEN";
            m_gv->statusTextText(n, sstr.c_str());
            ++n;
         }
      }
      else
      {
         sstr = "sh " + sstr;
         m_gv->statusTextText(n, sstr.c_str());
         ++n;
      }
   }
         
   if(n > m_gv->statusTextNo()-1) return 0;   
   
   if(m_filterDeviceName != "")
   {
      if( m_dict->count(m_filterDeviceName + ".fsm.state") > 0)
      {
         std::string fwstate = std::string((char *)(*m_dict)[m_filterDeviceName + ".fsm.state"].m_blob);
         
         if(fwstate == "READY" || fwstate == "OPERATING")
         {
            dictionaryIteratorT start = m_dict->lower_bound(m_filterDeviceName + ".filterName");
            dictionaryIteratorT end = m_dict->upper_bound(m_filterDeviceName + ".filterNameZ");
            
            std::string fkey;
            while(start != end)
            {
//               std::cerr << start->first << " " << std::string((char*)start->second.m_blob) << "\n";
               if(std::string((char*)start->second.m_blob) == "on")
               {
                  fkey = start->first;
                  break;
               }
               ++start;
            }
            
            size_t pp = fkey.rfind('.');
            if(pp != std::string::npos && pp < fkey.size()-1) //need to be able to add 1
            {
               m_gv->statusTextText(n, fkey.substr(pp+1).c_str());
            }
            else
            {
               m_gv->statusTextText(n, "filt unk");
            }
         }
         else
         {
            fwstate = "f/w " + fwstate;
            m_gv->statusTextText(n, fwstate.c_str());
            ++n;
         }
      }
      if(n > m_gv->statusTextNo()-1) return 0;   
   }
   
   return 0;
}

void cameraStatus::keyPressEvent( QKeyEvent * ke)
{
   char key = ke->text()[0].toLatin1();
   
   if(key == 'C')
   {
      if(m_enabled) disableOverlay();
      else enableOverlay();
   }
}

bool cameraStatus::overlayEnabled()
{
   return m_enabled;
}

void cameraStatus::enableOverlay()
{
   if(m_enableable == false) return;
   
   std::cerr << "cameraStatus enabled\n";
   
   m_enabled = true;
}

void cameraStatus::disableOverlay()
{
   std::cerr << "cameraStatus disabled\n";

   for(size_t n=0; n<m_gv->statusTextNo();++n)
   {
      m_gv->statusTextText(n, "");
   }
   
   m_enabled = false;
}
