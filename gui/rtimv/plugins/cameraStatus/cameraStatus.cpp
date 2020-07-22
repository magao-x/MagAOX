
#include "cameraStatus.hpp"

cameraStatus::cameraStatus() : rtimvOverlayInterface()
{
}

cameraStatus::~cameraStatus()
{
}

int cameraStatus::attachOverlay( rtimvGraphicsView* gv, 
                                 std::unordered_map<std::string, rtimvDictBlob> * dict,
                                 mx::app::appConfigurator & config
                               )
{
   std::cerr << "cameraStatus attached -- w config\n";
   
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
   
   std::cerr << "cameraStatus deviceName: " << m_deviceName << "\n";
   
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
      snprintf(tstr, sizeof(tstr), "%s sec", str);
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
   
   if( m_dict->count(m_deviceName + ".shutter.current") > 0)
   {
      str = (char *)(*m_dict)[m_deviceName + ".shutter.current"].m_blob;
      m_gv->statusTextText(n, str);
      ++n;
   }
   if(n > m_gv->statusTextNo()-1) return 0;   
   
   
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
