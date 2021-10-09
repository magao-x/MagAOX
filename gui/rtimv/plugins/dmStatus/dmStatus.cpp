
#include "dmStatus.hpp"

dmStatus::dmStatus() : rtimvOverlayInterface()
{
}

dmStatus::~dmStatus()
{
}

int dmStatus::attachOverlay( rtimvOverlayAccess & roa,
                                 mx::app::appConfigurator & config
                               )
{   
   m_roa = roa;
   m_qgs = roa.m_graphicsView->scene();
   
   
   config.configUnused(m_deviceName, mx::app::iniFile::makeKey("dm", "name"));
   
   if(m_deviceName == "")
   {
      m_enableable = false;
      disableOverlay();
      return 1; //Tell rtimv to unload me since not configured.
   }
   
   m_enableable = true;
   m_enabled = true;
   
   config.configUnused(m_rhDeviceName, mx::app::iniFile::makeKey("dm", "rhDevice"));
      
   if(m_enabled) enableOverlay();
   else disableOverlay();
   
   return 0;
}
      
int dmStatus::updateOverlay()
{
   if(!m_enabled) return 0;
   
   if(m_roa.m_dictionary == nullptr) return 0;
   
   if(m_roa.m_graphicsView == nullptr) return 0;
   
   size_t n = 0;
   char * str;
   char tstr[128];
   std::string sstr;
   
   if( m_roa.m_dictionary->count(m_rhDeviceName + ".humidity.current") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_rhDeviceName + ".humidity.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "RH: %0.1f%%", strtod(str,0));
      m_roa.m_graphicsView->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
   
   return 0;
}

void dmStatus::keyPressEvent( QKeyEvent * ke)
{
   static_cast<void>(ke);
}

bool dmStatus::overlayEnabled()
{
   return m_enabled;
}

void dmStatus::enableOverlay()
{
   if(m_enableable == false) return;
   
   m_enabled = true;
}

void dmStatus::disableOverlay()
{
   for(size_t n=0; n<m_roa.m_graphicsView->statusTextNo();++n)
   {
      m_roa.m_graphicsView->statusTextText(n, "");
   }
   
   m_enabled = false;
}
