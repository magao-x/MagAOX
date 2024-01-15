
#include "warnings.hpp"

#define errPrint(expl) std::cerr << "warnings: " << __FILE__ << " " << __LINE__ << " " << expl << std::endl;

warnings::warnings() : rtimvOverlayInterface()
{
}

warnings::~warnings()
{
}

int warnings::attachOverlay( rtimvOverlayAccess & roa,
                             mx::app::appConfigurator & config
                           )
{   
    m_roa = roa;
   
    config.configUnused(m_deviceName, mx::app::iniFile::makeKey("rules", "device"));
   
    if(m_deviceName == "")
    {
        m_enableable = false;
        disableOverlay();
        return 1; //Tell rtimv to unload me since not configured.
    }
   
    config.configUnused(m_cautionKeys, mx::app::iniFile::makeKey("rules", "cautions"));
    config.configUnused(m_warningKeys, mx::app::iniFile::makeKey("rules", "warnings"));
    config.configUnused(m_alertKeys, mx::app::iniFile::makeKey("rules", "alerts"));

    if(m_cautionKeys.size() == 0 && m_warningKeys.size() == 0 && m_alertKeys.size() == 0) 
    {
        m_enableable = false;
        disableOverlay();
        return 1;
    }
    
    connect(this, SIGNAL(warningLevel(rtimv::warningLevel)), m_roa.m_mainWindowObject, SLOT(borderWarningLevel(rtimv::warningLevel)));

    if(m_roa.m_dictionary != nullptr)
    {
        for(size_t n = 0; n < m_cautionKeys.size(); ++n)
        {
            (*m_roa.m_dictionary)[m_deviceName + ".caution." + m_cautionKeys[n]].setBlob(nullptr, 0);
        }

        for(size_t n = 0; n < m_warningKeys.size(); ++n)
        {
            (*m_roa.m_dictionary)[m_deviceName + ".warning." + m_warningKeys[n]].setBlob(nullptr, 0);
        }

        for(size_t n = 0; n < m_alertKeys.size(); ++n)
        {
            (*m_roa.m_dictionary)[m_deviceName + ".alert." + m_alertKeys[n]].setBlob(nullptr, 0);
        }
        
    }

    m_enableable = true;
    m_enabled = true;
    enableOverlay();

    return 0;
}
      
int warnings::updateOverlay()
{
    if(!m_enabled) return 0;
   
    if(m_roa.m_dictionary == nullptr) return 0;

    if(m_roa.m_graphicsView == nullptr) return 0;
   
    bool caution = false;

    for(size_t n = 0; n < m_cautionKeys.size(); ++n)
    {
        if( ((*m_roa.m_dictionary)[m_deviceName + ".caution." + m_cautionKeys[n]].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) 
        {
            errPrint("bad string"); //Don't trust this as a string.
            continue;
        }

        if(std::string(m_blob) == "on") caution = true;
    }

    bool warn = false;

    for(size_t n = 0; n < m_warningKeys.size(); ++n)
    {
        if( ((*m_roa.m_dictionary)[m_deviceName + ".warning." + m_warningKeys[n]].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) 
        {
            errPrint("bad string"); //Don't trust this as a string.
            continue;
        }

        if(std::string(m_blob) == "on") warn = true;
    }

    bool alert = false;

    for(size_t n = 0; n < m_alertKeys.size(); ++n)
    {
        if( ((*m_roa.m_dictionary)[m_deviceName + ".alert." + m_alertKeys[n]].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) 
        {
            errPrint("bad string"); //Don't trust this as a string.
            continue;
        }

        if(std::string(m_blob) == "on") alert = true;
    }

    if(alert)
    {
        emit warningLevel(rtimv::warningLevel::alert);
    }
    else if(warn)
    {
        emit warningLevel(rtimv::warningLevel::warning);
    }
    else if(caution)
    {
        emit warningLevel(rtimv::warningLevel::caution);
    }
    else
    {
        emit warningLevel(rtimv::warningLevel::normal);
    }

    return 0;
}

void warnings::keyPressEvent( QKeyEvent * ke)
{
   static_cast<void>(ke);
}

bool warnings::overlayEnabled()
{
   return m_enabled;
}

void warnings::enableOverlay()
{
   if(m_enableable == false) return;
   
   m_enabled = true;
}

void warnings::disableOverlay()
{
   for(size_t n=0; n<m_roa.m_graphicsView->statusTextNo();++n)
   {
      m_roa.m_graphicsView->statusTextText(n, "");
   }
   
   m_enabled = false;
}

std::vector<std::string> warnings::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back("Warnings overlay: " + m_deviceName);
    /*if(m_deviceName != "")
    {
        vinfo.push_back("                   " + m_deviceName);
    }*/
    
    return vinfo;
}