/** \file stdCamera.cpp
  * \brief Standard camera interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#include "stdCamera.hpp"

namespace MagAOX
{
namespace app
{
namespace dev 
{

int loadCameraConfig( cameraConfigMap & ccmap, //  [out] the map in which to place the configurations found in config
                      mx::app::appConfigurator & config // [in] the application configuration structure
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
      
      unsigned centerX = 0;
      config.configUnused(centerX, mx::app::iniFile::makeKey(sections[i], "centerX" ));
      
      unsigned centerY = 0;
      config.configUnused(centerY, mx::app::iniFile::makeKey(sections[i], "centerY" ));
      
      unsigned sizeX = 0;
      config.configUnused(sizeX, mx::app::iniFile::makeKey(sections[i], "sizeX" ));
      
      unsigned sizeY = 0;
      config.configUnused(sizeY, mx::app::iniFile::makeKey(sections[i], "sizeY" ));
      
      unsigned binningX = 1;
      config.configUnused(binningX, mx::app::iniFile::makeKey(sections[i], "binningX" ));
      
      unsigned binningY = 1;
      config.configUnused(binningY, mx::app::iniFile::makeKey(sections[i], "binningY" ));
      
      unsigned dbinningX = 1;
      config.configUnused(dbinningX, mx::app::iniFile::makeKey(sections[i], "digital_binningX" ));
      
      unsigned dbinningY = 1;
      config.configUnused(dbinningY, mx::app::iniFile::makeKey(sections[i], "digital_binningY" ));

      float maxFPS = 0;
      config.configUnused(maxFPS, mx::app::iniFile::makeKey(sections[i], "maxFPS" ));
      
      ccmap[sections[i]] = cameraConfig({configFile, serialCommand, centerX, centerY, sizeX, sizeY, 
                                                    binningX, binningY, dbinningX, dbinningY, maxFPS});
   }
   
   return 0;
}



} //namespace dev
} //namespace app
} //namespace MagAOX

