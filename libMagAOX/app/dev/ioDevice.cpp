/** \file ioDevice.cpp
 * \author Jared R. Males
 * \brief Configuration and control of an input and output device
 * 
 * \ingroup app_files
 *
 */

#include "ioDevice.hpp"


namespace MagAOX
{
namespace app
{
namespace dev 
{
   

int ioDevice::setupConfig( mx::app::appConfigurator & config )
{
   config.add("device.readTimeout", "", "device.readTimeout", mx::app::argType::Required, "device", "readTimeout", false, "int", "timeout for reading from device");
   config.add("device.writeTimeout", "", "device.writeTimeout", mx::app::argType::Required, "device", "writeTimeout", false, "int", "timeout for writing to device");

   return 0;
}

int ioDevice::loadConfig( mx::app::appConfigurator & config )
{
   config(m_readTimeout, "device.readTimeout");
   config(m_writeTimeout, "device.writeTimeout");
   
   return 0;
}

int ioDevice::appStartup()
{
   return 0;
}

int ioDevice::appLogic()
{
   return 0;
}

} //namespace dev
} //namespace tty
} //namespace MagAOX

