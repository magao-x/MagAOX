/** \file ioDevice.hpp
 * \author Jared R. Males
 * \brief Configuration and control of input and output to a device
 * 
 * \ingroup tty_files
 *
 */

#ifndef app_tty_ioDevice_hpp
#define app_tty_ioDevice_hpp



namespace MagAOX
{
namespace app
{
namespace dev 
{
   
/// A USB device as a TTY device.
/**   
  * \ingroup tty
  */ 
struct ioDevice
{
   std::string m_readTimeout {1000};  ///< The read timeout [msec]
   std::string m_writeTimeout {1000}; ///< The write timeout [msec]

   ///Setup an application configurator for the device section
   /**
     * \returns 0 on success.
     * \returns -1 on error (nothing implemented yet)
     */ 
   int setupConfig( appConfigurator & config /**< [in] an application configuration to setup */);

   ///Load the device section from an application configurator
   /**
     *
     * \returns 0 on success
     * \returns -1 on error (nothing implemented yet)
     */
   int loadConfig( appConfigurator & config /**< [in] an application configuration from which to load values */);
   
};

int ioDevice::setupConfig( mx::app::appConfigurator & config )
{
   config.add("device.readTimeout", "", "device.readTimeout", argType::Required, "device", "readTimeout", false, "int", "timeout for reading from device");
   config.add("device.writeTimeout", "", "device.writeTimeout", argType::Required, "device", "writeTimeout", false, "int", "timeout for writing to device");

   return 0;
}

int ioDevice::loadConfig( mx::app::appConfigurator & config )
{
   config(m_readTimeout, "device.readTimeOut");
   config(m_writeTimeout, "device.writeTimeout");
}


} //namespace dev
} //namespace tty
} //namespace MagAOX

#endif //tty_ioDevice_hpp
