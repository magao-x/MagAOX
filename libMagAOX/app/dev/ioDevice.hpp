/** \file ioDevice.hpp
 * \author Jared R. Males
 * \brief Configuration and control of an input and output device
 * 
 * \ingroup app_files
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
   
/// An input/output capable device.
/**   
  * \ingroup appdev
  */ 
struct ioDevice
{
   unsigned m_readTimeout {1000};  ///< The read timeout [msec]
   unsigned m_writeTimeout {1000}; ///< The write timeout [msec]

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
   
   /// Perform application startup steps specific to an ioDevice
   /**
     * This is currently an empty function which always returns 0.  Could be ignored,
     * but for future changes it is recommended to include a call to this in derivedT::appStartup().
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int appStartup();
   
   /// Perform application logic steps specific to an ioDevice during the main event loop
   /**
     * This is currently an empty function which always returns 0.  Could be ignored,
     * but for future changes it is recommended to include a call to this in derivedT::appLogic().
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int appLogic();
};

int ioDevice::setupConfig( mx::app::appConfigurator & config )
{
   config.add("device.readTimeout", "", "device.readTimeout", argType::Required, "device", "readTimeout", false, "int", "timeout for reading from device");
   config.add("device.writeTimeout", "", "device.writeTimeout", argType::Required, "device", "writeTimeout", false, "int", "timeout for writing to device");

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

#endif //tty_ioDevice_hpp
