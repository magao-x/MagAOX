/** \file usbDevice.hpp
 * \author Jared R. Males
 * \brief Manage a USB TTY device in the MagAOXApp context
 * 
 * \ingroup tty_files
 *
 */

#ifndef tty_usbDevice_hpp
#define tty_usbDevice_hpp


#include <string>


#include <unistd.h>
//#include <fcntl.h>
//#include <poll.h>
#include <termios.h>

#include <mx/app/appConfigurator.hpp>

namespace MagAOX
{
namespace tty
{

/// A USB device as a TTY device.
/**   
  * \ingroup tty
  */ 
struct usbDevice
{
   std::string m_idVendor;  ///< The vendor id 4-digit code
   std::string m_idProduct; ///< The product id 4-digit code
   std::string m_serial;    ///< The serial number

   speed_t m_baudRate {0}; ///< The baud rate specification. 

   std::string m_deviceName; ///< The device path name, e.g. /dev/ttyUSB0

   int m_fileDescrip {0}; ///< The file descriptor

   ///Setup an application configurator for the USB section
   /**
     * \returns 0 on success.
     * \returns -1 on error (nothing implemented yet)
     */ 
   int setupConfig( mx::app::appConfigurator & config /**< [in] an application configuration to setup */);

   ///Load the USB section from an application configurator
   /**
     * If config does not contain a baud rate, m_baudRate is unchanged.  If m_baudRate is 0 at the end of this
     * method, an error is returned.  Set m_baudRate prior to calling to avoid this error.
     *
     * \returns 0 on success
     * \returns -1 on error (nothing implemented yet)
     */
   int loadConfig( mx::app::appConfigurator & config /**< [in] an application configuration from which to load values */);

   ///Get the device name from udev using the vendor, product, and serial number.
   int getDeviceName();

   ///Connect to the device.
   /** Closes the device file descriptor if open, then calls ttyOpenRaw.
     * 
     * \returns TTY_E_NOERROR on success.
     * \returns TTY_E_TCGETATTR on a error from tcgetattr.
     * \returns TTY_E_TCSETATTR on an error from tcsetattr.
     * \returns TTY_E_SETISPEED on a cfsetispeed error.
     * \returns TTY_E_SETOSPEED on a cfsetospeed error.
     */
   int connect();
};


} //namespace tty
} //namespace MagAOX

#endif //tty_usbDevice_hpp
