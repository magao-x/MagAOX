/** \file usbDevice.hpp
 * \author Jared R. Males
 * \brief Manage a USB TTY device in the MagAOXApp context
 * 
 * \ingroup tty_files
 *
 */

#ifndef tty_usbDevice_hpp
#define tty_usbDevice_hpp


#include "../../libMagAOX/tty/ttyIOUtils.hpp"
#include "../../libMagAOX/tty/ttyUSB.hpp"

using namespace mx::app;

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
   int setupConfig( appConfigurator & config /**< [in] an application configuration to setup */);

   ///Load the USB section from an application configurator
   /**
     * If config does not contain a baud rate, m_baudRate is unchanged.  If m_baudRate is 0 at the end of this
     * method, an error is returned.  Set m_baudRate prior to calling to avoid this error.
     *
     * \returns 0 on success
     * \returns -1 on error (nothing implemented yet)
     */
   int loadConfig( appConfigurator & config /**< [in] an application configuration from which to load values */);

   ///Get the device name from udev using the vendor, product, and serial number.
   int getDeviceName();

   ///Connect to the device.
   int connect();
};

int usbDevice::setupConfig( mx::app::appConfigurator & config )
{
   config.add("usb.idVendor", "", "usb.idVendor", argType::Required, "usb", "idVendor", false, "string", "USB vendor id, 4 digits");
   config.add("usb.idProduct", "", "usb.idProduct", argType::Required, "usb", "idProduct", false, "string", "USB product id, 4 digits");
   config.add("usb.serial", "", "usb.serial", argType::Required, "usb", "serial", false, "string", "USB serial number");
   config.add("usb.baud", "", "usb.baud", argType::Required, "usb", "baud", false, "real", "USB tty baud rate (i.e. 9600)");

   return 0;
}

int usbDevice::loadConfig( mx::app::appConfigurator & config )
{
   config(m_idVendor, "usb.idVendor");
   config(m_idProduct, "usb.idProduct");
   config(m_serial, "usb.serial");

   //We read the config as a float to allow 134.5
   //Then multiply by 10 for the switch statement.
   float baud = 0;
   config(baud, "usb.baud");
   
   switch((int)(baud*10))
   {
      case 0:
         break; //Don't change default.
      case 500:
         m_baudRate = B50;
         break;
      case 750:
         m_baudRate = B75;
         break;
      case 1100:
         m_baudRate = B110;
         break;
      case 1345:
         m_baudRate = B134;
         break;
      case 1500:
         m_baudRate = B150;
         break;
      case 2000:
         m_baudRate = B200;
         break;
      case 3000:
         m_baudRate = B300;
         break;
      case 6000:
         m_baudRate = B600;
         break;
      case 18000:
         m_baudRate = B1800;
         break;
      case 24000:
         m_baudRate = B2400;
         break;
      case 48000:
         m_baudRate = B4800;
         break;
      case 96000:
         m_baudRate = B9600;
         break;
      case 192000:
         m_baudRate = B19200;
         break;
      case 576000:
         m_baudRate = B57600;
         break;
      case 384000:
         m_baudRate = B38400;
         break;
      default:
         break;
   }

   if(m_baudRate == 0) return TTY_E_BADBAUDRATE;

   return getDeviceName();
}

int usbDevice::getDeviceName()
{
   return ttyUSBDevName( m_deviceName, m_idVendor, m_idProduct, m_serial );
}

int usbDevice::connect()
{
   if(m_fileDescrip)
   {
      ::close(m_fileDescrip);
      m_fileDescrip = 0;
   }
   
   return ttyOpenRaw( m_fileDescrip, m_deviceName, m_baudRate );
}

} //namespace tty
} //namespace MagAOX

#endif //tty_usbDevice_hpp
