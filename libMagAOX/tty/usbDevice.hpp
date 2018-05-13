/** \file usbDevice.hpp
 * \author Jared R. Males
 * \brief Manage a USB TTY device in the MagAOXApp context
 * 
 */

#ifndef tty_usbDevice_hpp
#define tty_usbDevice_hpp


#include "../../libMagAOX/tty/ttyIOUtils.hpp"
#include "../../libMagAOX/tty/ttyUSB.hpp"

namespace MagAOX 
{
namespace tty 
{
   
/// A USB device as a TTY device.
struct usbDevice
{
   std::string m_idVendor;  ///< The vendor id 4-digit code
   std::string m_idProduct; ///< The product id 4-digit code
   std::string m_serial;    ///< The serial number
   
   speed_t m_speed {0}; ///< The baud rate specification.
   
   std::string m_deviceName; ///< The device path name, e.g. /dev/ttyUSB0
   
   int m_fileDescrip {0}; ///< The file descriptor
   
   ///Setup an application configurator for the USB section
   int setupConfig( mx::appConfigurator & config /**< [in] an application configuration to setup */);
   
   ///Load the USB section from an application configurator
   /**
     * If config does not contain a baud rate, m_speed is unchanged.  If m_speed is 0 at the end of this
     * method, an error is returned.  Set m_speed prior to calling to avoid this error.
     */ 
   int loadConfig( mx::appConfigurator & config /**< [in] an application configuration from which to load values */);
   
   ///Get the device name from udev using the vendor, product, and serial number.
   int getDeviceName();
   
   ///Connect to the device.
   int connect();
};

int usbDevice::setupConfig( mx::appConfigurator & config )
{
   std::cerr << "Setting up\n";
   config.add("usb.idVendor", "", "idVendor", mx::argType::Required, "usb", "idVendor", false, "<string>", "USB vendor id, 4 digits");
   config.add("usb.idProduct", "", "idProduct", mx::argType::Required, "usb", "idProduct", false, "<string>", "USB product id, 4 digits");
   config.add("usb.serial", "", "serial", mx::argType::Required, "usb", "serial", false, "<string>", "USB serial number");
   config.add("usb.baud", "", "baud", mx::argType::Required, "usb", "baud", false, "int", "USB tty baud rate (i.e. 9600)");
   
   return 0;
}

int usbDevice::loadConfig( mx::appConfigurator & config )
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
         m_speed = B50;
         break;
      case 750:
         m_speed = B75;
         break;
      case 1100:
         m_speed = B110;
         break;
      case 1345:
         m_speed = B134;
         break;
      case 1500:
         m_speed = B150;
         break;
      case 2000:
         m_speed = B200;
         break;
      case 3000:
         m_speed = B300;
         break;
      case 6000:
         m_speed = B600;
         break;
      case 18000:
         m_speed = B1800;
         break;
      case 24000:
         m_speed = B2400;
         break;
      case 48000:
         m_speed = B4800;
         break;
      case 96000:
         m_speed = B9600;
         break;
      case 192000:
         m_speed = B19200;
         break;
      case 384000:
         m_speed = B38400;
         break;
      default:
         m_speed = 0;
         break;
   }

   if(m_speed == 0) return TTY_E_BADBAUDRATE;
   
   return getDeviceName();
}

int usbDevice::getDeviceName()
{
   return ttyUSBDevName( m_deviceName, m_idVendor, m_idProduct, m_serial );   
}

int usbDevice::connect()
{
   return ttyOpenRaw( m_fileDescrip, m_deviceName, m_speed );
}

} //namespace tty 
} //namespace MagAOX

#endif //tty_usbDevice_hpp
