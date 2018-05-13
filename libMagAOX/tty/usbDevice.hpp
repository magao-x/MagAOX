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
   
   speed_t m_speed; ///< The baud rate specification
   
   std::string m_deviceName; ///< The device path name, e.g. /dev/ttyUSB0
   
   int m_fileDescrip {0}; ///< The file descriptor
   
   ///Setup an application configurator for the USB section
   int setupConfig( mx::appConfigurator & config /**< [in] an application configuration to setup */);
   
   ///Load the USB section from an application configurator
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
   
   ///\todo add all baud rates
   int baud = 0;
   config(baud, "usb.baud");
   switch(baud)
   {
      case 9600:
         m_speed = B9600;
         break;
      default:
         m_speed = B9600;
         break;
   }
   
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
