/** \file usbDevice.cpp
 * \author Jared R. Males
 * \brief Manage a USB TTY device in the MagAOXApp context
 * 
 * \ingroup tty_files
 *
 */

#include "usbDevice.hpp"


#include "ttyIOUtils.hpp"
#include "ttyUSB.hpp"
#include "ttyErrors.hpp"

using namespace mx::app;

namespace MagAOX
{
namespace tty
{


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

