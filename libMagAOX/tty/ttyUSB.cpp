/** \file ttyUSB.cpp
 * \author Jared R. Males
 * \brief Find the details for USB serial devices
 * 
 * \ingroup tty_files
 *
 */


#include <iostream>

#include <libudev.h>

#include <string>
#include <cstring>

#include <boost/filesystem.hpp>
#include <mx/ioutils/fileUtils.hpp>

#include "ttyUSB.hpp"

#include "ttyErrors.hpp"


namespace MagAOX
{
namespace tty
{

int ttyUSBDevName( std::string & devName,       // [out] the /dev/ttyUSBX device name.
                   const std::string & vendor,  // [in] the 4-digit vendor identifier.
                   const std::string & product, // [in] the 4-digit product identifier.
                   const std::string & serial   // [in] the serial number.  Can be "".
                 )
{
   std::vector<std::string> devNames;

   devName = "";
   devNames = mx::ioutils::getFileNames("/sys/class/tty/", "ttyUSB", "", "");

   if(devNames.size() == 0) return TTY_E_NODEVNAMES;

   struct udev *udev;


   /* Create the udev object */
   udev = udev_new();
   if (!udev) return TTY_E_UDEVNEWFAILED;

   for(size_t i=0; i< devNames.size(); ++i)
   {
      struct udev_device *dev0;

      dev0 = udev_device_new_from_syspath(udev, devNames[i].c_str());

      if(!dev0) 
      {
         std::cerr << "udev_device_new_from_syspath failed: " << strerror(errno) << "\n";
         perror("");
         continue;
      }

      struct udev_device *dev;

      dev = udev_device_get_parent_with_subsystem_devtype( dev0, "usb", "usb_device");

      if (!dev) 
      {   
         std::cerr << "udev_device_get_parent_with_subsystem_devtype failed: " << strerror(errno) << "\n";
         perror("");
         udev_device_unref(dev0);
         continue;
      }

      const char * idVendor = udev_device_get_sysattr_value( dev, "idVendor" );
      
      if(idVendor == nullptr) 
      {
         udev_device_unref(dev0);
         continue;
      }

      if( strcmp( idVendor, vendor.c_str()) != 0) 
      {
         udev_device_unref(dev0);
         continue;
      }

      const char * idProduct = udev_device_get_sysattr_value( dev, "idProduct" );

      if(idProduct == nullptr) 
      {
         udev_device_unref(dev0);
         continue;
      }

      if( strcmp( idProduct, product.c_str()) != 0) 
      {
         udev_device_unref(dev0);
         continue;
      }

      const char * dserial = udev_device_get_sysattr_value( dev, "serial" );

      if(dserial == nullptr)
      {
         if( serial != "") 
         {
            udev_device_unref(dev0);
            continue;
         }
      }
      else if( strcmp( dserial, serial.c_str()) != 0 ) 
      {
         udev_device_unref(dev0);
         continue;
      }
    
      //If we make it through all comparisons we found it!
      boost::filesystem::path p(devNames[i]);
      devName = "/dev/" + p.filename().string();

      udev_device_unref(dev0);

      udev_unref(udev);

      return TTY_E_NOERROR;
   }

   devName = "";

   udev_unref(udev);

   return TTY_E_DEVNOTFOUND;
}

int ttyUSBDevNames( std::vector<std::string> & devNames, // [out] the /dev/ttyUSBX device names for all matching devices.
                    const std::string & vendor,           // [in] the 4-digit vendor identifier.
                    const std::string & product           // [in] the 4-digit product identifier.
                  )
{
   std::vector<std::string> pdevNames;

   devNames.clear();
   
   pdevNames = mx::ioutils::getFileNames("/sys/class/tty/", "ttyUSB", "", "");

   if(pdevNames.size() == 0) return TTY_E_NODEVNAMES;

   struct udev *udev;

   /* Create the udev object */
   udev = udev_new();
   if (!udev) return TTY_E_UDEVNEWFAILED;

   for(size_t i=0; i< pdevNames.size(); ++i)
   {
      struct udev_device *dev0;

      dev0 = udev_device_new_from_syspath(udev, pdevNames[i].c_str());

      if(!dev0) 
      {
         continue;
      }
      
      struct udev_device * dev;
      dev = udev_device_get_parent_with_subsystem_devtype( dev0, "usb", "usb_device");

      if (!dev) 
      {
         udev_device_unref(dev0);
         continue;
      }

      const char * idVendor = udev_device_get_sysattr_value( dev, "idVendor" );

      if(idVendor == nullptr) 
      {
         udev_device_unref(dev0);
         continue;
      }

      if( strcmp( idVendor, vendor.c_str()) != 0) 
      {
         udev_device_unref(dev0);
         continue;
      }

      const char * idProduct = udev_device_get_sysattr_value( dev, "idProduct" );

      if(idProduct == nullptr) 
      {
         udev_device_unref(dev0);
         continue;
      }

      if( strcmp( idProduct, product.c_str()) != 0) 
      {
         udev_device_unref(dev0);
         continue;
      }

      //If we make it through all comparisons we found it!
      boost::filesystem::path p(pdevNames[i]);
      devNames.push_back( "/dev/" + p.filename().string());

      udev_device_unref(dev0);
   }

   udev_unref(udev);

   if( devNames.size() > 0) return TTY_E_NOERROR;   
   else return TTY_E_DEVNOTFOUND;
   
}
} //namespace tty
} //namespace MagAOX

