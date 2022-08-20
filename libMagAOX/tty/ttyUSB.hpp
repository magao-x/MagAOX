/** \file ttyUSB.hpp
 * \author Jared R. Males
 * \brief Find the details for USB serial devices
 * 
 * \ingroup tty_files
 *
 */

#ifndef tty_ttyUSB_hpp
#define tty_ttyUSB_hpp

#include <string>



namespace MagAOX
{
namespace tty
{

///Get the ttyUSB device name for a specific device
/**
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_NODEVNAMES if no device names found in sys
  * \returns TTY_E_UDEVNEWFAILED if initializing libudev failed.
  * \returns TTY_E_DEVNOTFOUND if no matching device found.
  * 
  * \ingroup tty 
  */
int ttyUSBDevName( std::string & devName,       ///< [out] the /dev/ttyUSBX device name.
                   const std::string & vendor,  ///< [in] the 4-digit vendor identifier.
                   const std::string & product, ///< [in] the 4-digit product identifier.
                   const std::string & serial   ///< [in] the serial number.  Can be "".
                 );

///Get the ttyUSB device name for a set of devices specified by their vendor and product ids.
/**
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_NODEVNAMES if no device names found in sys
  * \returns TTY_E_UDEVNEWFAILED if initializing libudev failed.
  * \returns TTY_E_DEVNOTFOUND if no matching device found.
  * 
  * \ingroup tty 
  */
int ttyUSBDevNames( std::vector<std::string> & devNames, ///< [out] the /dev/ttyUSBX device names for all matching devices.
                    const std::string & vendor,           ///< [in] the 4-digit vendor identifier.
                    const std::string & product           ///< [in] the 4-digit product identifier.
                  );

} //namespace tty
} //namespace MagAOX

#endif //tty_ttyUSB_hpp
