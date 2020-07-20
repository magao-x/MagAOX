
#include <iostream>
#include "BrainStem2/BrainStem-all.h"

/*==========================================================================================================================*/
/* Simple program to configure an Acroname USB 3 programmable hub for MagAO-X 
 * 
 * Does the following:
 *    1) Reads model, version, and serial number (you need serial number for configuration of the app)
 *    2) Turns off all USB ports
 *    3) Saves the all-ports-off state so that the unit will start up off.
 * 
 * Compile with:
 *   $ g++ -o configusbhub configusbhub.cpp -I../../../libs/BrainStem2/ ../../../libs/BrainStem2/libBrainStem2.a  -lpthread
 * 
 * Run with:
 *   $ sudo ./configusbhub
 */
/*==========================================================================================================================*/
int main()
{
 
   aUSBHub3p hub; // BrainStem library handle

   aErr err = aErrNone;

    
   err = hub.discoverAndConnect(USB);

   if(err != aErrNone)
   {
      std::cerr << "No hub found.\n";
      return -1;
   }

   SystemClass sys;
   sys.init(&hub,0);
   
   uint8_t model;
   sys.getModel(&model);
   
   uint32_t version;
   sys.getVersion(&version);
   std::cout << version << "\n";
   std::cout << "Version: " << aVersion_ParseMajor(version) << "." << aVersion_ParseMinor(version) << "." << aVersion_ParsePatch(version) << "\n";
   uint32_t serial;
   sys.getSerialNumber(&serial);
   
   std::cout << "Serial Number: " << serial << "\n";
   
   for(int n=0;n<8;++n)
   {
      hub.usb.setPortDisable(n);
   }
   
   sys.save();
   
   hub.disconnect();

}
