
#ifndef sysMonitor_hpp
#define sysMonitor_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"

#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>



namespace MagAOX
{
namespace app
{

/** MagAO-X application to do math on some numbers
  *
  */
class sysMonitor : public MagAOXApp
{

protected:
   

public:

   /// Default c'tor.
   sysMonitor();

   ~sysMonitor() noexcept
   {
   }


   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// Load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Checks if the device was found during loadConfig.
   virtual int appStartup();

   /// Implementation of the FSM for the maths.
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();

   int criticalTemperature(std::vector<int>);

};

sysMonitor::sysMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void sysMonitor::setupConfig()
{

}

void sysMonitor::loadConfig()
{

}

int sysMonitor::appStartup()
{

   return 0;
}

int sysMonitor::appLogic()
{
   char command[35];

   strcpy( command, "sensors > /dev/shm/sensors_out" );
   int rv = system(command);
   if(rv == -1) //system call error
   {
      //handle error
      std::cerr << "There's been an error with the system command" << std::endl;
      return 1;
   }
   std::ifstream inFile;
    
   inFile.open("/dev/shm/sensors_out");
   if (!inFile) 
   {
      std::cerr << "Unable to open file" << std::endl;
      return 1;
   }
   std::string line;
   std::vector<int> temps;
   while (getline (inFile,line)) {
      std::string str = line.substr(0, 5);
      if (str.compare("Core ") == 0) 
      {
         std::string temp_str = line.substr(17, 4);
         std::string::size_type sz;
         double temp = std::stod (temp_str,&sz);
         temps.push_back(temp);
         std::cout << temp << std::endl;
      }
   }
   criticalTemperature(temps);
   /*
   for (std::vector<int>::const_iterator i = temps.begin(); i != temps.end(); ++i)
      std::cout << *i << ' ';
      */

   return 0;
}

int sysMonitor::appShutdown()
{

   return 0;
}

int sysMonitor::criticalTemperature(std::vector<int> temps)
{
   int warningTempValue = 80, criticalTempValue = 90, iterator = 1;
   for (std::vector<int>::const_iterator i = temps.begin(); i != temps.end(); ++i)
   {
      int temp = *i;
      if (temp >=warningTempValue && temp < criticalTempValue ) {
         std::cout << "Warning temperature for Core " << iterator << std::endl;
      }
      else if (temp >= criticalTempValue) 
      {   
         std::cout << "Critical temperature for Core " << iterator << std::endl;
      }
      ++iterato;r
   }
}



} //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
