
#ifndef sysMonitor_hpp
#define sysMonitor_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>



namespace MagAOX
{
   namespace app
   {

/** MagAO-X application to do math on some numbers
  *
  */
      class sysMonitor : public MagAOXApp<> {

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

         // For core temps
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
         while (getline (inFile,line)) 
         {
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

         // For hard drive temp
         // wget http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/h/hddtemp-0.3-0.31.beta15.el7.x86_64.rpm (binary package)
         // su
         // rpm -Uvh hddtemp-0.3-0.31.beta15.el7.x86_64.rpm
         // Check install with rpm -q -a | grep -i hddtemp
         //
         strcpy( command, "hddtemp > /dev/shm/hddtemp" );
         rv = system(command);
         if(rv == -1) //system call error
         {
            //handle error
            std::cerr << "There's been an error with the system command" << std::endl;
            return 1;
         }

         std::ifstream inFile2;
         inFile2.open("/dev/shm/hddtemp");
         if (!inFile2) 
         {
            std::cerr << "Unable to open file" << std::endl;
            return 1;
         }
         
         getline (inFile2,line);
         std::istringstream iss(line);
         std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
         for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it) 
         {
            std::string temp_s = *it;
            if (isdigit(temp_s.at(0)) && temp_s.substr(temp_s.length() - 1, 1) == "C") 
            {
               temp_s.pop_back();
               temp_s.pop_back();
               std::string::size_type sz2;
               double hdd_temp = std::stod (temp_s,&sz2);
               std::cout << hdd_temp << std::endl;
            }
         }

         // For disk usage
         strcpy( command, "df > /dev/shm/diskusage" );
         rv = system(command);
         if(rv == -1) //system call error
         {
            //handle error
            std::cerr << "There's been an error with the system command" << std::endl;
            return 1;
         }

         std::ifstream inFile3;
         inFile3.open("/dev/shm/diskusage");
         if (!inFile3) 
         {
            std::cerr << "Unable to open file" << std::endl;
            return 1;
         }
         
         // Want second line
         getline (inFile3,line);
         getline (inFile3,line);
         std::istringstream iss2(line);
         std::vector<std::string> tokens2{std::istream_iterator<std::string>{iss2},std::istream_iterator<std::string>{}};
         tokens2[4].pop_back();
         std::string::size_type sz3;
         double disk_usage = std::stod (tokens2[4],&sz3);
         std::cout << disk_usage << std::endl;

         
         // For ram usage
         strcpy( command, "free -m > /dev/shm/ramusage" );
         rv = system(command);
         if(rv == -1) //system call error
         {
            //handle error
            std::cerr << "There's been an error with the system command" << std::endl;
            return 1;
         }

         std::ifstream inFile4;
         inFile4.open("/dev/shm/ramusage");
         if (!inFile4) 
         {
            std::cerr << "Unable to open file" << std::endl;
            return 1;
         }
         
         // Want second line
         getline (inFile4,line);
         getline (inFile4,line);
         std::istringstream iss3(line);
         std::vector<std::string> tokens3{std::istream_iterator<std::string>{iss3},std::istream_iterator<std::string>{}};
         std::string::size_type sz4;
         double ram_usage = std::stod (tokens3[2],&sz4);
         double ram_total = std::stod (tokens3[1],&sz4);
         double ram_usage_percent = ram_usage/ram_total;
         std::cout << ram_usage_percent << std::endl;


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
            ++iterator;
         }
      }



   } //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
