
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

         int findCPUTemperatures(std::vector<int>);
         int parseCPUTemperatures(std::string, std::vector<int>); //done
         int criticalCoreTemperature(std::vector<int>);
         int findCPULoads();
         double parseCPULoads(std::string);
         int findDiskTemperature();
         int parseDiskTemperature(std::string);
         int criticalDiskTemperature(int);
         int findDiskUsage();
         int parseDiskUsage(std::string);
         int findRamUsage();
         double parseRamUsage(std::string);


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
         std::vector<int> coreTemps;
         int CPUTempMeasurement = findCPUTemperatures(coreTemps);
         criticalCoreTemperature(coreTemps);
         int CPULoad = findCPULoads();

         int diskTemp = findDiskTemperature();
         std::cout << diskTemp << std::endl;
         criticalDiskTemperature(diskTemp);
         int diskUsage = findDiskUsage();

         int ramUsage = findRamUsage();
         
         return 0;
      }

      int sysMonitor::appShutdown()
      {

         return 0;
      }

      int sysMonitor::findCPUTemperatures(std::vector<int> temps) {
         char command[35];
         std::string line;

         // For core temps
         //TODO: User defined warning level (use config)
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
         while (getline (inFile,line)) 
         {
            int rv = parseCPUTemperatures(line, temps);
         }
         return 0;
      }

      int sysMonitor::parseCPUTemperatures(std::string line, std::vector<int> temps) {
      	std::string str = line.substr(0, 5);
        if (str.compare("Core ") == 0) 
        {
        	std::string temp_str = line.substr(17, 4);
            double temp = std::stod (temp_str);
			temps.push_back(temp);
            //std::cout << temp << std::endl;
        }
        else 
        {
        	return 0;
        }
      }

      int sysMonitor::criticalCoreTemperature(std::vector<int> temps)
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

      int sysMonitor::findCPULoads() {
         char command[35];
         std::string line;
         // For cpu load
         strcpy( command, "mpstat -P ALL > /dev/shm/cpuload" );
         int rv = system(command);
         if(rv == -1) //system call error
         {
            //handle error
            std::cerr << "There's been an error with the system command" << std::endl;
            return 1;
         }

         std::ifstream inFile5;
         inFile5.open("/dev/shm/cpuload");
         if (!inFile5) 
         {
            std::cerr << "Unable to open file" << std::endl;
            return 1;
         }
         std::vector<double> cpu_core_loads; double cpu_load;
         int cores = 0;
         // Want to start at third line
         getline (inFile5,line);
         getline (inFile5,line);
         getline (inFile5,line);
         getline (inFile5,line);
         while (getline (inFile5,line)) 
         {
            cores++;
            cpu_load = parseCPULoads(line);
            cpu_core_loads.push_back(cpu_load);
         }
         return (int)cpu_load;
      }

      double sysMonitor::parseCPULoads(std::string line)
      {
      	std::istringstream iss4(line);
        std::vector<std::string> tokens4{std::istream_iterator<std::string>{iss4},std::istream_iterator<std::string>{}};
        double cpu_load = 100.0 - std::stod (tokens4[12]);
        cpu_load /= 100;
        //std::cout << "core load " << cpu_load << std::endl;
        return cpu_load;
      }

      int sysMonitor::findDiskTemperature() {
         char command[35];
         std::string line;

         // For hard drive temp
         // wget http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/h/hddtemp-0.3-0.31.beta15.el7.x86_64.rpm (binary package)
         // su
         // rpm -Uvh hddtemp-0.3-0.31.beta15.el7.x86_64.rpm
         // Check install with rpm -q -a | grep -i hddtemp
         //
         strcpy( command, "hddtemp > /dev/shm/hddtemp" );
         int rv = system(command);
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
         
         double hdd_temp = 0;
         getline (inFile2,line);
         hdd_temp = parseDiskTemperature(line);
         //std::cout << hdd_temp << std::endl;
         return (int) hdd_temp;
      }

      int sysMonitor::parseDiskTemperature(std::string line) {
      	double hdd_temp;
      	std::istringstream iss(line);
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
        for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it) 
        {
        	std::string temp_s = *it;
          	if (isdigit(temp_s.at(0)) && temp_s.substr(temp_s.length() - 1, 1) == "C") 
            {
               temp_s.pop_back();
               temp_s.pop_back();
               hdd_temp = std::stod (temp_s);
            }
         }
         return hdd_temp;
      }

      int sysMonitor::criticalDiskTemperature(int temp)
      {
         int warningTempValue = 80, criticalTempValue = 90;
         if (temp >=warningTempValue && temp < criticalTempValue ) {
            std::cout << "Warning temperature for Disk" << std::endl;
         }
         else if (temp >= criticalTempValue) 
         {   
            std::cout << "Critical temperature for Disk " << std::endl;
         }
      }

      int sysMonitor::findDiskUsage() {
         char command[35];
         std::string line;
         // For disk usage
         strcpy( command, "df > /dev/shm/diskusage" );
         int rv = system(command);
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
         double disk_usage = parseDiskUsage(line);
         
         return disk_usage;
      }

      int sysMonitor::parseDiskUsage(std::string line) 
      {
      	std::istringstream iss2(line);
        std::vector<std::string> tokens2{std::istream_iterator<std::string>{iss2},std::istream_iterator<std::string>{}};
        tokens2[4].pop_back();
        double disk_usage = std::stod (tokens2[4]);
        //std::cout << disk_usage << std::endl;
        return disk_usage;
      }

      int sysMonitor::findRamUsage() {
         char command[35];
         std::string line;
          // For ram usage
         strcpy( command, "free -m > /dev/shm/ramusage" );
         int rv = system(command);
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
         double ram_usage_percent = parseRamUsage(line);
         //std::cout << ram_usage_percent << std::endl;
         return ram_usage_percent;
      }

      double sysMonitor::parseRamUsage(std::string line) 
      {
      	std::istringstream iss3(line);
        std::vector<std::string> tokens3{std::istream_iterator<std::string>{iss3},std::istream_iterator<std::string>{}};
        double ram_usage = std::stod (tokens3[2]);
        double ram_total = std::stod (tokens3[1]);
        double ram_usage_percent = ram_usage/ram_total;
        return ram_usage_percent;
      }

   } //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
