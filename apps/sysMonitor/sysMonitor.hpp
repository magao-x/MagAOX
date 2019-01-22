/** \file sysMonitor.hpp
  * \brief The MagAO-X sysMonitor app main program which provides functions to read and report system statistics
  * \author Chris Bohlman (cbohlmanaz@gmail.com)
  *
  * \ingroup sysMonitor_files
  *
  * History:
  * - 2018-08-10 created by CJB
  */
#ifndef sysMonitor_hpp
#define sysMonitor_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

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

/** MagAO-X application to read and report system statistics
  *
  */
class sysMonitor : public MagAOXApp<> 
{

protected:
   int m_warningCoreTemp = 0;   ///< User defined warning temperature for CPU cores
   int m_criticalCoreTemp = 0;   ///< User defined critical temperature for CPU cores
   int m_warningDiskTemp = 0;   ///< User defined warning temperature for drives
   int m_criticalDiskTemp = 0;   ///< User defined critical temperature for drives

   pcf::IndiProperty core_loads;   ///< Indi variable for repoting CPU core loads
   pcf::IndiProperty core_temps;   ///< Indi variable for repoting CPU core temperature(s)
   pcf::IndiProperty drive_temps;   ///< Indi variable for repoting drive temperature(s)
   pcf::IndiProperty root_usage;   ///< Indi variable for repoting drive usage of root path
   pcf::IndiProperty boot_usage;   ///< Indi variable for repoting drive usage of /boot path
   pcf::IndiProperty data_usage;   ///< Indi variable for repoting drive usage of /data path
   pcf::IndiProperty ram_usage_indi;   ///< Indi variable for repoting ram usage

   std::vector<float> coreTemps;   ///< List of current core temperature(s)
   std::vector<float> cpu_core_loads;   ///< List of current core load(s)
   std::vector<float> diskTemp;   ///< List of current disk temperature(s)
   float rootUsage = 0;   ///< Disk usage in root path as a value out of 100
   float dataUsage = 0;   ///< Disk usage in /data path as a value out of 100
   float bootUsage = 0;   ///< Disk usage in /boot path as a value out of 100
   float ramUsage = 0;   ///< RAM usage as a decimal value between 0 and 1

   /// Updates Indi property values of all system statistics
   /** This includes updating values for core loads, core temps, drive temps, / usage, /boot usage, /data usage, and RAM usage
     * Unsure if this method can fail in any way, as of now always returns 0
     *
     * \TODO: Check to see if any method called in here can fail
     * \returns 0 on completion
     */
   int updateVals();

public:

   /// Default c'tor.
   sysMonitor();

   /// D'tor, declared and defined for noexcept.
   ~sysMonitor() noexcept
   {
   }

   /// Setup the user-defined warning and critical values for core and drive temperatures
   virtual void setupConfig();

   /// Load the warning and critical temperature values for core and drive temperatures
   virtual void loadConfig();

   /// Registers all new Indi properties for each of the reported values to publish
   virtual int appStartup();

   /// Implementation of reading and logging each of the measured statistics
   virtual int appLogic();

   /// Do any needed shutdown tasks; currently nothing in this app
   virtual int appShutdown();


   /// Finds all CPU core temperatures
   /** Makes system call and then parses result to add temperatures to vector of values
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on successful completion otherwise
     */
   int findCPUTemperatures(
      std::vector<float>&  /**< [out] the vector of measured CPU core temperatures*/
   );


   /// Parses string from system call to find CPU temperatures
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseCPUTemperatures(
      std::string,  /**< [in] the string to be parsed*/
      float&   /**< [out] the return value from the string*/
   );


   /// Checks if any core temperatures are warning or critical levels
   /** Warning and critical temperatures are either user-defined or generated based on initial core temperature values
     *
     * \returns 1 if a temperature value is at the warning level
     * \returns 2 if a temperature value is at critical level
     * \returns 0 otherwise (all temperatures are considered normal)
     */
   int criticalCoreTemperature(
      std::vector<float>&   /**< [in] the vector of temperature values to be checked*/
   );


   /// Finds all CPU core usage loads
   /** Makes system call and then parses result to add usage loads to vector of values
     *
     * \returns -1 on error with system command or output file reading
     * \returns 0 on completion
     */
   int findCPULoads(
      std::vector<float>&   /**< [out] the vector of measured CPU usages*/
   );


   /// Parses string from system call to find CPU usage loads
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseCPULoads(
      std::string,   /**< [in] the string to be parsed*/
      float&   /**< [out] the return value from the string*/
   );


   /// Finds all drive temperatures
   /** Makes system call and then parses result to add temperatures to vector of values
     * For hard drive temp utility:
     * `wget http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/h/hddtemp-0.3-0.31.beta15.el7.x86_64.rpm`
     * `su`
     * `rpm -Uvh hddtemp-0.3-0.31.beta15.el7.x86_64.rpm`
     * Check install with rpm -q -a | grep -i hddtemp
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on successful completion otherwise
     */
   int findDiskTemperature(
      std::vector<float>&   /**< [out] the vector of measured drive temperatures*/
   );


   /// Parses string from system call to find drive temperatures
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseDiskTemperature(
      std::string,  /**< [in] the string to be parsed*/
      float&   /**< [out] the return value from the string*/
   );


   /// Checks if any drive temperatures are warning or critical levels
   /** Warning and critical temperatures are either user-defined or generated based on initial drive temperature values
     *
     * \returns 1 if a temperature value is at the warning level
     * \returns 2 if a temperature value is at critical level
     * \returns 0 otherwise (all temperatures are considered normal)
     */
   int criticalDiskTemperature(
      std::vector<float>&   /**< [in] the vector of temperature values to be checked*/
   );


   /// Finds usages of space for following directory paths: /; /data; /boot
   /** These usage values are stored as integer values between 0 and 100 (e.g. value of 39 means directory is 39% full)
     * If directory is not found, space usage value will remain 0
     * TODO: What about multiple drives? What does this do?
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 if at least one of the return values is found
     */
   int findDiskUsage(
      float&,   /**< [out] the return value for usage in root path*/
      float&,   /**< [out] the return value for usage in /data path*/
      float&    /**< [out] the return value for usage in /boot path*/
   );


   /// Parses string from system call to find drive usage space
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseDiskUsage(
      std::string,   /**< [in] the string to be parsed*/
      float&,   /**< [out] the return value for usage in root path*/
      float&,   /**< [out] the return value for usage in /data path*/
      float&    /**< [out] the return value for usage in /boot path*/
   );


   /// Finds current RAM usage
   /** This usage value is stored as a decimal value between 0 and 1 (e.g. value of 0.39 means RAM usage is 39%)
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on completion
     */
   int findRamUsage(
      float&    /**< [out] the return value for current RAM usage*/
   );


   /// Parses string from system call to find RAM usage
   /** When a valid string is read in, the value from that string is stored
    * 
    * \returns -1 on invalid string being read in
    * \returns 0 on completion and storing of value
    */
   int parseRamUsage(
      std::string,   /**< [in] the string to be parsed*/
      float&    /**< [out] the return value for current RAM usage*/
   );

};

inline sysMonitor::sysMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void sysMonitor::setupConfig()
{
   config.add("warningCoreTemp", "", "warningCoreTemp", argType::Required, "", "warningCoreTemp", false, "int", "The warning temperature for CPU cores.");
   config.add("criticalCoreTemp", "", "criticalCoreTemp", argType::Required, "", "criticalCoreTemp", false, "int", "The critical temperature for CPU cores.");
   config.add("warningDiskTemp", "", "warningDiskTemp", argType::Required, "", "warningDiskTemp", false, "int", "The warning temperature for the disk.");
   config.add("criticalDiskTemp", "", "criticalDiskTemp", argType::Required, "", "criticalDiskTemp", false, "int", "The critical temperature for disk.");
}

void sysMonitor::loadConfig()
{
   config(m_warningCoreTemp, "warningCoreTemp");
   config(m_criticalCoreTemp, "criticalCoreTemp");
   config(m_warningDiskTemp, "warningDiskTemp");
   config(m_criticalDiskTemp, "criticalDiskTemp");
}

int sysMonitor::appStartup()
{
   REG_INDI_NEWPROP_NOCB(core_loads, "core_loads", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(core_temps, "core_temps", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(drive_temps, "drive_temps", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(root_usage, "root_usage", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(boot_usage, "boot_usage", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(data_usage, "data_usage", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(ram_usage_indi, "ram_usage_indi", pcf::IndiProperty::Number);

   unsigned int i;
   std::string coreStr = "core";

   findCPULoads(cpu_core_loads);
   for (i = 0; i < cpu_core_loads.size(); i++) 
   {
      coreStr.append(std::to_string(i));
      core_loads.add (pcf::IndiElement(coreStr));
      core_loads[coreStr].set<double>(0.0);
      coreStr.pop_back();
   }

   findCPUTemperatures(coreTemps);
   for (i = 0; i < coreTemps.size(); i++) 
   {
      coreStr.append(std::to_string(i));
      core_temps.add (pcf::IndiElement(coreStr));
      core_temps[coreStr].set<double>(0.0);
      coreStr.pop_back();
   }

   std::string driveStr = "drive";
   findDiskTemperature(diskTemp);
   for (i = 0; i < diskTemp.size(); i++) 
   {
      driveStr.append(std::to_string(i));
      drive_temps.add (pcf::IndiElement(driveStr));
      drive_temps[driveStr].set<double>(0.0);
      driveStr.pop_back();
   }

   root_usage.add(pcf::IndiElement("root_usage"));
   root_usage["root_usage"].set<double>(0.0);

   boot_usage.add(pcf::IndiElement("boot_usage"));
   boot_usage["boot_usage"].set<double>(0.0);

   data_usage.add(pcf::IndiElement("data_usage"));
   data_usage["data_usage"].set<double>(0.0);

   ram_usage_indi.add(pcf::IndiElement("ram_usage"));
   ram_usage_indi["ram_usage"].set<double>(0.0);

   return 0;
}

int sysMonitor::appLogic()
{
   coreTemps.clear();
   int rvCPUTemp = findCPUTemperatures(coreTemps);
   if (rvCPUTemp >= 0) 
   {
      for (auto i: coreTemps)
      {
         std::cout << "Core temp: " << i << ' ';
      }   
      std::cout << std::endl;
      rvCPUTemp = criticalCoreTemperature(coreTemps);
   }
   
   cpu_core_loads.clear();
   int rvCPULoad = findCPULoads(cpu_core_loads);
   if (rvCPULoad >= 0) {
      for (auto i: cpu_core_loads)
      {
         std::cout << "CPU load: " << i << ' ';
      }
      std::cout << std::endl;
   }
   
   if (rvCPUTemp >= 0 && rvCPULoad >= 0)
   {
      if (rvCPUTemp == 1)
      {
         log<core_mon>({coreTemps, cpu_core_loads}, logPrio::LOG_WARNING);
      } 
      else if (rvCPUTemp == 2) 
      {
         log<core_mon>({coreTemps, cpu_core_loads}, logPrio::LOG_ALERT);
      } 
      else 
      {
         log<core_mon>({coreTemps, cpu_core_loads}, logPrio::LOG_INFO);
      }
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for CPU core temperatures and usages."});
   }

   diskTemp.clear();
   int rvDiskTemp = findDiskTemperature(diskTemp);
   if (rvDiskTemp >= 0)
   {
      for (auto i: diskTemp)
      {
         std::cout << "Disk temp: " << i << ' ';
      }
      std::cout << std::endl;
      rvDiskTemp = criticalDiskTemperature(diskTemp);
   }  

   int rvDiskUsage = findDiskUsage(rootUsage, dataUsage, bootUsage);
   if (rvDiskUsage >= 0)
   {
      std::cout << "/ usage: " << rootUsage << std::endl;
      std::cout << "/data usage: " << dataUsage << std::endl; 
      std::cout << "/boot usage: " << bootUsage << std::endl;
   }
   
   if (rvDiskTemp >= 0 && rvDiskUsage >= 0)
   {
      if (rvDiskTemp == 1) 
      {
         log<drive_mon>({diskTemp, rootUsage, dataUsage, bootUsage}, logPrio::LOG_WARNING);
      } 
      else if (rvDiskTemp == 2) 
      {
         log<drive_mon>({diskTemp, rootUsage, dataUsage, bootUsage}, logPrio::LOG_ALERT);
      } 
      else 
      {
         log<drive_mon>({diskTemp, rootUsage, dataUsage, bootUsage}, logPrio::LOG_INFO);
      }
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for drive temperatures and usages."});
   }
   

   int rvRamUsage = findRamUsage(ramUsage);
   if (rvRamUsage >= 0)
   {
      std::cout << "Ram usage: " << ramUsage << std::endl;
      log<ram_usage>({ramUsage}, logPrio::LOG_INFO);
   }
   else {
      log<software_error>({__FILE__, __LINE__,"Could not log values for RAM usage."});
   }

   updateVals();

   return 0;
}

int sysMonitor::appShutdown()
{
    return 0;
}

int sysMonitor::findCPUTemperatures(std::vector<float>& temps) 
{
   char command[35];
   std::string line;

   // For core temps
   strcpy( command, "sensors > /dev/shm/sensors_out" );
   int rv = system(command);
   if(rv == -1) //system call error
   {
      log<software_error>({__FILE__, __LINE__,"Could not complete CPU temperature `system` command."});
      return -1;
   }
   std::ifstream inFile;

   inFile.open("/dev/shm/sensors_out");
   if (!inFile) 
   {
      log<software_error>({__FILE__, __LINE__,"Could not open CPU temperature value file."});
      return -1;
   }

   rv = 1;
   while (getline (inFile,line)) 
   {
      float tempVal;
      if (parseCPUTemperatures(line, tempVal) == 0)
      {
         temps.push_back(tempVal);
         rv = 0;
      }
   }

   return rv;
}

int sysMonitor::parseCPUTemperatures(std::string line, float& temps) 
{
   if (line.length() <= 1)
   {
      return -1;
   }

   std::string str = line.substr(0, 5);
   if (str.compare("Core ") == 0) 
   {
      std::string temp_str = line.substr(17, 4);
      float temp;
      try
      {
         temp = std::stof (temp_str);
      }
      catch (const std::invalid_argument& e) 
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing CPU temperatures."});
         return -1;
      }

      temps = temp;

      if (m_warningCoreTemp == 0)
      {
         std::istringstream iss(line);
         std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
         try
         {
            tokens[5].pop_back();
            tokens[5].pop_back();
            tokens[5].pop_back();
            tokens[5].pop_back();
            tokens[5].erase(0,1);
            m_warningCoreTemp = std::stof(tokens[5]);
         }
         catch (const std::invalid_argument& e) 
         {
            log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing warning CPU temperatures."});
            return -1;
         }
      }
      if (m_criticalCoreTemp == 0) 
      {
         std::istringstream iss(line);
         std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
         try
         {
            tokens[8].pop_back();
            tokens[8].pop_back();
            tokens[8].pop_back();
            tokens[8].pop_back();
            tokens[8].erase(0,1);
            m_criticalCoreTemp = std::stof(tokens[8]);
         }
         catch (const std::invalid_argument& e) 
         {
            log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing critical CPU temperatures."});
            return -1;
         }
      }
      return 0;
   }
   else 
   {
      return -1;
   }

}

int sysMonitor::criticalCoreTemperature(std::vector<float>& v)
{
   int coreNum = 0, rv = 0;
   for(auto it: v)
   {
      float temp = it;
      if (temp >= m_warningCoreTemp && temp < m_criticalCoreTemp ) 
      {
         std::cout << "Warning temperature for Core " << coreNum << std::endl;
         if (rv < 2) 
         {
            rv = 1;
         }
      }
      else if (temp >= m_criticalCoreTemp) 
      {   
         std::cout << "Critical temperature for Core " << coreNum << std::endl;
         rv = 2;
      }
      ++coreNum;
   }
   return rv;
}

int sysMonitor::findCPULoads(std::vector<float>& loads) 
{
   char command[35];
   std::string line;
   // For cpu load
   strcpy( command, "mpstat -P ALL > /dev/shm/cpuload" );
   int rv = system(command);
   if(rv == -1) //system call error
   {
      log<software_error>({__FILE__, __LINE__,"Could not complete CPU core usage `system` command."});
      return -1;
   }

   std::ifstream inFile;
   inFile.open("/dev/shm/cpuload");
   if (!inFile) 
   {
      log<software_error>({__FILE__, __LINE__,"Could not open CPU core usage value file."});
      return -1;
   }

   // Want to start parsing at fifth line
   int iterator = 0;
   while (getline (inFile,line)) {
      iterator++;
      if (iterator == 4) {
         break;
      }
   }
   rv = 1;
   while (getline (inFile,line)) 
   {
      float loadVal;
      if (parseCPULoads(line, loadVal) == 0) 
      {
         loads.push_back(loadVal);
         rv = 0;
      }
   }

   return rv;
}

int sysMonitor::parseCPULoads(std::string line, float& loadVal)
{
   if (line.length() <= 1)
   {
      return -1;
   }
   std::istringstream iss(line);
   std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
   float cpu_load;
   try
   {
      cpu_load = 100.0 - std::stof(tokens[12]);
   }
   catch (const std::invalid_argument& e)
   {
      log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing CPU core usage."});
      return -1;
   }
   cpu_load /= 100;
   loadVal = cpu_load;
   return 0;
}

int sysMonitor::findDiskTemperature(std::vector<float>& hdd_temp) 
{
   char command[35];
   std::string line;

   strcpy( command, "hddtemp > /dev/shm/hddtemp" );
   int rv = system(command);
   if(rv == -1) //system call error
   {
      log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive temperatures."});
      return -1;
   }

   std::ifstream inFile;
   inFile.open("/dev/shm/hddtemp");
   if (!inFile) 
   {
      log<software_error>({__FILE__, __LINE__,"Could not open drive temperature value file."});
      return -1;
   }

   rv = 1;
   while (getline (inFile,line)) 
   {
      float tempVal;
      if (parseDiskTemperature(line, tempVal) == 0)
      {
         hdd_temp.push_back(tempVal);
         rv = 0;
      }
   }

   return rv;
}

int sysMonitor::parseDiskTemperature(std::string line, float& hdd_temp) 
{
   float tempValue;
   if (line.length() <= 1) 
   {
      return -1;
   }
   std::istringstream iss(line);
   std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
   for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it) 
   {
      std::string temp_s = *it;
      if (isdigit(temp_s.at(0)) && temp_s.substr(temp_s.length() - 1, 1) == "C") 
      {
         temp_s.pop_back();
         temp_s.pop_back();
         try
         {
            tempValue = std::stof (temp_s);
         }
         catch (const std::invalid_argument& e) 
         {
            log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive temperatures."});
            return -1;
         }
         hdd_temp = tempValue;
         if (m_warningDiskTemp == 0)
         {
            m_warningDiskTemp = tempValue + (.1*tempValue);
         }
         if (m_criticalDiskTemp == 0) 
         {
            m_criticalDiskTemp = tempValue + (.2*tempValue);
         }
         return 0;
      }
   }

   return -1;
}

int sysMonitor::criticalDiskTemperature(std::vector<float>& v)
{
   int rv = 0;
   for(auto it: v)
   {
      float temp = it;
      if (temp >= m_warningDiskTemp && temp < m_criticalDiskTemp )
      {
         std::cout << "Warning temperature for Disk" << std::endl;
         if (rv < 2)
         {
            rv = 1;
         }
      }  
      else if (temp >= m_criticalDiskTemp) 
      {   
         std::cout << "Critical temperature for Disk " << std::endl;
         rv = 2;
      }
   }
   return rv;
}

int sysMonitor::findDiskUsage(float &rootUsage, float &dataUsage, float &bootUsage) 
{
   char command[35];
   std::string line;
   // For disk usage
   strcpy( command, "df > /dev/shm/diskusage" );
   int rv = system(command);
   if(rv == -1) //system call error
   {
      log<software_error>({__FILE__, __LINE__,"Could not complete drive usage `system` command."});
      return -1;
   }

   std::ifstream inFile;
   inFile.open("/dev/shm/diskusage");
   if (!inFile) 
   {
      log<software_error>({__FILE__, __LINE__,"Could not open drive usage value file."});
      return -1;
   }

   rv = 1;
   while(getline(inFile,line)) 
   {
      int rvDiskUsage = parseDiskUsage(line, rootUsage, dataUsage, bootUsage);
      if (rvDiskUsage == 0) {
         rv = 0;
      }
   }

   return rv;
}

int sysMonitor::parseDiskUsage(std::string line, float& rootUsage, float& dataUsage, float& bootUsage) 
{
   if (line.length() <= 1)
   {
      return -1;
   }

   std::istringstream iss(line);
   std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
   if (tokens[5].compare("/") == 0)
   {
      tokens[4].pop_back();
      try
      {
         rootUsage = std::stof (tokens[4])/100;
         return 0;
      }
      catch (const std::invalid_argument& e) 
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive usage."});
         return -1;
      }
   } 
   else if (tokens[5].compare("/data") == 0)
   {
      tokens[4].pop_back();
      try
      {
         dataUsage = std::stof (tokens[4])/100;
         return 0;
      }
      catch (const std::invalid_argument& e) 
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive usage."});
         return -1;
      }
   } 
   else if (tokens[5].compare("/boot") == 0)
   {
      tokens[4].pop_back();
      try
      {
         bootUsage = std::stof (tokens[4])/100;
         return 0;
      }
      catch (const std::invalid_argument& e) 
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive usage."});
         return -1;
      }
   }
   return -1;
}

int sysMonitor::findRamUsage(float& ramUsage) 
{
   char command[35];
   std::string line;
   // For ram usage
   strcpy( command, "free -m > /dev/shm/ramusage" );
   int rv = system(command);
   if(rv == -1) //system call error
   {
      log<software_error>({__FILE__, __LINE__,"Could not complete RAM usage `system` command."});
      return -1;
   }

   std::ifstream inFile;
   inFile.open("/dev/shm/ramusage");
   if (!inFile) 
   {
      log<software_error>({__FILE__, __LINE__,"Could not open RAM usage value file."});
      return -1;
   }

   while (getline (inFile,line)) {
      if (parseRamUsage(line, ramUsage) == 0)
      {
        return 0;
      }
   }
   return -1;
}

int sysMonitor::parseRamUsage(std::string line, float& ramUsage) 
{
   if (line.length() <= 1)
   {
      return -1;
   }
   std::istringstream iss(line);
   std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
   try
   {
      if (tokens[0].compare("Mem:") != 0)
      {
        return -1;
      }
      ramUsage = std::stof(tokens[2])/std::stof(tokens[1]);
      if (ramUsage > 1 || ramUsage == 0)
      {
         ramUsage = -1;  
         return -1;
      }
      return 0;
   }
   catch (const std::invalid_argument& e) 
   {
      log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing RAM usage."});
      return -1;
   }
}

int sysMonitor::updateVals()
{
   unsigned int i;
   std::string coreStr = "core";
   std::string driveStr = "drive";

   for (i = 0; i < cpu_core_loads.size(); i++) 
   {
      coreStr.append(std::to_string(i));
      // core_loads[coreStr] = cpu_core_loads[i];
      MagAOXApp::updateIfChanged(core_loads, coreStr, cpu_core_loads[i]);
      coreStr.pop_back();
   }
   // core_loads.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty(core_loads);

   for (i = 0; i < coreTemps.size(); i++) 
   {
      coreStr.append(std::to_string(i));
      // core_temps[coreStr] = coreTemps[i];
      MagAOXApp::updateIfChanged(core_temps, coreStr, coreTemps[i]);
      coreStr.pop_back();
   }
   // core_temps.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty (core_temps);

   for (i = 0; i < diskTemp.size(); i++) 
   {
      driveStr.append(std::to_string(i));
      // drive_temps[driveStr] = diskTemp[i];
      MagAOXApp::updateIfChanged(drive_temps, driveStr, diskTemp[i]);
      driveStr.pop_back();
   }
   // drive_temps.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty (drive_temps);

   // root_usage["root_usage"] = rootUsage;
   // root_usage.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty (root_usage);
   MagAOXApp::updateIfChanged(
      root_usage,
      "root_usage",
      rootUsage
   );

   // boot_usage["boot_usage"] = bootUsage;
   // boot_usage.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty (boot_usage);
   MagAOXApp::updateIfChanged(
      boot_usage,
      "boot_usage",
      bootUsage
   );

   //data_usage["data_usage"] = dataUsage;
   // data_usage.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty (data_usage);
   MagAOXApp::updateIfChanged(
      data_usage,
      "data_usage",
      dataUsage
   );

   // ram_usage_indi["ram_usage"] = ramUsage;
   // ram_usage_indi.setState (pcf::IndiProperty::Ok);
   // if(m_indiDriver) m_indiDriver->sendSetProperty (ram_usage_indi);
   MagAOXApp::updateIfChanged(
      ram_usage_indi,
      "ram_usage",
      ramUsage
   );

   return 0;
}


} //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
