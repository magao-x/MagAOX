/** \file sysMonitor.hpp
  * \brief The MagAO-X sysMonitor app main program which provides functions to read and report system statistics
  * \author Chris Bohlman (cbohlman@pm.me)
  *
  * To view logdump files: logdump -f sysMonitor
  *
  * To view sysMonitor with cursesIndi: 
  * 1. /opt/MagAOX/bin/xindiserver -n xindiserverMaths
  * 2. /opt/MagAOX/bin/sysMonitor -n sysMonitor
  * 3. /opt/MagAOX/bin/cursesINDI 
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
#include <sys/wait.h>


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

   pcf::IndiProperty m_indiP_core_loads;   ///< Indi variable for reporting CPU core loads
   pcf::IndiProperty m_indiP_core_temps;   ///< Indi variable for reporting CPU core temperature(s)
   pcf::IndiProperty m_indiP_drive_temps;   ///< Indi variable for reporting drive temperature(s)
   pcf::IndiProperty m_indiP_usage;   ///< Indi variable for reporting drive usage of all paths

   std::vector<float> m_coreTemps;   ///< List of current core temperature(s)
   std::vector<float> coreLoads;   ///< List of current core load(s)
   
   std::vector<std::string> m_diskNames; ///< vector of names of the hard disks
   std::vector<float> m_diskTemps;        ///< vector of current disk temperature(s)
   
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
   /** Makes 'hddtemp' system call and then parses result to add temperatures to vector of values
     * For hard drive temp utility:
     * `wget http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/h/hddtemp-0.3-0.31.beta15.el7.x86_64.rpm`
     * `su`
     * `rpm -Uvh hddtemp-0.3-0.31.beta15.el7.x86_64.rpm`
     * Check install with rpm -q -a | grep -i hddtemp
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on successful completion otherwise
     */
   int findDiskTemperature( std::vector<std::string> & hdd_names, ///< [out] the names of the drives reported by hddtemp 
                            std::vector<float> & hdd_temps        ///< [out] the vector of measured drive temperatures
                          );


   /// Parses string from system call to find drive temperatures
   /** When a valid string is read in, the drive name and value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseDiskTemperature( std::string & driveName, ///< [out] the name of the drive
                             float & temp,            ///< [out] the return value from the string
                             const std::string & line ///< [in] the string to be parsed
                           );


   /// Checks if any drive temperatures are warning or critical levels
   /** Warning and critical temperatures are either user-defined or generated based on initial drive temperature values
     *
     * \returns 1 if a temperature value is at the warning level
     * \returns 2 if a temperature value is at critical level
     * \returns 0 otherwise (all temperatures are considered normal)
     */
   int criticalDiskTemperature( std::vector<float>&   /**< [in] the vector of temperature values to be checked*/ );


   /// Finds usages of space for following directory paths: /; /data; /boot
   /** These usage values are stored as integer values between 0 and 100 (e.g. value of 39 means directory is 39% full)
     * If directory is not found, space usage value will remain 0
     * \TODO: What about multiple drives? What does this do?
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 if at least one of the return values is found
     */
   int findDiskUsage( float&,   /**< [out] the return value for usage in root path*/
                      float&,   /**< [out] the return value for usage in /data path*/
                      float&    /**< [out] the return value for usage in /boot path*/
                    );


   /// Parses string from system call to find drive usage space
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseDiskUsage( std::string,   /**< [in] the string to be parsed*/
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
   int findRamUsage( float&    /**< [out] the return value for current RAM usage*/ );


   /// Parses string from system call to find RAM usage
   /** When a valid string is read in, the value from that string is stored
    * 
    * \returns -1 on invalid string being read in
    * \returns 0 on completion and storing of value
    */
   int parseRamUsage( std::string,   /**< [in] the string to be parsed*/
                      float&    /**< [out] the return value for current RAM usage*/
                    );

   /// Runs a command (with parameters) passed in using fork/exec
   /** New process is made with fork(), and child runs execvp with command provided
    * 
    * \returns output of command, contained in a vector of strings
    * If an error occurs during the process, an empty vector of strings is returned.
    */
   std::vector<std::string> runCommand( std::vector<std::string>    /**< [in] command to be run, with any subsequent parameters stored after*/);

};

inline sysMonitor::sysMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   //m_loopPause = 100000; //Set default to 1 milli-second due to mpstat averaging time of 1 sec.
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
   REG_INDI_NEWPROP_NOCB(m_indiP_core_loads, "core_loads", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(m_indiP_core_temps, "core_temps", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(m_indiP_drive_temps, "drive_temps", pcf::IndiProperty::Number);
   REG_INDI_NEWPROP_NOCB(m_indiP_usage, "resource_use", pcf::IndiProperty::Number);
   
   findCPUTemperatures(m_coreTemps);
   for (unsigned int i = 0; i < m_coreTemps.size(); i++) 
   {
      std::string coreStr = "core" + std::to_string(i);
      m_indiP_core_temps.add (pcf::IndiElement(coreStr));
      m_indiP_core_temps[coreStr].set<double>(0.0);
   }
   
   findCPULoads(coreLoads);
   for (unsigned int i = 0; i < coreLoads.size(); i++) 
   {
      std::string coreStr = "core" + std::to_string(i);
      m_indiP_core_loads.add (pcf::IndiElement(coreStr));
      m_indiP_core_loads[coreStr].set<double>(0.0);
   }
 
   findDiskTemperature(m_diskNames, m_diskTemps);
   for (unsigned int i = 0; i < m_diskTemps.size(); i++) 
   {
      m_indiP_drive_temps.add (pcf::IndiElement(m_diskNames[i]));
      m_indiP_drive_temps[m_diskNames[i]].set<double>(m_diskTemps[i]);
   }

   m_indiP_usage.add(pcf::IndiElement("root_usage"));
   m_indiP_usage.add(pcf::IndiElement("boot_usage"));
   m_indiP_usage.add(pcf::IndiElement("data_usage"));
   m_indiP_usage.add(pcf::IndiElement("ram_usage"));
   
   m_indiP_usage["root_usage"].set<double>(0.0);
   m_indiP_usage["boot_usage"].set<double>(0.0);
   m_indiP_usage["data_usage"].set<double>(0.0);
   m_indiP_usage["ram_usage"].set<double>(0.0);

   return 0;
}

int sysMonitor::appLogic()
{
   m_coreTemps.clear();
   int rvCPUTemp = findCPUTemperatures(m_coreTemps);
   if (rvCPUTemp >= 0) 
   {
      rvCPUTemp = criticalCoreTemperature(m_coreTemps);
   }
   
   if (rvCPUTemp >= 0)
   {
      if (rvCPUTemp == 1)
      {
         log<telem_coretemps>(m_coreTemps, logPrio::LOG_WARNING);
      } 
      else if (rvCPUTemp == 2) 
      {
         log<telem_coretemps>(m_coreTemps, logPrio::LOG_ALERT);
      } 
      else 
      {
         log<telem_coretemps>(m_coreTemps, logPrio::LOG_INFO);
      }
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for CPU core temps."});
   }
   
   coreLoads.clear();
   int rvCPULoad = findCPULoads(coreLoads);
   
   if(rvCPULoad >= 0)
   {
      log<telem_coreloads>(coreLoads, logPrio::LOG_INFO);
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for CPU core loads."});
   }

   m_diskNames.clear();
   m_diskTemps.clear();
   int rvDiskTemp = findDiskTemperature(m_diskNames, m_diskTemps);
   
   if (rvDiskTemp >= 0)
   {
      rvDiskTemp = criticalDiskTemperature(m_diskTemps);
   }  

   if (rvDiskTemp >= 0)
   {
      if (rvDiskTemp == 1) 
      {
         log<telem_drivetemps>({m_diskNames, m_diskTemps}, logPrio::LOG_WARNING);
      } 
      else if (rvDiskTemp == 2) 
      {
         log<telem_drivetemps>({m_diskNames, m_diskTemps}, logPrio::LOG_ALERT);
      } 
      else 
      {
         log<telem_drivetemps>({m_diskNames, m_diskTemps}, logPrio::LOG_INFO);
      }
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for drive temps."});
   }

   int rvDiskUsage = findDiskUsage(rootUsage, dataUsage, bootUsage);
   int rvRamUsage = findRamUsage(ramUsage);

   
   if (rvDiskUsage >= 0 && rvRamUsage >= 0)
   {
      log<telem_usage>({ramUsage, bootUsage, rootUsage, dataUsage}, logPrio::LOG_INFO);
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for usage."});
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
   std::vector<std::string> commandList{"sensors"};
   
   std::vector<std::string> commandOutput = runCommand(commandList);
   
   int rv = -1;
   for (auto line: commandOutput) 
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
            tokens.at(5).pop_back();
            tokens.at(5).pop_back();
            tokens.at(5).pop_back();
            tokens.at(5).pop_back();
            tokens.at(5).erase(0,1);
            m_warningCoreTemp = std::stof(tokens.at(5));
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
            tokens.at(8).pop_back();
            tokens.at(8).pop_back();
            tokens.at(8).pop_back();
            tokens.at(8).pop_back();
            tokens.at(8).erase(0,1);
            m_criticalCoreTemp = std::stof(tokens.at(8));
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
   std::vector<std::string> commandList{"mpstat", "-P", "ALL", "1", "1"};
   std::vector<std::string> commandOutput = runCommand(commandList);
   int rv = -1;
   // If output lines are less than 5 (with one CPU, guarenteed output is 5)
   if (commandOutput.size() < 5) 
   {
      return rv;
   }
   //start iterating at fourth line
   for (auto line = commandOutput.begin()+4; line != commandOutput.end(); line++) 
   {
      float loadVal;
      if (parseCPULoads(*line, loadVal) == 0) 
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
      cpu_load = 100.0 - std::stof(tokens.at(12));
   }
   catch (const std::invalid_argument& e)
   {
      log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing CPU core usage."});
      return -1;
   }
   catch (const std::out_of_range& e) {
      return -1;
   }
   cpu_load /= 100;
   loadVal = cpu_load;
   return 0;
}

int sysMonitor::findDiskTemperature( std::vector<std::string> & hdd_names,
                                     std::vector<float>& hdd_temps
                                   ) 
{
   std::vector<std::string> commandList{"hddtemp"};
   std::vector<std::string> commandOutput = runCommand(commandList);
   
   int rv = -1;
   for (auto line: commandOutput) 
   {  
      std::string driveName;
      float tempVal;
      if (parseDiskTemperature(driveName, tempVal, line) == 0)
      {
         hdd_names.push_back(driveName);
         hdd_temps.push_back(tempVal);
         rv = 0;
      }
   }
   return rv;
}

int sysMonitor::parseDiskTemperature( std::string & driveName,
                                      float & hdd_temp,
                                      const std::string & line
                                    ) 
{
   float tempValue;
   if (line.length() <= 6) 
   {
      return -1;
   }
   
   size_t sp = line.find(':',0);
   driveName = line.substr(5, sp-5);

   std::istringstream iss(line);
   std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};

   for(auto temp_s: tokens) 
   {
      try 
      {
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
      catch (const std::out_of_range& e) 
      {
         return -1;
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
   std::vector<std::string> commandList{"df"};
   std::vector<std::string> commandOutput = runCommand(commandList);
   int rv = -1;
   for (auto line: commandOutput) 
   {  
      int rvDiskUsage = parseDiskUsage(line, rootUsage, dataUsage, bootUsage);
      if (rvDiskUsage == 0) 
      {
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

   try {
      if (tokens.at(5).compare("/") == 0)
      {
         tokens.at(4).pop_back();
         try
         {
            rootUsage = std::stof (tokens.at(4))/100;
            return 0;
         }
         catch (const std::invalid_argument& e) 
         {
            log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive usage."});
            return -1;
         }
      } 
      else if (tokens.at(5).compare("/data") == 0)
      {
         tokens.at(4).pop_back();
         try
         {
            dataUsage = std::stof (tokens.at(4))/100;
            return 0;
         }
         catch (const std::invalid_argument& e) 
         {
            log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive usage."});
            return -1;
         }
      } 
      else if (tokens.at(5).compare("/boot") == 0)
      {
         tokens.at(4).pop_back();
         try
         {
            bootUsage = std::stof (tokens.at(4))/100;
            return 0;
         }
         catch (const std::invalid_argument& e) 
         {
            log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing drive usage."});
            return -1;
         }
      }
   }
   catch (const std::out_of_range& e) {
      return -1;
   }
   return -1;
}

int sysMonitor::findRamUsage(float& ramUsage) 
{
   std::vector<std::string> commandList{"free", "-m"};
   std::vector<std::string> commandOutput = runCommand(commandList);
   for (auto line: commandOutput) 
   {  
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
      if (tokens.at(0).compare("Mem:") != 0)
      {
        return -1;
      }
      ramUsage = std::stof(tokens.at(2))/std::stof(tokens.at(1));
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
   catch (const std::out_of_range& e) {
      return -1;
   }
}

int sysMonitor::updateVals()
{
   updateIfChanged(m_indiP_core_loads, "core", coreLoads);

   updateIfChanged(m_indiP_core_temps, "core", m_coreTemps);

   updateIfChanged(m_indiP_drive_temps, m_diskNames, m_diskTemps);

   updateIfChanged(m_indiP_usage, "root_usage", rootUsage);
   updateIfChanged(m_indiP_usage, "boot_usage", bootUsage);
   updateIfChanged(m_indiP_usage, "data_usage", dataUsage);
   updateIfChanged(m_indiP_usage, "ram_usage", ramUsage);
   
   return 0;
}

std::vector<std::string> sysMonitor::runCommand( std::vector<std::string> commandList) 
{
   int link[2];
   pid_t pid;
   
   std::vector<std::string> commandOutput;

   if (pipe(link)==-1) 
   {
      perror("Pipe error");
      return commandOutput;
   }

   if ((pid = fork()) == -1) 
   {
      perror("Fork error");
      return commandOutput;
   }

   if(pid == 0) 
   {
      dup2 (link[1], STDOUT_FILENO);
      close(link[0]);
      close(link[1]);
      std::vector<const char *>charCommandList( commandList.size()+1, NULL);
      for(int index = 0; index < (int) commandList.size(); ++index)
      {
         charCommandList[index]=commandList[index].c_str();
      }
      execvp( charCommandList[0], const_cast<char**>(charCommandList.data()));
      perror("exec");
      return commandOutput;
   }
   else 
   {
      char commandOutput_c[4096];
         
      wait(NULL);
      close(link[1]);
      
      int rd;
      if ( (rd = read(link[0], commandOutput_c, sizeof(commandOutput_c))) < 0) 
      {
         perror("Read error");
         return commandOutput;
      }
      
      std::string line{};
      
      commandOutput_c[rd] = '\0';
      std::string commandOutputString(commandOutput_c);
      
      std::istringstream iss(commandOutputString);
      
      while (getline(iss, line)) 
      {
         commandOutput.push_back(line);
      }
      wait(NULL);
      return commandOutput;
   }
}


} //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
