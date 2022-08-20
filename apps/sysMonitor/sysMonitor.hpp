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
class sysMonitor : public MagAOXApp<>, public dev::telemeter<sysMonitor>
{

   friend class dev::telemeter<sysMonitor>;
   
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
   std::vector<float> m_coreLoads;   ///< List of current core load(s)
   
   std::vector<std::string> m_diskNameList; ///< vector of names of the hard disks to monitor
   std::vector<std::string> m_diskNames; ///< vector of names of the hard disks returned by hdd_temp
   std::vector<float> m_diskTemps;        ///< vector of current disk temperature(s)
   
   float m_rootUsage = 0;   ///< Disk usage in root path as a value out of 100
   float m_dataUsage = 0;   ///< Disk usage in /data path as a value out of 100
   float m_bootUsage = 0;   ///< Disk usage in /boot path as a value out of 100
   float m_ramUsage = 0;   ///< RAM usage as a decimal value between 0 and 1

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
   int findCPUTemperatures( std::vector<float>&  /**< [out] the vector of measured CPU core temperatures*/ );


   /// Parses string from system call to find CPU temperatures
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseCPUTemperatures( std::string,  /**< [in] the string to be parsed*/
                             float&   /**< [out] the return value from the string*/
                           );


   /// Checks if any core temperatures are warning or critical levels
   /** Warning and critical temperatures are either user-defined or generated based on initial core temperature values
     *
     * \returns 1 if a temperature value is at the warning level
     * \returns 2 if a temperature value is at critical level
     * \returns 0 otherwise (all temperatures are considered normal)
     */
   int criticalCoreTemperature( std::vector<float>&   /**< [in] the vector of temperature values to be checked*/ );

   /// Finds all CPU core usage loads
   /** Makes system call and then parses result to add usage loads to vector of values
     *
     * \returns -1 on error with system command or output file reading
     * \returns 0 on completion
     */
   int findCPULoads( std::vector<float>&   /**< [out] the vector of measured CPU usages*/ );


   /// Parses string from system call to find CPU usage loads
   /** When a valid string is read in, the value from that string is stored
     * 
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
   int parseCPULoads( std::string,   /**< [in] the string to be parsed*/
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

   /** \name Chrony Status
     * @{
     */
protected:
   std::string m_chronySourceMac;
   std::string m_chronySourceIP;
   std::string m_chronySynch;
   double m_chronySystemTime {0};
   double m_chronyLastOffset {0};
   double m_chronyRMSOffset {0};
   double m_chronyFreq {0};
   double m_chronyResidFreq {0};
   double m_chronySkew {0};
   double m_chronyRootDelay {0};
   double m_chronyRootDispersion {0};
   double m_chronyUpdateInt {0};
   std::string m_chronyLeap;
   
   pcf::IndiProperty m_indiP_chronyStatus;
   pcf::IndiProperty m_indiP_chronyStats;
   
public:
   /// Finds current chronyd status
   /** Uses the chrony tracking command
     * 
     * \returns -1 on error
     * \returns 0 on success 
     */
   int findChronyStatus();
   
   ///@}
   
   /** \name Set Latency
     * This thread spins up the cpus to minimize latency when requested
     *
     * @{
     */ 
   bool m_setLatency {false};
   
   int m_setlatThreadPrio {0}; ///< Priority of the set latency thread, should normally be > 00.

   std::thread m_setlatThread; ///< A separate thread for the actual setting of low latency

   bool m_setlatThreadInit {true}; ///< Synchronizer to ensure set lat thread initializes before doing dangerous things.
  
   pid_t m_setlatThreadID {0}; ///< Set latency thread ID.

   pcf::IndiProperty m_setlatThreadProp; ///< The property to hold the setlat thread details.
 
   ///Thread starter, called by threadStart on thread construction.  Calls setlatThreadExec.
   static void setlatThreadStart( sysMonitor * s /**< [in] a pointer to a sysMonitor instance (normally this) */);

   /// Execute the frame grabber main loop.
   void setlatThreadExec();

   pcf::IndiProperty m_indiP_setlat;
   
public:
   INDI_NEWCALLBACK_DECL(sysMonitor, m_indiP_setlat);
   
   ///@}
   
   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_coreloads * );
   
   int recordTelem( const telem_coretemps * );
   
   int recordTelem( const telem_drivetemps * );
   
   int recordTelem( const telem_usage * );
   
   int recordTelem( const telem_chrony_status * );
   
   int recordTelem( const telem_chrony_stats * );
   
   int recordCoreLoads(bool force = false);
   
   int recordCoreTemps(bool force = false);
   
   int recordDriveTemps(bool force = false);
   
   int recordUsage(bool force = false);
   
   int recordChronyStatus(bool force = false);
   
   int recordChronyStats(bool force = false);
   
   ///@}

};

inline sysMonitor::sysMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   //m_loopPause = 100000; //Set default to 1 milli-second due to mpstat averaging time of 1 sec.
   return;
}

void sysMonitor::setupConfig()
{
   config.add("diskNames", "", "diskNames", argType::Required, "", "diskNames", false, "vector<string>", "The names (/dev/sdX) of the drives to monitor");
   config.add("warningCoreTemp", "", "warningCoreTemp", argType::Required, "", "warningCoreTemp", false, "int", "The warning temperature for CPU cores.");
   config.add("criticalCoreTemp", "", "criticalCoreTemp", argType::Required, "", "criticalCoreTemp", false, "int", "The critical temperature for CPU cores.");
   config.add("warningDiskTemp", "", "warningDiskTemp", argType::Required, "", "warningDiskTemp", false, "int", "The warning temperature for the disk.");
   config.add("criticalDiskTemp", "", "criticalDiskTemp", argType::Required, "", "criticalDiskTemp", false, "int", "The critical temperature for disk.");
   
   dev::telemeter<sysMonitor>::setupConfig(config);
}

void sysMonitor::loadConfig()
{
   config(m_diskNameList, "diskNames");
   config(m_warningCoreTemp, "warningCoreTemp");
   config(m_criticalCoreTemp, "criticalCoreTemp");
   config(m_warningDiskTemp, "warningDiskTemp");
   config(m_criticalDiskTemp, "criticalDiskTemp");
   
   dev::telemeter<sysMonitor>::loadConfig(config);
}

int sysMonitor::appStartup()
{
   
   //REG_INDI_NEWPROP_NOCB(m_indiP_core_temps, "core_temps", pcf::IndiProperty::Number);
   
   
   
   REG_INDI_NEWPROP_NOCB(m_indiP_core_temps, "core_temps", pcf::IndiProperty::Number);
   m_indiP_core_temps.add(pcf::IndiElement("max"));
   m_indiP_core_temps.add(pcf::IndiElement("min"));
   m_indiP_core_temps.add(pcf::IndiElement("mean"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_core_loads, "core_loads", pcf::IndiProperty::Number);
   m_indiP_core_loads.add(pcf::IndiElement("max"));
   m_indiP_core_loads.add(pcf::IndiElement("min"));
   m_indiP_core_loads.add(pcf::IndiElement("mean"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_drive_temps, "drive_temps", pcf::IndiProperty::Number);
   findDiskTemperature(m_diskNames, m_diskTemps);
   for (unsigned int i = 0; i < m_diskTemps.size(); i++) 
   {
      m_indiP_drive_temps.add (pcf::IndiElement(m_diskNames[i]));
      m_indiP_drive_temps[m_diskNames[i]].set<double>(m_diskTemps[i]);
   }

   REG_INDI_NEWPROP_NOCB(m_indiP_usage, "resource_use", pcf::IndiProperty::Number);
   m_indiP_usage.add(pcf::IndiElement("root_usage"));
   m_indiP_usage.add(pcf::IndiElement("boot_usage"));
   m_indiP_usage.add(pcf::IndiElement("data_usage"));
   m_indiP_usage.add(pcf::IndiElement("ram_usage"));
   
   m_indiP_usage["root_usage"].set<double>(0.0);
   m_indiP_usage["boot_usage"].set<double>(0.0);
   m_indiP_usage["data_usage"].set<double>(0.0);
   m_indiP_usage["ram_usage"].set<double>(0.0);

  
   
   REG_INDI_NEWPROP_NOCB(m_indiP_chronyStatus, "chrony_status", pcf::IndiProperty::Text);
   m_indiP_chronyStatus.add(pcf::IndiElement("synch"));
   m_indiP_chronyStatus.add(pcf::IndiElement("source"));
   
   REG_INDI_NEWPROP_NOCB(m_indiP_chronyStats, "chrony_stats", pcf::IndiProperty::Number);
   m_indiP_chronyStats.add(pcf::IndiElement("system_time"));
   m_indiP_chronyStats.add(pcf::IndiElement("last_offset"));
   m_indiP_chronyStats.add(pcf::IndiElement("rms_offset"));
   
   createStandardIndiToggleSw(m_indiP_setlat, "set_latency");
   registerIndiPropertyNew(m_indiP_setlat, INDI_NEWCALLBACK(m_indiP_setlat));
   
   if(dev::telemeter<sysMonitor>::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if(threadStart( m_setlatThread, m_setlatThreadInit, m_setlatThreadID, m_setlatThreadProp, m_setlatThreadPrio, "", "set_latency", this, setlatThreadStart)  < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   state(stateCodes::READY);
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

      recordCoreTemps();
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for CPU core temps."});
   }
   
   m_coreLoads.clear();
   int rvCPULoad = findCPULoads(m_coreLoads);
   
   if(rvCPULoad >= 0)
   {
      recordCoreLoads();
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
         recordDriveTemps();
      }
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for drive temps."});
   }

   int rvDiskUsage = findDiskUsage(m_rootUsage, m_dataUsage, m_bootUsage);
   int rvRamUsage = findRamUsage(m_ramUsage);

   
   if (rvDiskUsage >= 0 && rvRamUsage >= 0)
   {
      recordUsage();
   }
   else 
   {
      log<software_error>({__FILE__, __LINE__,"Could not log values for usage."});
   }
   

   if( findChronyStatus() == 0)
   {
   }
   else
   {
      log<software_error>({__FILE__, __LINE__,"Could not get chronyd status."});
   }
   
   if(telemeter<sysMonitor>::appLogic() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
      return 0;
   }
   
   updateVals();

   return 0;
}

int sysMonitor::appShutdown()
{
   try
   {
      if(m_setlatThread.joinable())
      {
         m_setlatThread.join();
      }
   }
   catch(...){}
   
   dev::telemeter<sysMonitor>::appShutdown();
   
   return 0;
}

int sysMonitor::findCPUTemperatures(std::vector<float>& temps) 
{
   std::vector<std::string> commandList{"sensors"};
   
   std::vector<std::string> commandOutput, commandError;
   
   if(sys::runCommand(commandOutput, commandError, commandList) < 0)
   {
      if(commandOutput.size() < 1) return log<software_error,-1>({__FILE__, __LINE__});
      else return log<software_error,-1>({__FILE__, __LINE__, commandOutput[0]});
   }
   
   if(commandError.size() > 0)
   {
      for(size_t n=0; n< commandError.size(); ++n)
      {
         log<software_error>({__FILE__, __LINE__, "sensors stderr: " + commandError[n]});
      }
   }
   
   int rv = -1;
   for(size_t n=0; n < commandOutput.size(); ++n)
   {  
      float tempVal;
      if (parseCPUTemperatures(commandOutput[n], tempVal) == 0)
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
      temps = -999;
      return -1;
   }

   std::string str = line.substr(0, 5);
   if (str.compare("Core ") == 0) 
   {
      size_t st = line.find(':',0);
      if(st == std::string::npos)
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing CPU temperatures."});
         temps = -999;
         return -1;
      }
      
      ++st;
      
      size_t ed = line.find('C', st);
      if(ed == std::string::npos)
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing CPU temperatures."});
         temps = -999;
         return -1;
      }
      
      --ed;
      
      std::string temp_str = line.substr(st, ed-st);
      //std::cerr << str << " " << temp_str << "\n";
      
      
      float temp;
      try
      {
         temp = std::stof (temp_str);
      }
      catch (const std::invalid_argument& e) 
      {
         log<software_error>({__FILE__, __LINE__,"Invalid read occured when parsing CPU temperatures."});
         temps = -999;
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
            temps = -999;
            return -1;
         }
      }                           
      return 0;
   }
   else 
   {
      temps = -999;
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
   std::vector<std::string> commandOutput, commandError;
   
   if(sys::runCommand(commandOutput, commandError, commandList) < 0)
   {
      if(commandOutput.size() < 1) return log<software_error,-1>({__FILE__, __LINE__});
      else return log<software_error,-1>({__FILE__, __LINE__, commandOutput[0]});
   }
   
   if(commandError.size() > 0)
   {
      for(size_t n=0; n< commandError.size(); ++n)
      {
         log<software_error>({__FILE__, __LINE__, "mpstat stderr: " + commandError[n]});
      }
   }
   
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
   std::vector<std::string> commandList{"hddtemp"};//, "/dev/sda",  "/dev/sdb", "/dev/sdc", "/dev/sdd", "/dev/sde", "/dev/sdf"};
   for(size_t n=0;n<m_diskNameList.size();++n)
   {
      commandList.push_back(m_diskNameList[n]);
   }
   
   std::vector<std::string> commandOutput, commandError;
   
   if(sys::runCommand(commandOutput, commandError, commandList) < 0)
   {
      if(commandOutput.size() < 1) return log<software_error,-1>({__FILE__, __LINE__});
      else return log<software_error,-1>({__FILE__, __LINE__, commandOutput[0]});
   }
   
   if(commandError.size() > 0)
   {
      for(size_t n=0; n< commandError.size(); ++n)
      {
         log<software_error>({__FILE__, __LINE__, "hddtemp stderr: " + commandError[n]});
      }
   }
   
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
      driveName = "";
      hdd_temp = -999;
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
               hdd_temp = -999;
               driveName = "";
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
         hdd_temp = -999;
         driveName = "";
         return -1;
      }
   }
   
   hdd_temp = -999;
   driveName = "";
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
   
   std::vector<std::string> commandOutput, commandError;
   
   if(sys::runCommand(commandOutput, commandError, commandList) < 0)
   {
      if(commandOutput.size() < 1) return log<software_error,-1>({__FILE__, __LINE__});
      else return log<software_error,-1>({__FILE__, __LINE__, commandOutput[0]});
   }
   
   if(commandError.size() > 0)
   {
      for(size_t n=0; n< commandError.size(); ++n)
      {
         log<software_error>({__FILE__, __LINE__, "df stderr: " + commandError[n]});
      }
   }
   
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
   
   std::vector<std::string> commandOutput, commandError;
   
   if(sys::runCommand(commandOutput, commandError, commandList) < 0)
   {
      if(commandOutput.size() < 1) return log<software_error,-1>({__FILE__, __LINE__});
      else return log<software_error,-1>({__FILE__, __LINE__, commandOutput[0]});
   }
   
   if(commandError.size() > 0)
   {
      for(size_t n=0; n< commandError.size(); ++n)
      {
         log<software_error>({__FILE__, __LINE__, "free stderr: " + commandError[n]});
      }
   }
   
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

int sysMonitor::findChronyStatus()
{
   std::vector<std::string> commandList{"chronyc", "-c", "tracking"};
   
   std::vector<std::string> commandOutput, commandError;
   
   if(sys::runCommand(commandOutput, commandError, commandList) < 0)
   {
      if(commandOutput.size() < 1) return log<software_error,-1>({__FILE__, __LINE__});
      else return log<software_error,-1>({__FILE__, __LINE__, commandOutput[0]});
   }
   
   if(commandError.size() > 0)
   {
      for(size_t n=0; n< commandError.size(); ++n)
      {
         log<software_error>({__FILE__, __LINE__, "chronyc stderr: " + commandError[n]});
      }
   }
   
   if(commandOutput.size() < 1)
   {
      log<software_error>({__FILE__,__LINE__, "no response from chronyc -c"});
      return -1;
   }
   
   std::vector<std::string> results;
   mx::ioutils::parseStringVector(results, commandOutput[0], ',');
   
   if(results.size() < 1)
   {
      log<software_error>({__FILE__,__LINE__, "wrong number of fields from chronyc -c"});
      return -1;
   }
   
   static std::string last_mac;
   static std::string last_ip;
   m_chronySourceMac = results[0];
   m_chronySourceIP = results[1];
   if(m_chronySourceMac == "7F7F0101" || m_chronySourceIP == "127.0.0.1")
   {
      m_chronySynch = "NO";
      log<text_log>("chrony is not synchronized", logPrio::LOG_WARNING);
   }
   else
   {
      m_chronySynch = "YES";
   }

   if(last_mac != m_chronySourceMac || last_ip != m_chronySourceIP)
   {
      log<text_log>("chrony is synchronizing to " + m_chronySourceMac + " / " + m_chronySourceIP);
      last_mac = m_chronySourceMac;
      last_ip = m_chronySourceIP;
   }
   
   
   
   m_chronySystemTime = std::stod(results[4]);
   m_chronyLastOffset = std::stod(results[5]);
   m_chronyRMSOffset = std::stod(results[6]);
   m_chronyFreq = std::stod(results[7]);
   m_chronyResidFreq = std::stod(results[8]);
   m_chronySkew = std::stod(results[9]);
   m_chronyRootDelay = std::stod(results[10]);
   m_chronyRootDispersion = std::stod(results[11]);
   m_chronyUpdateInt = std::stod(results[12]);
   m_chronyLeap = results[13];
   
   recordChronyStatus();
   recordChronyStats();
   
   return 0;
}

int sysMonitor::updateVals()
{
   float min, max, mean;
   
   if(m_coreLoads.size() > 0)
   {
      min = m_coreLoads[0];
      max = m_coreLoads[0];
      mean = m_coreLoads[0];
      for(size_t n=1; n<m_coreLoads.size(); ++n)
      {
         if(m_coreLoads[n] < min) min = m_coreLoads[n];
         if(m_coreLoads[n] > max) max = m_coreLoads[n];
         mean += m_coreLoads[n];
      }
      mean /= m_coreLoads.size();
      
      updateIfChanged<float>(m_indiP_core_loads, {"min","max","mean"}, {min,max,mean});
   }

   
   if(m_coreTemps.size() > 0)
   {
      min = m_coreTemps[0];
      max = m_coreTemps[0];
      mean = m_coreTemps[0];
      for(size_t n=1; n<m_coreTemps.size(); ++n)
      {
         if(m_coreTemps[n] < min) min = m_coreTemps[n];
         if(m_coreTemps[n] > max) max = m_coreTemps[n];
         mean += m_coreTemps[n];
      }
      mean /= m_coreTemps.size();
      
      updateIfChanged<float>(m_indiP_core_temps, {"min","max","mean"}, {min,max,mean});
   }
   
   updateIfChanged(m_indiP_drive_temps, m_diskNames, m_diskTemps);

   updateIfChanged<float>(m_indiP_usage, {"root_usage","boot_usage","data_usage","ram_usage"}, {m_rootUsage,m_bootUsage,m_dataUsage,m_ramUsage});
   
   
   updateIfChanged(m_indiP_chronyStatus, "synch", m_chronySynch);
   updateIfChanged(m_indiP_chronyStatus, "source", m_chronySourceIP);
   
   updateIfChanged<double>(m_indiP_chronyStats, {"system_time", "last_offset", "rms_offset"}, {m_chronySystemTime, m_chronyLastOffset, m_chronyRMSOffset});
   
   if(m_setLatency)
   {
      updateSwitchIfChanged( m_indiP_setlat, "toggle", pcf::IndiElement::On, INDI_BUSY);
   }
   else
   {
      updateSwitchIfChanged( m_indiP_setlat, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   }
   
   return 0;
}

inline
void sysMonitor::setlatThreadStart( sysMonitor * s)
{
   s->setlatThreadExec();
}

inline
void sysMonitor::setlatThreadExec()
{
   m_setlatThreadID = syscall(SYS_gettid);

   //Wait fpr the thread starter to finish initializing this thread.
   while(m_setlatThreadInit == true && m_shutdown == 0)
   {
       sleep(1);
   }
   
   int fd = 0;      
   while(m_shutdown == 0)
   {
      if(m_setLatency)
      {
         if(fd <= 0)
         {
            elevatedPrivileges ep(this);
          
            for(size_t cpu =0; cpu < m_coreLoads.size(); ++cpu) ///\todo this needs error checks
            {
               std::string cpuFile = "/sys/devices/system/cpu/cpu";
               cpuFile += std::to_string(cpu);
               cpuFile += "/cpufreq/scaling_governor";
               int wfd = open( cpuFile.c_str(), O_WRONLY);
               write(wfd,"performance",sizeof("performance"));
               close(wfd);     
            }
            log<text_log>("set governor to performance", logPrio::LOG_NOTICE);

            fd = open("/dev/cpu_dma_latency", O_WRONLY);
            
            if(fd <=0) log<software_error>({__FILE__,__LINE__,"error opening cpu_dma_latency"});
            else
            {
               int l=0;
               if (write(fd, &l, sizeof(l)) != sizeof(l)) 
               {
                  log<software_error>({__FILE__,__LINE__,"error writing to cpu_dma_latency"});
               }
               else
               {
                  log<text_log>("set latency to 0", logPrio::LOG_NOTICE);
               }
            }


         }
      }
      else
      {
         if(fd != 0)
         {
            close(fd);
            fd = 0;
            log<text_log>("restored CPU latency to default", logPrio::LOG_NOTICE);
         
            elevatedPrivileges ep(this);
            for(size_t cpu =0; cpu < m_coreLoads.size(); ++cpu) ///\todo this needs error checks
            {
               std::string cpuFile = "/sys/devices/system/cpu/cpu";
               cpuFile += std::to_string(cpu);
               cpuFile += "/cpufreq/scaling_governor";
               int wfd = open( cpuFile.c_str(), O_WRONLY);
               write(wfd,"powersave",sizeof("powersave"));
               close(wfd);  
            }
            log<text_log>("set governor to powersave", logPrio::LOG_NOTICE);
         }
      }
      
      sleep(1);
   }
   
   if(fd) close(fd);
      
}

INDI_NEWCALLBACK_DEFN(sysMonitor, m_indiP_setlat)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_setlat.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("toggle")) return 0;
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      m_setLatency = false;
   }
   
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      m_setLatency = true;
   }
   return 0;
}

inline
int sysMonitor::checkRecordTimes()
{
   return telemeter<sysMonitor>::checkRecordTimes(telem_coreloads(),telem_coretemps(),telem_drivetemps(),telem_usage(),telem_chrony_status(),telem_chrony_stats());
}

int sysMonitor::recordTelem( const telem_coreloads * )
{
   return recordCoreLoads(true);
}
   
int sysMonitor::recordTelem( const telem_coretemps * )
{
   return recordCoreTemps(true);
}

int sysMonitor::recordTelem( const telem_drivetemps * )
{
   return recordDriveTemps(true);
}
  
int sysMonitor::recordTelem( const telem_usage * )
{
   return recordUsage(true);
}
 
int sysMonitor::recordTelem( const telem_chrony_status * )
{
   return recordChronyStatus(true);
}

int sysMonitor::recordTelem( const telem_chrony_stats * )
{
   return recordChronyStats(true);
}

int sysMonitor::recordCoreLoads(bool force)
{
   static std::vector<float> old_coreLoads;
   
   if(old_coreLoads.size() != m_coreLoads.size())
   {
      old_coreLoads.resize(m_coreLoads.size(), -1e30);
   }
   
   bool write = false;
   
   for(size_t n = 0; n < m_coreLoads.size(); ++n)
   {
      if( m_coreLoads[n] != old_coreLoads[n]) write = true;
   }
   
   if(force || write)
   {
      telem<telem_coreloads>(m_coreLoads);
      
      for(size_t n = 0; n < m_coreLoads.size(); ++n)
      {
         old_coreLoads[n] = m_coreLoads[n];
      }
   }
   
   return 0;
}
   
int sysMonitor::recordCoreTemps(bool force)
{
   static std::vector<float> old_coreTemps;
   
   if(old_coreTemps.size() != m_coreTemps.size())
   {
      old_coreTemps.resize(m_coreTemps.size(), -1e30);
   }
   
   bool write = false;
   
   for(size_t n = 0; n < m_coreTemps.size(); ++n)
   {
      if( m_coreTemps[n] != old_coreTemps[n]) write = true;
   }
   
   if(force || write)
   {
      telem<telem_coretemps>(m_coreTemps);
      for(size_t n = 0; n < m_coreTemps.size(); ++n)
      {
         old_coreTemps[n] = m_coreTemps[n];
      }
   }
   
   return 0;
}

int sysMonitor::recordDriveTemps(bool force)
{
   static std::vector<std::string> old_diskNames;
   static std::vector<float> old_diskTemps;
   
   if(old_diskTemps.size() != m_diskTemps.size() || old_diskNames.size() != m_diskNames.size())
   {
      old_diskNames.resize(m_diskNames.size());
      old_diskTemps.resize(m_diskTemps.size(), -1e30);
   }
   
   bool write = false;
   
   for(size_t n = 0; n < m_diskTemps.size(); ++n)
   {
      if( m_diskTemps[n] != old_diskTemps[n] || m_diskNames[n] != old_diskNames[n]) write = true;
   }
   
   if(force || write)
   {
      telem<telem_drivetemps>({m_diskNames, m_diskTemps});
      for(size_t n = 0; n < m_diskTemps.size(); ++n)
      {
         old_diskNames[n] = m_diskNames[n];
         old_diskTemps[n] = m_diskTemps[n];
      }
   }
   
   return 0;
}

int sysMonitor::recordUsage(bool force)
{
   static float old_ramUsage = 0;
   static float old_bootUsage = 0;
   static float old_rootUsage = 0;
   static float old_dataUsage = 0;
   
   if( old_ramUsage != m_ramUsage || old_bootUsage != m_bootUsage || old_rootUsage != m_rootUsage || old_dataUsage != m_dataUsage || force)
   {
      telem<telem_usage>({m_ramUsage, m_bootUsage, m_rootUsage, m_dataUsage});
      
      old_ramUsage = m_ramUsage;
      old_bootUsage = m_bootUsage;
      old_rootUsage = m_rootUsage;
      old_dataUsage = m_dataUsage;
   }
   
   return 0;
}

int sysMonitor::recordChronyStatus(bool force)
{
   static std::string old_chronySourceMac;
   static std::string old_chronySourceIP;
   static std::string old_chronySynch;
   static std::string old_chronyLeap;
   
   
   if( old_chronySourceMac != m_chronySourceMac || old_chronySourceIP != m_chronySourceIP || old_chronySynch != m_chronySynch || old_chronyLeap != m_chronyLeap || force)
   {
      telem<telem_chrony_status>({m_chronySourceMac, m_chronySourceIP, m_chronySynch, m_chronyLeap});
      
      old_chronySourceMac = m_chronySourceMac;
      old_chronySourceIP = m_chronySourceIP;
      old_chronySynch = m_chronySynch;
      old_chronyLeap = m_chronyLeap;
   }
   
   return 0;
}

int sysMonitor::recordChronyStats(bool force)
{
   double old_chronySystemTime = 1e50; //to force an update the first time no matter what
   double old_chronyLastOffset = 0;
   double old_chronyRMSOffset = 0;
   double old_chronyFreq = 0;
   double old_chronyResidFreq = 0;
   double old_chronySkew = 0;
   double old_chronyRootDelay = 0;
   double old_chronyRootDispersion = 0;
   double old_chronyUpdateInt = 0;

   if( old_chronySystemTime == m_chronySystemTime || old_chronyLastOffset == m_chronyLastOffset ||
          old_chronyRMSOffset == m_chronyRMSOffset || old_chronyFreq == m_chronyFreq ||
             old_chronyResidFreq == m_chronyResidFreq || old_chronySkew == m_chronySkew ||
                old_chronyRootDelay == m_chronyRootDelay || old_chronyRootDispersion == m_chronyRootDispersion ||
                   old_chronyUpdateInt == m_chronyUpdateInt || force )
   {
      
      telem<telem_chrony_stats>({m_chronySystemTime, m_chronyLastOffset, m_chronyRMSOffset, m_chronyFreq, m_chronyResidFreq, m_chronySkew, 
                                     m_chronyRootDelay, m_chronyRootDispersion, m_chronyUpdateInt});
      
      old_chronySystemTime = m_chronySystemTime;
      old_chronyLastOffset = m_chronyLastOffset;
      old_chronyRMSOffset = m_chronyRMSOffset;
      old_chronyFreq = m_chronyFreq;
      old_chronyResidFreq = m_chronyResidFreq;
      old_chronySkew = m_chronySkew;
      old_chronyRootDelay = m_chronyRootDelay;
      old_chronyRootDispersion = m_chronyRootDispersion;
      old_chronyUpdateInt = m_chronyUpdateInt;
      
   }
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
