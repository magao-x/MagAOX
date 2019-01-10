/** \file sysMonitor.hpp
  * \brief The MagAO-X sysMonitor app header file.
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

/** MagAO-X application to do math on some numbers
  *
  */
		class sysMonitor : public MagAOXApp<> {

		protected:
			int m_warningCoreTemp = 0;
			int m_criticalCoreTemp = 0;
			int m_warningDiskTemp = 0;
			int m_criticalDiskTemp = 0;

			pcf::IndiProperty core_loads;
			pcf::IndiProperty core_temps;
			pcf::IndiProperty drive_temps;
			pcf::IndiProperty root_usage;
			pcf::IndiProperty boot_usage;
			pcf::IndiProperty data_usage;
			pcf::IndiProperty ram_usage_indi;

			int updateVals();

			std::vector<float> coreTemps;
			std::vector<float> cpu_core_loads;
			std::vector<float> diskTemp;
			float rootUsage = 0, dataUsage = 0, bootUsage = 0;
			float ramUsage = 0;

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

			int findCPUTemperatures(std::vector<float>&);
			int parseCPUTemperatures(std::string, float&);
			int criticalCoreTemperature(std::vector<float>&);
			int findCPULoads(std::vector<float>&);
			int parseCPULoads(std::string, float&);
			int findDiskTemperature(std::vector<float>&);
			int parseDiskTemperature(std::string, float&);
			int criticalDiskTemperature(std::vector<float>&);
			int findDiskUsage(float&, float&, float&);
			int parseDiskUsage(std::string, float&, float&, float&);
			int findRamUsage(float&);
			int parseRamUsage(std::string, float&);


		};

		sysMonitor::sysMonitor() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
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


			int i;
			std::string coreStr = "core";

			findCPULoads(cpu_core_loads);
			for (i = 0; i < cpu_core_loads.size(); i++) {
				coreStr.append(std::to_string(i));
				core_loads.add (pcf::IndiElement(coreStr));
   				core_loads[coreStr].set<double>(0.0);
				coreStr.pop_back();
			}

			findCPUTemperatures(coreTemps);
			for (i = 0; i < coreTemps.size(); i++) {
				coreStr.append(std::to_string(i));
				core_temps.add (pcf::IndiElement(coreStr));
   				core_temps[coreStr].set<double>(0.0);
				coreStr.pop_back();
			}

			std::string driveStr = "drive";
			findDiskTemperature(diskTemp);
			for (i = 0; i < diskTemp.size(); i++) {
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
			//std::cout << m_warningCoreTemp << " " << m_criticalCoreTemp << " " << m_warningDiskTemp << " " << m_criticalDiskTemp << std::endl;

			bool warningLog = false;
			bool alertLog = false;

			coreTemps.clear();
			int rvCPUTemp = findCPUTemperatures(coreTemps);
			for (auto i: coreTemps)
			{
				std::cout << "Core temp: " << i << ' ';
			}	
			std::cout << std::endl;
			int rv = criticalCoreTemperature(coreTemps);

			cpu_core_loads.clear();
			int rvCPULoad = findCPULoads(cpu_core_loads);
			for (auto i: cpu_core_loads)
			{
				std::cout << "CPU load: " << i << ' ';
			}
			std::cout << std::endl;
			
			if (rv == 1) {
				log<core_mon>({coreTemps, cpu_core_loads}, logPrio::LOG_WARNING);
			} else if (rv == 2) {
				log<core_mon>({coreTemps, cpu_core_loads}, logPrio::LOG_ALERT);
			} else {
				log<core_mon>({coreTemps, cpu_core_loads}, logPrio::LOG_INFO);
			}
			
			diskTemp.clear();
			int rvDiskTemp = findDiskTemperature(diskTemp);
			for (auto i: diskTemp)
			{
				std::cout << "Disk temp: " << i << ' ';
			}
			std::cout << std::endl;
			rv = criticalDiskTemperature(diskTemp);

			int rvDiskUsage = findDiskUsage(rootUsage, dataUsage, bootUsage);
			std::cout << "/ usage: " << rootUsage << std::endl;
			std::cout << "/data usage: " << dataUsage << std::endl;	
			std::cout << "/boot usage: " << bootUsage << std::endl;

			if (rv == 1) {
				log<drive_mon>({diskTemp, rootUsage, dataUsage, bootUsage}, logPrio::LOG_WARNING);
			} else if (rv == 2) {
				log<drive_mon>({diskTemp, rootUsage, dataUsage, bootUsage}, logPrio::LOG_ALERT);
			} else {
				log<drive_mon>({diskTemp, rootUsage, dataUsage, bootUsage}, logPrio::LOG_INFO);
			}

			int rvRamUsage = findRamUsage(ramUsage);
			std::cout << "Ram usage: " << ramUsage << std::endl;

			log<ram_usage>({ramUsage}, logPrio::LOG_INFO);

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
         		float tempVal;
         		int rv = parseCPUTemperatures(line, tempVal);
         		if (rv == 0)
         			temps.push_back(tempVal);
         	}
         	return 0;
     	}

	    int sysMonitor::parseCPUTemperatures(std::string line, float& temps) 
	    {
	    	if (line.length() <= 1) return 1;
	     	std::string str = line.substr(0, 5);
	     	if (str.compare("Core ") == 0) 
	     	{
	     		std::string temp_str = line.substr(17, 4);
	     		float temp;
	     		try
	     		{
	     			temp = std::stof (temp_str);
	     		}
	     		catch (const std::invalid_argument& e) {
	     			std::cerr << "Invalid read occuered when parsing CPU temperatures" << std::endl;
	     			return 1;
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
	     			} catch (const std::invalid_argument& e) {
	     				std::cerr << "Invalid read occuered when parsing warning CPU temperature" << std::endl;
	     				return 1;
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
	     			} catch (const std::invalid_argument& e) {
	     				std::cerr << "Invalid read occuered when parsing critical CPU temperature" << std::endl;
	     				return 1;
	     			}
	     		}
	     		return 0;
	     	}
	     	else 
	     	{
	     		return 1;
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
	     			if (rv < 2) rv = 1;
	     			
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
	        	//handle error
	         	std::cerr << "There's been an error with the system command" << std::endl;
	         	return 1;
	        }

	        std::ifstream inFile;
	        inFile.open("/dev/shm/cpuload");
	        if (!inFile) 
	        {
	        	std::cerr << "Unable to open file" << std::endl;
	         	return 1;
	        }
	        float cpu_load;
	        // Want to start at third line
	        getline (inFile,line);
	        getline (inFile,line);
	        getline (inFile,line);
	        getline (inFile,line);
	        while (getline (inFile,line)) 
	        {
	         	float loadVal;
	         	int rv = parseCPULoads(line, loadVal);
	         	if (rv == 0)
	         		loads.push_back(loadVal);
	        }

	        return 0;
	    }

	    int sysMonitor::parseCPULoads(std::string line, float& loadVal)
	    {
	     	if (line.length() <= 1) return 1;
	     	std::istringstream iss(line);
	     	std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
	     	float cpu_load;
	     	try
	     	{
	     		cpu_load = 100.0 - std::stof(tokens[12]);
	     	}
	     	catch (const std::invalid_argument& e)
	     	{
	     		std::cerr << "Invalid read occuered when parsing CPU loads" << std::endl;
	     		return 1;
	     	}
	     	cpu_load /= 100;
	     	loadVal = cpu_load;
	     	return 0;
	    }

	    int sysMonitor::findDiskTemperature(std::vector<float>& hdd_temp) 
	    {
	     	char command[35];
	     	std::string line;

	        // For hard drive temp utility:
	        // wget http://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/h/hddtemp-0.3-0.31.beta15.el7.x86_64.rpm (binary package)
	        // su
	        // rpm -Uvh hddtemp-0.3-0.31.beta15.el7.x86_64.rpm
	        // Check install with rpm -q -a | grep -i hddtemp

	     	strcpy( command, "hddtemp > /dev/shm/hddtemp" );
	     	int rv = system(command);
	        if(rv == -1) //system call error
	        {
	            //handle error
	         	std::cerr << "There's been an error with the system command" << std::endl;
	         	return 1;
	        }

	        std::ifstream inFile;
	        inFile.open("/dev/shm/hddtemp");
	        if (!inFile) 
	        {
	         	std::cerr << "Unable to open file" << std::endl;
	         	return 1;
	        }
	         
	        while (getline (inFile,line)) 
	    	{
	         	float tempVal;
	         	int rvHddTemp = parseDiskTemperature(line, tempVal);
	         	if (rvHddTemp == 0) 
	         		hdd_temp.push_back(tempVal);
	        }
	        return 0;
	    }

	    int sysMonitor::parseDiskTemperature(std::string line, float& hdd_temp) 
	    {
	     	float tempValue;
	     	if (line.length() <= 1) return 1;
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
	     			catch (const std::invalid_argument& e) {
	     				std::cerr << "Invalid read occuered when parsing disk temperature" << std::endl;
	     				return 1;
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
	     	return 1;
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
	     			if (rv < 2) rv = 1;
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
	        	//handle error
	         	std::cerr << "There's been an error with the system command" << std::endl;
	         	return 1;
	        }

	        std::ifstream inFile;
	        inFile.open("/dev/shm/diskusage");
	        if (!inFile) 
	        {
	         	std::cerr << "Unable to open file" << std::endl;
	         	return 1;
	        }
	        rv = 1; 
	        // Want second line
	        getline (inFile,line);

	        while(getline(inFile,line)) 
	        {
	        	int rvDiskUsage = parseDiskUsage(line, rootUsage, dataUsage, bootUsage);
	        	if (rvDiskUsage == 0)
	        		rv = 0;
	        }
	         
	        return rv;
	    }

	    int sysMonitor::parseDiskUsage(std::string line, float& rootUsage, float& dataUsage, float& bootUsage) 
	    {
	     	if (line.length() <= 1) return 1;

	     	std::istringstream iss(line);
	     	std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
	     	if (tokens[5].compare("/") == 0)
	     	{
	     		tokens[4].pop_back();
	     		try
	     		{
	     			rootUsage = std::stof (tokens[4]);
	     			return 0;
	     		}
	     		catch (const std::invalid_argument& e) 
	     		{
	     			std::cerr << "Invalid read occuered when parsing disk usage" << std::endl;
	     			return 1;
	     		}
	     	} else if (tokens[5].compare("/data") == 0)
	     	{
	     		tokens[4].pop_back();
	     		try
	     		{
	     			dataUsage = std::stof (tokens[4]);
	     			return 0;
	     		}
	     		catch (const std::invalid_argument& e) 
	     		{
	     			std::cerr << "Invalid read occuered when parsing disk usage" << std::endl;
	     			return 1;
	     		}
	     	} else if (tokens[5].compare("/boot") == 0)
	     	{
	     		tokens[4].pop_back();
	     		try
	     		{
	     			bootUsage = std::stof (tokens[4]);
	     			return 0;
	     		}
	     		catch (const std::invalid_argument& e) 
	     		{
	     			std::cerr << "Invalid read occuered when parsing disk usage" << std::endl;
	     			return 1;
	     		}
	     	}        
	     	return 1;
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
	            //handle error
	        	std::cerr << "There's been an error with the system command" << std::endl;
	         	return 1;
	        }

	        std::ifstream inFile;
	        inFile.open("/dev/shm/ramusage");
	        if (!inFile) 
	        {
	         	std::cerr << "Unable to open file" << std::endl;
	         	return 1;
	        }
	         
	        // Want second line
	        getline (inFile,line);
	        getline (inFile,line);
	        int rvRamUsage = parseRamUsage(line, ramUsage);
	        return rvRamUsage;
	    }

	    int sysMonitor::parseRamUsage(std::string line, float& ramUsage) 
	    {
	     	if (line.length() <= 1) return 1;
	     	std::istringstream iss(line);
	     	std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},std::istream_iterator<std::string>{}};
	     	try
	     	{
	     		ramUsage = std::stof(tokens[2])/std::stof(tokens[1]);
	     		if (ramUsage > 1 || ramUsage == 0)
	     		{
	     			ramUsage = -1;  
	     			return 1;
	     		}
	     		return 0;
	     	}
	     	catch (const std::invalid_argument& e) 
	     	{
	     		std::cerr << "Invalid read occuered when parsing ram usage" << std::endl;
	     		return 1;
	     	}
	    }
	    int sysMonitor::updateVals()
		{
			int i;
			std::string coreStr = "core";
			std::string driveStr = "drive";

			for (i = 0; i < cpu_core_loads.size(); i++) {
				coreStr.append(std::to_string(i));
				core_loads[coreStr] = cpu_core_loads[i];
				coreStr.pop_back();
			}
			core_loads.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (core_loads);

			for (i = 0; i < coreTemps.size(); i++) {
				coreStr.append(std::to_string(i));
				core_temps[coreStr] = coreTemps[i];
				coreStr.pop_back();
			}
			core_temps.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (core_temps);

		   	for (i = 0; i < diskTemp.size(); i++) {
				driveStr.append(std::to_string(i));
				drive_temps[driveStr] = diskTemp[i];
				driveStr.pop_back();
			}
			drive_temps.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (drive_temps);

		   	root_usage["root_usage"] = rootUsage;
		   	root_usage.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (root_usage);

		   	boot_usage["boot_usage"] = bootUsage;
		   	boot_usage.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (boot_usage);

		   	data_usage["data_usage"] = dataUsage;
		   	data_usage.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (data_usage);

		   	ram_usage_indi["ram_usage"] = ramUsage;
		   	ram_usage_indi.setState (pcf::IndiProperty::Ok);
		   	if(m_indiDriver) m_indiDriver->sendSetProperty (ram_usage_indi);

			
		   	return 0;
		}

   } //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
