
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
			int m_warningCoreTemp = 0;
			int m_criticalCoreTemp = 0;
			int m_warningDiskTemp = 0;
			int m_criticalDiskTemp = 0;


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
			config.add("warningCoreTemp", "", "warningCoreTemp", mx::argType::Required, "", "warningCoreTemp", false, "int", "The warning temperature for CPU cores.");
			config.add("criticalCoreTemp", "", "criticalCoreTemp", mx::argType::Required, "", "criticalCoreTemp", false, "int", "The critical temperature for CPU cores.");
			config.add("warningDiskTemp", "", "warningDiskTemp", mx::argType::Required, "", "warningDiskTemp", false, "int", "The warning temperature for the disk.");
			config.add("criticalDiskTemp", "", "criticalDiskTemp", mx::argType::Required, "", "criticalDiskTemp", false, "int", "The critical temperature for disk.");
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
			return 0;
		}

		int sysMonitor::appLogic()
		{
			std::vector<float> coreTemps;
			int rvCPUTemp = findCPUTemperatures(coreTemps);
			for (auto i: coreTemps)
			{
				std::cout << "Core temps: " << i << ' ';
				std::string line = "Core temp: ";
				std::string result = line + std::to_string(i);
				log<text_log>( result, logPrio::LOG_CRITICAL );
			}
			std::cout << std::endl;
			criticalCoreTemperature(coreTemps);

			std::vector<float> cpu_core_loads;
			int rvCPULoad = findCPULoads(cpu_core_loads);
			for (auto i: cpu_core_loads)
			{
				std::cout << "CPU loads: " << i << ' ';
				std::string line = "CPU load: ";
				std::string result = line + std::to_string(i);
				log<text_log>( result, logPrio::LOG_CRITICAL );
			}
			std::cout << std::endl;

			std::vector<float> diskTemp;
			int rvDiskTemp = findDiskTemperature(diskTemp);
			for (auto i: diskTemp)
			{
				std::cout << "Disk temps: " << i << ' ';
				std::string line = "Disk temp: ";
				std::string result = line + std::to_string(i);
				log<text_log>( result, logPrio::LOG_CRITICAL );
			}
			std::cout << std::endl;
			criticalDiskTemperature(diskTemp);

			float rootUsage = 0, dataUsage = 0, bootUsage = 0;
			int rvDiskUsage = findDiskUsage(rootUsage, dataUsage, bootUsage);

			std::cout << "/ usage: " << rootUsage << std::endl;
			std::string line = "Usage in '/': ";
			std::string result = line + std::to_string(rootUsage);
			log<text_log>( result, logPrio::LOG_CRITICAL );
			std::cout << "/data usage: " << dataUsage << std::endl;
			line = "Usage in '/data': ";
			result = line + std::to_string(dataUsage);
			log<text_log>( result, logPrio::LOG_CRITICAL );
			std::cout << "/boot usage: " << bootUsage << std::endl;
			line = "Usage in '/boot': ";
			result = line + std::to_string(bootUsage);
			log<text_log>( result, logPrio::LOG_CRITICAL );

			float ramUsage = 0;
			int rvRamUsage = findRamUsage(ramUsage);
			std::cout << "Ram usage: " << ramUsage << std::endl;
			line = "Ram usage: ";
			result = line + std::to_string(ramUsage);
			log<text_log>( result, logPrio::LOG_CRITICAL );
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

	    int sysMonitor::criticalCoreTemperature(std::vector<float>& temps)
	    {
	    	float warningTempValue = 80, criticalTempValue = 90, iterator = 1;
	     	for (std::vector<float>::const_iterator i = temps.begin(); i != temps.end(); ++i)
	     	{
	     		int temp = *i;
	     		if (temp >=warningTempValue && temp < criticalTempValue ) 
	     		{
	     			std::cout << "Warning temperature for Core " << iterator << std::endl;
	     		}
	     		else if (temp >= criticalTempValue) 
	     		{   
	     			std::cout << "Critical temperature for Core " << iterator << std::endl;
	     		}
	     		++iterator;
	     	}
	    }

	    int sysMonitor::findCPULoads(std::vector<float>& cpu_core_loads) 
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
	        int cores = 0;
	        // Want to start at third line
	        getline (inFile,line);
	        getline (inFile,line);
	        getline (inFile,line);
	        getline (inFile,line);
	        while (getline (inFile,line)) 
	        {
	        	cores++;
	         	float loadVal;
	         	int rv = parseCPULoads(line, loadVal);
	         	if (rv == 0)
	         		cpu_core_loads.push_back(loadVal);
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
	     	int warningTempValue = 80, criticalTempValue = 90;
	     	for(auto it: v)
	     	{
	     		float temp = it;
	     		if (temp >=warningTempValue && temp < criticalTempValue )
	     		{
	     			std::cout << "Warning temperature for Disk" << std::endl;
	     			rv = 1;
	     		}  
	     		else if (temp >= criticalTempValue) 
	     		{   
	     			std::cout << "Critical temperature for Disk " << std::endl;
	     			rv = 1;
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
	         
	        // Want second line
	        getline (inFile,line);
	        while(getline(inFile,line)) 
	        {
	        	int rvDiskUsage = parseDiskUsage(line, rootUsage, dataUsage, bootUsage);
	        }
	         
	        return 0;
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
	        return 0;
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

   } //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
