/** \file xindiserver.hpp
  * \brief The MagAO-X INDI Server wrapper header.
  *
  * \ingroup xindiserver_files
  */

#ifndef xindiserver_hpp
#define xindiserver_hpp

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup xindiserver INDI Server wrapper.
  * \brief Manages INDI server in the MagAO-X context.
  *
  * <a href="..//apps_html/page_module_xindiserver.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup xindiserver_files xindiserver Files
  * \ingroup xindiserver
  */

namespace MagAOX
{
namespace app
{
   
#define SSHTUNNEL_E_NOTUNNELS (-10)

/// Structure to hold an sshTunnel specification, used for created command line args for indiserver
struct sshTunnel 
{
   std::string m_remoteHost;
   int m_localPort {0};
   int m_remotePort {0};
   int m_monitorPort {0};
};

///The map used to hold tunnel specifications.
typedef std::unordered_map<std::string, sshTunnel> tunnelMapT;

/// Create the tunnel map from a configurator
/**
  * \returns 0 on success 
  * \returns SSHTUNNEL_E_NOTUNNELS if no tunnels are found (< 0).
  */ 
inline 
int loadSSHTunnelConfigs( tunnelMapT & tmap, ///< [out] the tunnel map which will be populated
                          mx::app::appConfigurator & config ///< [in] the configurator which contains tunnel specifications.
                        )
{
   std::vector<std::string> sections;

   config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      return SSHTUNNEL_E_NOTUNNELS;
   }

   size_t matched = 0;
   
   //Now see if any sections match a tunnel specification
   for(size_t i=0; i< sections.size(); ++i)
   {
      //A tunnel as remoteHost, localPort, and remotePort.
      if( config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "remoteHost" )) &&
             config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "localPort" )) &&
                config.isSetUnused(mx::app::iniFile::makeKey(sections[i], "remotePort" )) )
      {
         
         std::string remoteHost;
         int localPort = 0;
         int remotePort = 0;
         int monitorPort = 0;
      
         config.configUnused( remoteHost, mx::app::iniFile::makeKey(sections[i], "remoteHost" ) );
         config.configUnused( localPort, mx::app::iniFile::makeKey(sections[i], "localPort" ) );
         config.configUnused( remotePort, mx::app::iniFile::makeKey(sections[i], "remotePort" ) );
         config.configUnused( monitorPort, mx::app::iniFile::makeKey(sections[i], "monitorPort" ) );
         
         tmap[sections[i]] = sshTunnel({remoteHost, localPort, remotePort, monitorPort});
      
         ++matched;
      }
   }

   if(matched == 0) return SSHTUNNEL_E_NOTUNNELS;
   
   return 0;
}


#define XINDISERVER_E_BADDRIVERSPEC (-100)
#define XINDISERVER_E_DUPLICATEDRIVER (-101)
#define XINDISERVER_E_VECTOREXCEPT (-102)
#define XINDISERVER_E_NOTUNNELS (-103) 
#define XINDISERVER_E_TUNNELNOTFOUND (-104)
 

 
/** The INDI Server wrapper application class.
  *
  * \ingroup xindiserver
  * 
  */  
class xindiserver : public MagAOXApp<false>
{

   //Give the test harness access.
   friend class xindiserver_test;

protected:

   int indiserver_m {-1};  ///< The indiserver MB behind setting (passed to indiserver)
   bool indiserver_n {false}; ///< The indiserver ignore /tmp/noindi flag (passed to indiserver)
   int indiserver_p {-1}; ///< The indiserver port (passed to indiserver)
   int indiserver_v {-1}; ///< The indiserver verbosity (passed to indiserver)
   bool indiserver_x {false}; ///< The indiserver terminate after last exit flag (passed to indiserver)
   
   std::string m_driverPath; ///< The path to the local drivers
   std::vector<std::string> m_local; ///< List of local drivers passed in by config
   std::vector<std::string> m_remote; ///< List of remote drivers passed in by config
   std::unordered_set<std::string> m_driverNames; ///< List of driver names processed for command line, used to prevent duplication.
   
   tunnelMapT m_tunnels; ///< Map of the ssh tunnels, used for processing the remote drivers in m_remote.
   
   std::vector<std::string> m_indiserverCommand; ///< The command line arguments to indiserver
      
   pid_t m_isPID {0}; ///< The PID of the indiserver process
   
   int m_isSTDERR {-1}; ///< The output of stderr of the indiserver process
   int m_isSTDERR_input {-1}; ///< The input end of stderr, used to wake up the log thread on shutdown.
   
   int m_isLogThreadPrio {0}; ///< Priority of the indiserver log capture thread, should normally be 0.
   
   std::thread m_isLogThread; ///< A separate thread for capturing indiserver logs
   
public:
   /// Default c'tor.
   xindiserver();

   /// D'tor, declared and defined for noexcept.
   ~xindiserver() noexcept
   {}
   
   virtual void setupConfig();

   virtual void loadConfig();

   ///Construct the vector of indiserver arguments for exec.
   /** The first entry is argv[0], that is "indiserver".
     *
     * \returns 0 on success.
     * \returns -1 on error, including if an exception is caught.
     */ 
   int constructIndiserverCommand(std::vector<std::string> & indiserverCommand /**< [out] the vector of command line arguments for exec */);
   
   ///Validate the local driver strings, and append them to the indi server command line arguments.
   /** Checks that the local driver specs don't contain @,:, or /.  Then prepends the MagAO-X standard
     * driver path, and then appends to the driverArgs vector passed in.
     *
     * \returns 0 on success.
     * \returns -1 on error, either from failed validation or an exception in std::vector.
     */ 
   int addLocalDrivers( std::vector<std::string> & driverArgs /**< [out] the vector of command line arguments for exec*/);
   
   
   ///Validate the remote driver entries, and append them to the indi server command line arguments.
   /** Parses the remote driver specs, then
     * constructs the command line arguments and appends them to the driverArgs vector passed in.
     *
     * \returns 0 on success.
     * \returns -1 on error, either from failed validation or an exception in std::vector.
     */ 
   int addRemoteDrivers( std::vector<std::string> & driverArgs /**< [out] the vector of command line arguments for exec*/);
   
   
   ///Forks and exec's the indiserver process with the command constructed from local, remote, and hosts.
   /** Also saves the PID and stderr pipe file descriptors for log capture.
     *
     * \returns 0 on success
     * \returns -1 on error (fatal)
     */
   int forkIndiserver();
      
   ///Thread starter, called by isLogThreadStart on thread construction.  Calls isLogThreadExec.
   static void _isLogThreadStart( xindiserver * l /**< [in] a pointer to a xindiserver instance (normally this) */);

   /// Start the log capture.
   int isLogThreadStart();

   /// Execute the log capture.
   void isLogThreadExec();
   
   /// Process a log entry from indiserver, putting it into MagAO-X standard form 
   int processISLog( std::string logs );
   
   /// Startup functions
   /** 
     * Forks and execs the actual indiserver.  Captures its stderr output for logging.
     */
   virtual int appStartup();

   /// Implementation of the FSM for xindiserver.
   virtual int appLogic();

   /// Kills indiserver, and wakes up the log capture thread.
   virtual int appShutdown();
   

};

inline
xindiserver::xindiserver() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   //Use the sshTunnels.conf config file
   m_configBase = "sshTunnels";
   
   return;
}

inline
void xindiserver::setupConfig()
{
   config.add("indiserver.m", "m", "", argType::Required, "indiserver", "m", false,  "int", "indiserver kills client if it gets more than this many MB behind, default 50");
   config.add("indiserver.N", "N", "", argType::True, "indiserver", "N", false,  "bool", "indiserver: ignore /tmp/noindi.  Capitalized to avoid conflict with --name");
   config.add("indiserver.p", "p", "", argType::Required, "indiserver", "p", false,  "int", "indiserver: alternate IP port, default 7624");
   config.add("indiserver.v", "v", "", argType::True, "indiserver", "v", false,  "int", "indiserver: log verbosity, -v, -vv or -vvv");
   config.add("indiserver.x", "x", "", argType::True, "indiserver", "x", false,  "bool", "exit after last client disconnects -- FOR PROFILING ONLY");
   
   config.add("local.drivers","L", "local" , argType::Required, "local", "drivers", false,  "vector string", "List of local drivers to start.");
   config.add("remote.drivers","R", "remote" , argType::Required, "remote", "drivers", false,  "vector string", "List of remote drivers to start, in the form of name@tunnel, where tunnel is the name of a tunnel specified in sshTunnels.conf.");

}



inline
void xindiserver::loadConfig()
{
   //indiserver config:
   config(indiserver_m, "indiserver.m");
   config(indiserver_n, "indiserver.N");
   config(indiserver_p, "indiserver.p");
   
   indiserver_v = config.verbosity("indiserver.v");
   
   config(indiserver_x, "indiserver.x");
   
   config(m_local, "local.drivers");
   config(m_remote, "remote.drivers");
   
   loadSSHTunnelConfigs(m_tunnels, config);
}

inline
int xindiserver::constructIndiserverCommand( std::vector<std::string> & indiserverCommand)
{
   try
   {
      indiserverCommand.push_back("indiserver");
        
      if(indiserver_m > 0) 
      {
         indiserverCommand.push_back("-m");
         indiserverCommand.push_back(mx::ioutils::convertToString(indiserver_m));
      }
      
      if(indiserver_n == true) indiserverCommand.push_back("-n");
      
      if(indiserver_p > 0) 
      {
         indiserverCommand.push_back("-p");
         indiserverCommand.push_back(mx::ioutils::convertToString(indiserver_p));
      }
      
      if(indiserver_v == 1) indiserverCommand.push_back("-v");
      
      if(indiserver_v == 2) indiserverCommand.push_back("-vv");
      
      if(indiserver_v >= 3) indiserverCommand.push_back("-vvv");
      
      if(indiserver_x == true) indiserverCommand.push_back("-x");
   }
   catch(...)
   {
      log<software_critical>(software_log::messageT(__FILE__, __LINE__, "Exception thrown by std::vector."));
      return -1;
   }
   
   return 0;
}
 
inline
int xindiserver::addLocalDrivers( std::vector<std::string> & driverArgs )
{
   m_driverPath = MAGAOX_path;
   m_driverPath += "/";
   m_driverPath += MAGAOX_driverRelPath;
   m_driverPath += "/";
   
   for(size_t i=0; i< m_local.size(); ++i)
   {
      size_t bad = m_local[i].find_first_of("@:/", 0);
      
      if(bad != std::string::npos)
      {
         log<software_critical>({__FILE__, __LINE__, "Local driver can't have host spec or path(@,:,/): " + m_local[i]});
         
         return XINDISERVER_E_BADDRIVERSPEC;
      }
      
      if( m_driverNames.count(m_local[i]) > 0)
      {
         log<software_critical>({__FILE__, __LINE__, "Duplicate driver name: " + m_local[i]});
         return XINDISERVER_E_DUPLICATEDRIVER;
      }
      
      m_driverNames.insert(m_local[i]);
      
      std::string dname = m_driverPath + m_local[i];
      
      try
      {
         driverArgs.push_back(dname);
      }
      catch(...)
      {
         log<software_critical>({__FILE__, __LINE__, "Exception thrown by std::vector"});
         return XINDISERVER_E_VECTOREXCEPT;
      }
   }
   
   return 0;
}

inline
int xindiserver::addRemoteDrivers( std::vector<std::string> & driverArgs )
{
   for(size_t i=0; i < m_remote.size(); ++i)
   {
      std::string driver;
      std::string tunnel;
      
      size_t p = m_remote[i].find('@');
      
      if(p == 0 || p == std::string::npos)
      {
         log<software_critical>({__FILE__, __LINE__, "Error parsing remote driver specification: " + m_remote[i] + "\n"});         
         return XINDISERVER_E_BADDRIVERSPEC;
      }
      
      driver = m_remote[i].substr(0, p);
      tunnel = m_remote[i].substr(p+1);
      
      if( m_driverNames.count(driver) > 0)
      {
         log<software_critical>({__FILE__, __LINE__, "Duplicate driver name: " + driver});
         return XINDISERVER_E_DUPLICATEDRIVER;
      }
      
      std::ostringstream oss;
      
      if(m_tunnels.size() == 0)
      {
         log<software_critical>({__FILE__, __LINE__, "No tunnels specified.\n"});         
         return XINDISERVER_E_NOTUNNELS;
      }
      
      if(m_tunnels.count(tunnel) != 1)
      {
         log<software_critical>({__FILE__, __LINE__, "Tunnel not found for: " + m_remote[i] + "\n"});         
         return XINDISERVER_E_TUNNELNOTFOUND;
      }
      
      m_driverNames.insert(driver);
      
      oss << driver << "@localhost:" << m_tunnels[tunnel].m_localPort;
      
      try
      {
         driverArgs.push_back(oss.str());
      }
      catch(...)
      {
         log<software_critical>({__FILE__, __LINE__, "Exception thrown by vector::push_back."});
         return XINDISERVER_E_VECTOREXCEPT;
      }
   }
   
   return 0;

}

inline
int xindiserver::forkIndiserver()
{
   
   if(m_log.logLevel() >= logPrio::LOG_INFO)
   {
      std::string coml = "Starting indiserver with command: ";
      for(size_t i=0;i<m_indiserverCommand.size();++i)
      {
         coml += m_indiserverCommand[i];
         coml += " ";
      }
   
      log<text_log>(coml);
   }
   
   int filedes[2];
   if (pipe(filedes) == -1) 
   {
      log<software_error>({__FILE__, __LINE__, errno});
      return -1;
   }


   m_isPID = fork();
   
   if(m_isPID < 0)
   {
      log<software_error>({__FILE__, __LINE__, errno, "fork failed"});
      return -1;
   }

   
   if(m_isPID == 0)
   {
      //Route STDERR of child to pipe input.
      while ((dup2(filedes[1], STDERR_FILENO) == -1) && (errno == EINTR)) {}
      close(filedes[1]);
      close(filedes[0]);
  
      const char ** drivers = new const char*[m_indiserverCommand.size()+1];

      for(size_t i=0; i< m_indiserverCommand.size(); ++i)
      {
         drivers[i] = (m_indiserverCommand[i].data());
      }
      drivers[m_indiserverCommand.size()] = NULL;


      execvp("indiserver", (char * const*) drivers);

      log<software_error>({__FILE__, __LINE__, errno, "execvp returned"});
   
      delete[] drivers;
      
      return -1;
   }
   
   m_isSTDERR = filedes[0];
   m_isSTDERR_input = filedes[1];
   
   if(m_log.logLevel() <= logPrio::LOG_INFO)
   {
      std::string coml = "indiserver started with PID " + mx::ioutils::convertToString(m_isPID);   
      log<text_log>(coml);
   }
   
   return 0;
}

inline
void xindiserver::_isLogThreadStart( xindiserver * l)
{
   l->isLogThreadExec();
}

inline
int xindiserver::isLogThreadStart()
{
   try
   {
      m_isLogThread  = std::thread( _isLogThreadStart, this);
   }
   catch( const std::exception & e )
   {
      log<software_error>({__FILE__,__LINE__, std::string("Exception on I.S. log thread start: ") + e.what()});
      return -1;
   }
   catch( ... )
   {
      log<software_error>({__FILE__,__LINE__, "Unkown exception on I.S. log thread start"});
      return -1;
   }
   
   if(!m_isLogThread.joinable())
   {
      log<software_error>({__FILE__, __LINE__, "I.S. log thread did not start"});
      return -1;
   }
   
   sched_param sp;
   sp.sched_priority = m_isLogThreadPrio;

   int rv = pthread_setschedparam( m_isLogThread.native_handle(), SCHED_OTHER, &sp);
   
   if(rv != 0)
   {
      log<software_error>({__FILE__, __LINE__, rv, "Error setting thread params."});
      return -1;
   }
   
   return 0;

}


   
inline
void xindiserver::isLogThreadExec()
{
   char buffer[4096];

   std::string logs;
   while(m_shutdown == 0)
   {
      ssize_t count = read(m_isSTDERR, buffer, sizeof(buffer));
      if (count <= 0) 
      {
         continue;
      }
      else 
      {
         buffer[count] = '\0';
         
         logs += buffer;
         
         //Keep reading until \n found, then process.
         if(logs.back() == '\n')
         {
            size_t bol = 0;
            while(bol < logs.size())
            {
               size_t eol = logs.find('\n', bol);
               if(eol == std::string::npos) break;
               
               processISLog(logs.substr(bol, eol-bol));               
               bol = eol + 1;
            }
            logs = "";
         }
      }      
   }

}

inline
int xindiserver::processISLog( std::string logs )
{
   size_t st = 0;
   size_t ed;
   
   ed = logs.find(':', st);
   if(ed != std::string::npos) ed = logs.find(':', ed+1);
   if(ed != std::string::npos) ed = logs.find(':', ed+1);
   
   if(ed == std::string::npos)
   {
      log<software_error>({__FILE__, __LINE__, "Did not find timestamp : in log entry"});
      return -1;
   }
   
   std::string ts = logs.substr(st, ed-st);
   
   double dsec;

   tm bdt;   
   mx::ISO8601dateBreakdown(bdt.tm_year, bdt.tm_mon, bdt.tm_mday, bdt.tm_hour, bdt.tm_min, dsec, ts);
   
   bdt.tm_year -= 1900;
   bdt.tm_mon -= 1;
   bdt.tm_sec = (int) dsec;
   bdt.tm_isdst = 0;
   bdt.tm_gmtoff = 0;
   
   timespecX tsp;
   
   tsp.time_s = timegm(&bdt);
   tsp.time_ns = (nanosecT) ((dsec-bdt.tm_sec)*1e9 + 0.5);
    
   ++ed;
   st = logs.find_first_not_of(" ", ed);
   
   if(st == std::string::npos) st = ed;
   if(st == logs.size())
   {
      log<software_error>({__FILE__, __LINE__, "Did not find log entry."});
      return -1;
   }
      
   m_log.log<text_log>(tsp, "IS: " + logs.substr(st, logs.size()-st));

   return 0;
}

inline
int xindiserver::appStartup()
{
   
   if( constructIndiserverCommand(m_indiserverCommand) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   
   if( addLocalDrivers(m_indiserverCommand) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   
   if( addRemoteDrivers(m_indiserverCommand) < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
   

   
   //--------------------
   //Make symlinks
   //--------------------
   std::string path1 = "/opt/MagAOX/bin/xindidriver";
   for(size_t i=0; i<m_local.size(); ++i)
   {
      int rv = euidCalled();
      
      if(rv < 0)
      {
         log<software_error>({__FILE__, __LINE__, "Failed to create symlink for driver: " + m_local[i] + ". Continuing."});
      }
      
      std::cerr << "creating symlink " << path1 << " " << m_driverPath + m_local[i] << "\n";
      rv = symlink(path1.c_str(), (m_driverPath + m_local[i]).c_str());
      
      if(rv < 0 && errno != EEXIST)
      {
         log<software_error>({__FILE__, __LINE__, errno});
         log<software_error>({__FILE__, __LINE__, "Failed to create symlink for driver: " + m_local[i] + ". Continuing."});
      }
      
      rv = euidReal();
      if(rv < 0)
      {
         log<software_critical>({__FILE__, __LINE__, "Failed to reset privileges."});
         return -1;
      }
   }
   
   m_local.clear();
   m_remote.clear();
   m_tunnels.clear();
   
   //--------------------
   //Now start indiserver
   //--------------------
   if(forkIndiserver() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
      
   if(isLogThreadStart() < 0)
   {
      log<software_critical>({__FILE__, __LINE__});
      return -1;
   }  
   
   return 0;
}

inline
int xindiserver::appLogic()
{
   state(stateCodes::CONNECTED);
   
   return 0;
}

inline
int xindiserver::appShutdown()
{
   kill(m_isPID, SIGTERM);
   
   char w = '\0';
   ssize_t nwr = write(m_isSTDERR_input, &w, 1);
   if(nwr != 1)
   {
      log<software_error>({__FILE__, __LINE__, errno });
      log<software_error>({__FILE__, __LINE__, "Error on write to i.s. log thread. Sending SIGTERM."});
      pthread_kill(m_isLogThread.native_handle(), SIGTERM);
      
   }
      
   if(m_isLogThread.joinable()) m_isLogThread.join();
   return 0;
}


} //namespace app 
} //namespace MagAOX

#endif //xindiserver_hpp
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
