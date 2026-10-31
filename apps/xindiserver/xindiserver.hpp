/** \file xindiserver.hpp
  * \brief The MagAO-X INDI Server wrapper header.
  *
  * \ingroup xindiserver_files
  */

#ifndef xindiserver_hpp
#define xindiserver_hpp

#include <sys/prctl.h>
#include <sys/wait.h>

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
  * <a href="../handbook/operating/software/apps/network.html#xindiserver">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup xindiserver_files xindiserver Files
  * \ingroup xindiserver
  */

//Forward decl for test harness
namespace libXWCTest
{
namespace xindiserverTest
{
    struct xindiserver_test;
}
}

namespace MagAOX
{
namespace app
{

#define SSHTUNNEL_E_NOTUNNELS (-10)

/// Structure to hold an sshTunnel specification, used for created command line args for indiserver
struct sshTunnel
{
   std::string m_remoteHost;
   int m_remotePort {0};
   int m_localPort {0};
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
         int remotePort = 0;
         int localPort = 0;
         bool compress = false;

         config.configUnused( remoteHost, mx::app::iniFile::makeKey(sections[i], "remoteHost" ) );
         config.configUnused( remotePort, mx::app::iniFile::makeKey(sections[i], "remotePort" ) );
         config.configUnused( localPort, mx::app::iniFile::makeKey(sections[i], "localPort" ) );
         config.configUnused( compress, mx::app::iniFile::makeKey(sections[i], "compress" ) );

         tmap[sections[i]] = sshTunnel({remoteHost, remotePort, localPort});

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
#define XINDISERVER_E_BADSERVERSPEC (-110)


/** The INDI Server wrapper application class.
  *
  * \ingroup xindiserver
  *
  */
class xindiserver : public MagAOXApp<false>
{

   //Give the test harness access.
   friend class libXWCTest::xindiserverTest::xindiserver_test;

protected:

   int indiserver_d {-1};  ///< The indiserver stream BLOB drop MB behind setting (passed to indiserver)
   int indiserver_m {-1};  ///< The indiserver MB behind setting (passed to indiserver)
   bool indiserver_n {false}; ///< The indiserver ignore /tmp/noindi flag (passed to indiserver)
   int indiserver_p {-1}; ///< The indiserver port (passed to indiserver)
   int indiserver_v {-1}; ///< The indiserver verbosity (passed to indiserver)
   bool indiserver_x {false}; ///< The indiserver terminate after last exit flag (passed to indiserver)

   std::string m_driverFIFOPath; ///< The path to the local drivers' FIFOs directory

   std::string m_driverPath; ///< The path to the local drivers
   std::vector<std::string> m_local; ///< List of local drivers passed in by config
   std::vector<std::string> m_remote; ///< List of remote drivers, using tunnels, passed in by config
   std::vector<std::string> m_driversAtHosts; ///< List of remote drivers, using driver@host[:port], passed in by config
   std::unordered_set<std::string> m_driverNames; ///< List of driver names processed for command line, used to prevent duplication.

   std::vector<std::string> m_remoteServers; ///< List of other INDI server config files to read remote drivers from.

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


   /// Validate the remote driver entries, and append them to the indi server command line arguments.
   /** Parses the remote driver specs, then
     * constructs the command line arguments and appends them to the driverArgs vector passed in.
     *
     * \returns 0 on success.
     * \returns -1 on error, either from failed validation or an exception in std::vector.
     */
   int addRemoteDrivers( std::vector<std::string> & driverArgs /**< [out] the vector of command line arguments for exec*/);

   ///Validate the remote server entries, read the associated config files for local drivers, and append them to the indi server command line arguments as remote ddrivers.
   /** Parses the remote server specs, then reads the remote server config files, and then
     * constructs the command line arguments and appends them to the driverArgs vector passed in.
     *
     * \returns 0 on success.
     * \returns -1 on error, either from failed validation or an exception in std::vector.
     */
   int addRemoteServers( std::vector<std::string> & driverArgs /**< [out] the vector of command line arguments for exec*/);

   ///Forks and exec's the indiserver process with the command constructed from local, remote, and hosts.
   /** Also saves the PID and stderr pipe file descriptors for log capture.
     *
     * \returns 0 on success
     * \returns -1 on error (fatal)
     */
   int initINDIServer();

   ///Thread starter, called by isLogThreadStart on thread construction.  Calls isLogThreadExec.
   static void _isLogThreadStart( xindiserver * l /**< [in] a pointer to a xindiserver instance (normally this) */);

   /// Start the log capture.
   int isLogThreadStart();

   /// Execute the log capture.
   void isLogThreadExec();

   /// Process a log entry from indiserver, putting it into MagAO-X standard form
   /** The entry is timestamped with the time the line is received. The timestamp created by
     * indiserver is kept in the message.
     * 
     * \returns 0 on success
     */
   int processISLog( std::string logs /**< [in] log entry from indiserver */ );

   /// Add data read from indiserver to the pending text, and move out every complete line
   /** The data is appended by count, so a zero byte does not truncate it. Only an incomplete last line is kept in
     * pending, to be completed by the next read.
     */
   static void extractISLines( std::string &pending,           /**< [in/out] text received but not yet processed */
                               const char *data,               /**< [in] the data just read */
                               size_t count,                   /**< [in] the number of bytes in data */
                               std::vector<std::string> &lines /**< [out] complete lines appended */
   );

   /// Get the log priority for a line from indiserver
   /** Fatal errors from xindidriver ("failed to lock") and indiserver ("bind: Address already in use") are
     * critical, everything else is informational.
     */
   static logPrioT isLogPriority( const std::string &line /**< [in] the line from indiserver */ );

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
   m_loopPause = 100000000;//Default 0.1s; indiRun has 1s select timeout

   return;
}

inline
void xindiserver::setupConfig()
{
   config.add("indiserver.d", "d", "", argType::Required, "indiserver", "d", false,  "int", "indiserver drops streaming BLOBs to a client more than this many MB behind, default 5, 0 to disable");
   config.add("indiserver.m", "m", "", argType::Required, "indiserver", "m", false,  "int", "indiserver kills client if it gets more than this many MB behind, default 50");
   config.add("indiserver.N", "N", "", argType::True, "indiserver", "N", false,  "bool", "indiserver: ignore /tmp/noindi.  Capitalized to avoid conflict with --name");
   config.add("indiserver.p", "p", "", argType::Required, "indiserver", "p", false,  "int", "indiserver: alternate IP port, default 7624");
   config.add("indiserver.v", "v", "", argType::True, "indiserver", "v", false,  "int", "indiserver: log verbosity, -v, -vv or -vvv");
   config.add("indiserver.x", "x", "", argType::True, "indiserver", "x", false,  "bool", "exit after last client disconnects -- FOR PROFILING ONLY");

   config.add("local.drivers","L", "local.drivers" , argType::Required, "local", "drivers", false,  "vector string", "List of local drivers to start.");
   config.add("remote.drivers","R", "remote.drivers" , argType::Required, "remote", "drivers", false,  "vector string", "List of remote drivers to start, in the form of name@tunnel, where tunnel is the name of a tunnel specified in sshTunnels.conf.");
   config.add("remote.drivers@hosts","R", "remote.drivers@hosts" , argType::Required, "remote", "drivers@hosts", false,  "vector string", "List of remote drivers to start, in the form of name@host[:port], where host is the name of a host where another INDI server is running.");

   config.add("remote.servers","", "remote.servers" , argType::Required, "remote", "servers", false,  "vector string", "List of servers to load remote drivers for, in the form of name@tunnel.  Name is used to load the name.conf configuration file, and tunnel is the name of a tunnel specified in sshTunnels.conf.");

}



inline
void xindiserver::loadConfig()
{
   //indiserver config:
   config(indiserver_d, "indiserver.d");
   config(indiserver_m, "indiserver.m");
   config(indiserver_n, "indiserver.N");
   config(indiserver_p, "indiserver.p");

   //From command-line:
   indiserver_v = config.verbosity("indiserver.v");

   if (!indiserver_v) {
      //If there were no -v options on the command line, then
      //check if v=1[,2[,3[...]]] was in the config file
      std::vector<std::string> is_vs;
      config(is_vs,"indiserver.v");
      indiserver_v = is_vs.size();
   }

   config(indiserver_x, "indiserver.x");

   config(m_local, "local.drivers");
   config(m_remote, "remote.drivers");
   config(m_driversAtHosts, "remote.drivers@hosts");
   config(m_remoteServers, "remote.servers");

   if(loadSSHTunnelConfigs(m_tunnels, config) < 0)
   {
      m_shutdown = true;
      return;
   }

   if( constructIndiserverCommand(m_indiserverCommand) < 0)
   {
      log<software_critical>();
      m_shutdown = true;
      return;
   }

   if( addLocalDrivers(m_indiserverCommand) < 0)
   {
      log<software_critical>();
      m_shutdown = true;
      return;
   }

   if( addRemoteDrivers(m_indiserverCommand) < 0)
   {
      log<software_critical>();
      m_shutdown = true;
      return;
   }

   if( addRemoteServers(m_indiserverCommand) < 0)
   {
      log<software_critical>();
      m_shutdown = true;
      return;
   }


   m_local.clear();
   m_remote.clear();
   m_tunnels.clear();

}

inline
int xindiserver::constructIndiserverCommand( std::vector<std::string> & indiserverCommand)
{
   try
   {
      indiserverCommand.push_back("indiserver");

      // The indiserver stream BLOB drop MB behind setting (passed to indiserver)
      if(indiserver_d >= 0)
      {
         indiserverCommand.push_back("-d");
         indiserverCommand.push_back(std::format("{}", indiserver_d));
      }

      // The indiserver MB behind setting (passed to indiserver)
      if(indiserver_m > 0)
      {
         indiserverCommand.push_back("-m");
         indiserverCommand.push_back(std::format("{}", indiserver_m));
      }

      // The indiserver ignore /tmp/noindi flag
      if(indiserver_n == true) indiserverCommand.push_back("-n");

      //< The indiserver port
      if(indiserver_p > 0)
      {
         indiserverCommand.push_back("-p");
         indiserverCommand.push_back(std::format("{}", indiserver_p));
      }

      // The indiserver verbosity
      if(indiserver_v == 1) indiserverCommand.push_back("-v");

      if(indiserver_v == 2) indiserverCommand.push_back("-vv");

      if(indiserver_v >= 3) indiserverCommand.push_back("-vvv");

      // The indiserver terminate after last exit flag
      if(indiserver_x == true) indiserverCommand.push_back("-x");

      // The local drivers' FIFO directory
      m_driverFIFOPath = MAGAOX_path;
      m_driverFIFOPath += "/";
      m_driverFIFOPath += MAGAOX_driverFIFORelPath;
      indiserverCommand.push_back("-F");
      indiserverCommand.push_back(m_driverFIFOPath);
   }
   catch(...)
   {
      log<software_critical>({"Exception thrown by std::vector."});
      return -1;
   }

   return 0;
}

inline
int xindiserver::addLocalDrivers( std::vector<std::string> & driverArgs )
{
   // Loop through local driver names from INDI server configuration
   // TOML file element
   //
   //   local.drivers = driver_name[, ...]

   for(size_t i=0; i< m_local.size(); ++i)
   {
      if(m_local[i].size() == 0 || std::all_of(m_local[i].begin(),m_local[i].end(), isspace))
      {
          log<text_log>("empty local driver", logPrio::LOG_WARNING);
          continue;
      }

      size_t bad = m_local[i].find_first_of("@:/", 0);

      if(bad != std::string::npos)
      {
         log<software_critical>("Local driver can't have host spec or path(@,:,/): " + m_local[i]);

         return XINDISERVER_E_BADDRIVERSPEC;
      }

      if( m_driverNames.count(m_local[i]) > 0)
      {
         log<software_critical>("Duplicate driver name: " + m_local[i]);
         return XINDISERVER_E_DUPLICATEDRIVER;
      }

      m_driverNames.insert(m_local[i]);

      std::string dname = m_local[i];

      try
      {
         driverArgs.push_back(dname);
      }
      catch(...)
      {
         log<software_critical>("Exception thrown by std::vector");
         return XINDISERVER_E_VECTOREXCEPT;
      }
   }

   return 0;
}

inline
int xindiserver::addRemoteDrivers( std::vector<std::string> & driverArgs )
{
   // Loop through remote driver@host names from INDI server
   // configuration TOML file element
   //
   //   remote.drivers@hosts = [driver_name]@hostname[:port_#][, ...]

   for(size_t i=0; i < m_driversAtHosts.size(); ++i)
   {
      std::string driver;
      std::string host_port;

      size_t p = m_driversAtHosts[i].find('@');

      if(p == 0 || p == std::string::npos)
      {
         log<software_critical>("Error parsing remote driver@host[:port] specification: " + m_driversAtHosts[i]);
         return XINDISERVER_E_BADDRIVERSPEC;
      }

      // if @ is first character, then use @hostname as driver name
      driver = p ? m_driversAtHosts[i].substr(0, p) : m_driversAtHosts[i];
      host_port = m_driversAtHosts[i].substr(p+1);

      if( m_driverNames.count(driver) > 0)
      {
         log<software_critical>("Duplicate driver name: " + driver);
         return XINDISERVER_E_DUPLICATEDRIVER;
      }

      m_driverNames.insert(driver);

      try
      {
         driverArgs.push_back(m_driversAtHosts[i]);
      }
      catch(...)
      {
         log<software_critical>("Exception thrown by vector::push_back.");
         return XINDISERVER_E_VECTOREXCEPT;
      }
   }

   // Loop through remote driver@tunnel names from INDI server
   // configuration TOML file element
   //
   //   remote.drivers = driver_name@tunnel[, ...]

   for(size_t i=0; i < m_remote.size(); ++i)
   {
      if(m_remote[i].size() == 0 || std::all_of(m_remote[i].begin(),m_remote[i].end(), isspace))
      {
          log<text_log>("empty remote driver", logPrio::LOG_WARNING);
          continue;
      }

      std::string driver;
      std::string tunnel;

      size_t p = m_remote[i].find('@');

      if(p == 0 || p == std::string::npos)
      {
         log<software_critical>("Error parsing remote driver specification: " + m_remote[i]);
         return XINDISERVER_E_BADDRIVERSPEC;
      }

      // if @ is first character, then use "" as driver name for now
      driver = p ? m_remote[i].substr(0, p) : std::string("");
      tunnel = m_remote[i].substr(p+1);

      // Don't allow duplicate driver name;
      // if @ is first character, then don't check for duplicate driver
      // name here, and check @hostname later
      if( p && (m_driverNames.count(driver) > 0))
      {
         log<software_critical>("Duplicate driver name: " + driver);
         return XINDISERVER_E_DUPLICATEDRIVER;
      }

      std::ostringstream oss;

      if(m_tunnels.size() == 0)
      {
         log<software_critical>("No tunnels specified.");
         return XINDISERVER_E_NOTUNNELS;
      }

      if(m_tunnels.count(tunnel) != 1)
      {
         log<software_critical>("Tunnel not found for: " + m_remote[i]);
         return XINDISERVER_E_TUNNELNOTFOUND;
      }

      if (m_tunnels[tunnel].m_localPort > -1)
      {
          // Non-negative local port value:  connect to local SSH tunnel (sshDigger) at that port
          // N.B. driver may be empty string "" if @ was first character
          //      in m_remote[i]
          oss << driver << "@localhost:" << m_tunnels[tunnel].m_localPort;
      }
      else
      {
          // Found "localPort = -1" (or similar) for this tunnel in sshTunnels.conf
          // Negative local port value:  connect directly to remote host and remote port
          // N.B. driver may be empty string "" if @ was first character
          //      in m_remote[i]
          oss << driver << "@" << m_tunnels[tunnel].m_remoteHost << ":" << m_tunnels[tunnel].m_remotePort;
      }

      // If @ is first character in m_remote[i], then don't allow
      // duplicate @hostname
      if( !p && (m_driverNames.count(oss.str()) > 0))
      {
         log<software_critical>("Duplicate @hostname name: " + oss.str());
         return XINDISERVER_E_DUPLICATEDRIVER;
      }

      // Insert non-duplicate (driver name or @hostname) into set to
      // prevent any duplicates later
      m_driverNames.insert(p ? driver : oss.str());

      try
      {
         driverArgs.push_back(oss.str());
      }
      catch(...)
      {
         log<software_critical>("Exception thrown by vector::push_back.");
         return XINDISERVER_E_VECTOREXCEPT;
      }
   }

   return 0;

}

inline
int xindiserver::addRemoteServers( std::vector<std::string> & driverArgs )
{

   // Loop through remote driver names from INDI server configuration
   // TOML file element
   //
   //   remote.servers = server_name@tunnel[, ...]

   for(size_t j=0; j < m_remoteServers.size(); ++j)
   {
      std::string server;
      std::string tunnel;

      size_t p = m_remoteServers[j].find('@');

      if(p == 0 || p == std::string::npos)
      {
         log<software_critical>("Error parsing remote server specification: " + m_remote[j] );
         return XINDISERVER_E_BADSERVERSPEC;
      }

      server = m_remoteServers[j].substr(0, p);
      tunnel = m_remoteServers[j].substr(p+1);

      if(m_tunnels.size() == 0)
      {
         log<software_critical>("No tunnels specified.");
         return XINDISERVER_E_NOTUNNELS;
      }

      if(m_tunnels.count(tunnel) != 1)
      {
         log<software_critical>("Tunnel not found for: " + m_remote[j]);
         return XINDISERVER_E_TUNNELNOTFOUND;
      }

      //Now we create a local app configurator, and read the other server's config file
      mx::app::appConfigurator rsconfig;

      rsconfig.add("local.drivers", "", "" , argType::Required, "local", "drivers", false,  "", "");

      std::string rsconfigPath = m_configDir + "/" + server + ".conf";

      rsconfig.readConfig(rsconfigPath);

      std::vector<std::string> local;

      rsconfig(local, "local.drivers");

      // Loop through local driver names from remote INDI server
      // configuration TOML file (server_name.conf) element
      //
      //   local.drivers = driver_name[, ...]

      for(size_t i=0; i < local.size(); ++i)
      {
         size_t bad = local[i].find_first_of("@:/", 0);

         if(bad != std::string::npos)
         {
            log<software_critical>("Remote server's Local driver can't have host spec or path(@,:,/): " + local[i]);

            return XINDISERVER_E_BADDRIVERSPEC;
         }

         if( m_driverNames.count(local[i]) > 0)
         {
            log<software_critical>("Duplicate driver name from remote server: " + local[i]);
            return XINDISERVER_E_DUPLICATEDRIVER;
         }

         m_driverNames.insert(local[i]);

         std::ostringstream oss;

         if (m_tunnels[tunnel].m_localPort > -1)
         {
            // Non-negative local port value:  connect to local SSH tunnel (sshDigger) at that port
            oss << local[i] << "@localhost:" << m_tunnels[tunnel].m_localPort;
         }
         else
         {
            // Found "localPort = -1" (or similar) for this tunnel in sshTunnels.conf
            // Negative local port value:  connect directly to remote host and remote port
            oss << local[i] << "@" << m_tunnels[tunnel].m_remoteHost << ":" << m_tunnels[tunnel].m_remotePort;
         }


         try
         {
            driverArgs.push_back(oss.str());
         }
         catch(...)
         {
            log<software_critical>("Exception thrown by vector::push_back.");
            return XINDISERVER_E_VECTOREXCEPT;
         }
      }

   }

   return 0;
}

inline
int xindiserver::initINDIServer()
{

   if(m_log.logLevel() >= logPrio::LOG_INFO)
   {
      std::string coml = "Initializing indiserver with command:";
      for(size_t i=0;i<m_indiserverCommand.size();++i)
      {
         coml += " ";
         coml += m_indiserverCommand[i];
      }

      log<text_log>(coml);
      std::cerr << coml << std::endl;
   }

   // Create pipe pair to route fprintf(stderr,...) in indiserver.c to
   // logging via intermediate logging thread isLogThreadExec
   int filedes[2];
   if (pipe(filedes) == -1)
   {
      log<software_error>(errno);
      return -1;
   }

   pid_t parentPID = getpid();

   m_isPID = fork();

   if(m_isPID < 0)
   {
      log<software_error>({errno, "fork failed"});
      return -1;
   }

   if(m_isPID == 0)
   {
      //Exit if xindiserver dies, even by SIGKILL, so a restarted xindiserver can bind the INDI port.
      if(prctl(PR_SET_PDEATHSIG, SIGTERM) < 0 || getppid() != parentPID)
      {
         _exit(EXIT_FAILURE);
      }

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

      log<software_error>({errno, "execvp returned"});

      delete[] drivers;

      _exit(EXIT_FAILURE);
   }

   // Save pipe for logging thread; see isLogThreadExec()
   m_isSTDERR = filedes[0];
   m_isSTDERR_input = filedes[1];

   if(isLogThreadStart() < 0)
   {
      log<software_critical>();
      return -1;
   }

   if(m_log.logLevel() <= logPrio::LOG_INFO)
   {
      log<text_log>(std::format("indiserver started with PID {}", m_isPID));
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
      log<software_error>(std::format("Exception on I.S. log thread start: {}", e.what()));
      return -1;
   }
   catch( ... )
   {
      log<software_error>("Unkown exception on I.S. log thread start");
      return -1;
   }

   if(!m_isLogThread.joinable())
   {
      log<software_error>("I.S. log thread did not start");
      return -1;
   }

   sched_param sp;
   sp.sched_priority = m_isLogThreadPrio;

   int rv = pthread_setschedparam( m_isLogThread.native_handle(), SCHED_OTHER, &sp);

   if(rv != 0)
   {
      log<software_error>({rv, "Error setting thread params."});
      return -1;
   }

   return 0;

}



inline
void xindiserver::isLogThreadExec()
{
   char buffer[4097];

   std::string logs;
   while(m_shutdown == 0)
   {
      ssize_t count = read(m_isSTDERR, buffer, sizeof(buffer)-1); //Make sure we always have room for \0
      if (count <= 0 || m_shutdown == 1)
      {
         continue;
      }
      else if(count > (ssize_t) sizeof(buffer)-1)
      {
         log<software_error>("read returned too many bytes.");
	 continue;
      }
      else
      {
         // Process every complete line as soon as it arrives, keeping only an incomplete tail for the next read,
         // so that each line is timestamped and logged on the read which completes it.
         std::vector<std::string> lines;
         extractISLines(logs, buffer, count, lines);

         for(const std::string &line : lines)
         {
            processISLog(line);
         }
      }
   }

}

inline
void xindiserver::extractISLines( std::string &pending, const char *data, size_t count, std::vector<std::string> &lines )
{
   pending.append(data, count);

   size_t bol = 0;
   size_t eol;
   while( (eol = pending.find('\n', bol)) != std::string::npos )
   {
      lines.push_back(pending.substr(bol, eol-bol));
      bol = eol + 1;
   }

   pending.erase(0, bol); //keep only the incomplete tail, if any
}

inline
logPrioT xindiserver::isLogPriority( const std::string &line )
{
   if(line.find("xindidriver") != std::string::npos) //Errors from xindidriver
   {
      if(line.find("failed to lock") != std::string::npos)
      {
         return logPrio::LOG_CRITICAL;
      }
   }
   else if(line.find("bind: Address already in use") != std::string::npos) //Errors from indiserver
   {
      return logPrio::LOG_CRITICAL;
   }
   else if(line.find("WARNING") != std::string::npos) //Warnings from indiserver
   {
      return logPrio::LOG_WARNING;
   }

   return logPrio::LOG_INFO;
}

inline
int xindiserver::processISLog( std::string logs )
{
   // Timestamped with the time the line is received. The time indiserver writes at the start of the line is kept
   // in the text but not parsed: lines relayed from drivers can arrive without one, and parsing whatever text was
   // there instead produced wrong times (year 0, stored as 2041).
   logPrioT prio = isLogPriority(logs);

   log<text_log>("IS: " + logs, prio);

   if(prio == logPrio::LOG_CRITICAL)
   {
      state(stateCodes::FAILURE);
      m_shutdown = true;
   }

   return 0;
}

inline
int xindiserver::appStartup()
{   
   //--------------------
   //Now start indiserver, including logging threads
   //--------------------
   if(initINDIServer() < 0)
   {
      log<software_critical>();
      return -1;
   }

   //Start of logging thread moved from here to initINDIserver 2024-SEP

   return 0;
}

inline
int xindiserver::appLogic()
{
   int status;
   pid_t result = waitpid(m_isPID, &status, WNOHANG);
   if (result == 0)
   {
      state(stateCodes::CONNECTED);
   }
   else
   {
      //We don't care why.  If indiserver is not alive while in this function then it's a fatal error.
      log<text_log>("indiserver has exited", logPrio::LOG_CRITICAL);
      state(stateCodes::FAILURE);
      return -1;
   }

   return 0;
}

inline
int xindiserver::appShutdown()
{
   if(m_isPID > 0)
   {
      kill(m_isPID, SIGTERM);
   }

   // Wake up logging thread by sending it data
   if(m_isSTDERR_input >= 0)
   {
      char w = '\0';
      ssize_t nwr = write(m_isSTDERR_input, &w, 1);
      if(nwr != 1)
      {
         log<software_error>({errno, "Error on write to i.s. log thread. Sending SIGTERM."});
         pthread_kill(m_isLogThread.native_handle(), SIGTERM);

      }
   }

   if(m_isLogThread.joinable()) { m_isLogThread.join(); }
   else { log<text_log>("indiserver is not joinable"); }
   return 0;
}


} //namespace app
} //namespace MagAOX

#endif //xindiserver_hpp
