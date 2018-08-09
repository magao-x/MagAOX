/** \file xindiserver.hpp
  * \brief The MagAO-X INDI Server wrapper main program.
  *
  * \ingroup xindiserver_files
  */

#ifndef xindiserver_hpp
#define xindiserver_hpp

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <mx/app/application.hpp>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "magaox_git_version.h"


#include "tunneledHost.hpp"
#include "remoteDriver.hpp"

namespace MagAOX
{
namespace app
{
   
class xindiserver : public MagAOXApp
{

   //Give the test harness access.
   friend class xindiserver_test;

public:   
   
   typedef std::map< std::string, tunneledHost> hostMapT;
   
   typedef std::map< std::string, remoteDriver> rdriverMapT;
      
protected:

   std::string indiserver_l;
   int indiserver_m {-1};
   bool indiserver_n {false};
   int indiserver_p {-1};
   int indiserver_v {-1};
   bool indiserver_x {false};
   
   std::vector<std::string> m_local;
   std::vector<std::string> m_remote;

   std::vector<std::string> m_hosts;

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
   
   
   ///Validate the remote driver and remote hosts strings, and append them to the indi server command line arguments.
   /** Uses remoteDriver and tunneledHost to parse the remote driver and host specs, then
     * constructs the command line arguments and appends them to the driverArgs vector passed in.
     *
     * \returns 0 on success.
     * \returns -1 on error, either from failed validation or an exception in std::vector.
     */ 
   int addRemoteDrivers( std::vector<std::string> & driverArgs /**< [out] the vector of command line arguments for exec*/);
   
   
   /// Startup functions
   /** Setsup the INDI vars.
     * Checks if the device was found during loadConfig.
     */
   virtual int appStartup();

   /// Implementation of the FSM for the tripp lite PDU.
   virtual int appLogic();

   /// Do any needed shutdown tasks.  Currently nothing in this app.
   virtual int appShutdown();
   

};

xindiserver::xindiserver() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

void xindiserver::setupConfig()
{
   config.add("indiserver.l", "l", "", mx::argType::Required, "indiserver", "l", false,  "string", "indiserver: log messages to <d>/YYYY-MM-DD.islog, else stderr");
   config.add("indiserver.m", "m", "", mx::argType::Required, "indiserver", "m", false,  "int", "indiserver kill client if gets more than this many MB behind, default 50");
   config.add("indiserver.n", "n", "", mx::argType::True, "indiserver", "n", false,  "bool", "indiserver: ignore /tmp/noindi");
   config.add("indiserver.p", "p", "", mx::argType::Required, "indiserver", "p", false,  "int", "indiserver: alternate IP port, default 7624");
   config.add("indiserver.v", "v", "", mx::argType::Required, "indiserver", "v", false,  "int", "indiserver: loglevel 1/2/3 means -v, -vv or -vvv");
   config.add("indiserver.x", "x", "", mx::argType::True, "indiserver", "x", false,  "bool", "exit after last client disconnects -- FOR PROFILING ONLY");
   
   config.add("local.drivers","L", "local" , mx::argType::Required, "local", "drivers", false,  "vector string", "List of local drivers to start.");
   config.add("remote.drivers","R", "remote" , mx::argType::Required, "remote", "drivers", false,  "vector string", "List of remote drivers to start, in the form of name@hostname without the port.  Hostname needs an entry in ");
   config.add("remote.hosts", "H", "hosts", mx::argType::Required, "remote", "hosts", false,  "vector string", "List of remote hosts, in the form of hostname[:remote_port]:local_port. remote_port is optional if it is the INDI default.");
}

void xindiserver::loadConfig()
{
   //indiserver config:
   //-->deal with logdir
   config(indiserver_m, "indiserver.m");
   config(indiserver_n, "indiserver.n");
   config(indiserver_p, "indiserver.p");
   
   config(indiserver_v, "indiserver.v");
   
   config(indiserver_x, "indiserver.x");
   
   config(m_local, "local.drivers");
   config(m_remote, "remote.drivers");
   config(m_hosts, "remote.hosts");
}

int xindiserver::constructIndiserverCommand( std::vector<std::string> & indiserverCommand)
{
   try
   {
      indiserverCommand.push_back("indiserver");
   
      if(indiserver_l != "") 
      {
         indiserverCommand.push_back("-l");
         indiserverCommand.push_back(indiserver_l);
      }
      
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
      std::cerr << "Exception thrown by std::vector.\n";
      return -1;
   }
   
   return 0;
}

int xindiserver::addLocalDrivers( std::vector<std::string> & driverArgs )
{
   std::string driverPath = MAGAOX_path;
   driverPath += "/";
   driverPath += MAGAOX_driverRelPath;
   driverPath += "/";
   
   for(int i=0; i< m_local.size(); ++i)
   {
      size_t bad = m_local[i].find_first_of("@:/", 0);
      
      if(bad != std::string::npos)
      {
         std::cerr << "Local driver can't have host spec or path(@,:,/): " << m_local[i] << "\n";
         return -1;
      }
      
      try
      {
         m_local[i].insert(0, driverPath);
         driverArgs.push_back(m_local[i]);
      }
      catch(...)
      {
         std::cerr << "Exception in std::vector.\n";
         return -1;
      }
   }
}
 
int xindiserver::addRemoteDrivers( std::vector<std::string> & driverArgs )
{
   hostMapT hostMap;
   
   for(int i=0; i < m_hosts.size(); ++i)
   {
      tunneledHost th;
      
      int rv = th.parse( m_hosts[i] );
      
      if(rv < 0)
      {
         std::cerr << "Error parsing host specification: " << m_hosts[i] << "\n";
         std::cerr << "Exiting . . . \n";
         return -1;
      }
   
      std::pair<hostMapT::iterator, bool> res;
      try
      {
          res = hostMap.insert( hostMapT::value_type(th.remoteSpec(), th) );
      }
      catch(...)
      {
         std::cerr << "Exception thrown by map::insert.\n";
         return -1;
      }
      
      if(res.second != true)
      {
         std::cerr << "Duplicate host specification: " << th.fullSpec() << "( from " << m_hosts[i] << " )"  << "\n";
         std::cerr << "Exiting . . . \n";
         return -1;
      }
      
   }
      

   rdriverMapT rdriverMap;
   
   
   for(int i=0; i < m_remote.size(); ++i)
   {
      remoteDriver rd;
      
      int rv = rd.parse(m_remote[i]);
      
      if(rv < 0)
      {
         std::cerr << "Error parsing remote driver specification: " << m_remote[i] << "\n";
         std::cerr << "Exiting . . . \n";
         return -1;
      }
      
      std::pair<rdriverMapT::iterator, bool> res;
      try
      {
          res = rdriverMap.insert( rdriverMapT::value_type(rd.name(), rd) );
      }
      catch(...)
      {
         std::cerr << "Exception thrown by map::insert.\n";
         return -1;
      }
      
      if(res.second != true)
      {
         std::cerr << "Duplicate remote driver specification: " << rd.fullSpec() << "( from " << m_remote[i] << " )"  << "\n";
         std::cerr << "Exiting . . . \n";
         return -1;
      }
   }
         
   
   rdriverMapT::iterator rdit = rdriverMap.begin();
   for(;rdit!=rdriverMap.end(); ++rdit)
   {
      hostMapT::iterator hit;
      try
      {
          hit = hostMap.find( rdit->second.hostSpec() );
      }
      catch(...)
      {
         std::cerr << "Exception thrown by map::find.\n";
         return -1;
      }
      
      if(hit == hostMap.end())
      {
         std::cerr << "No host " << rdit->second.hostSpec() << " specified for driver " << rdit->second.fullSpec() << "\n";
         std::cerr << "Exiting . . . \n";
         return -1;
      }
      
      std::ostringstream oss;
      
      oss << rdit->second.name() << "@localhost:" << hit->second.localPort();
      
      try
      {
         driverArgs.push_back(oss.str());
      }
      catch(...)
      {
         std::cerr << "Exception thrown in vector::push_back.\n";
         return -1;
      }
   }
   
   return 0;

}

int xindiserver::appStartup()
{
   
   std::vector<std::string> indiserverCommand;
   
   if( constructIndiserverCommand(indiserverCommand) < 0)
   {
      std::cerr << "Exiting . . .\n";
      return -1;
   }
   
   
   if( addLocalDrivers(indiserverCommand) < 0)
   {
      std::cerr << "Exiting . . .\n";
      return -1;
   }
   
   if( addRemoteDrivers(indiserverCommand) < 0)
   {
      std::cerr << "Exiting . . .\n";
      return -1;
   }
   
   
   for(int i=0;i<indiserverCommand.size();++i)
   {
      std::cerr << indiserverCommand[i] << "\n";
   }
   

   std::cerr << "about to fork!\n";
   
   pid_t isPID = fork();
   
   if(isPID == 0)
   {
      const char ** drivers = new const char*[indiserverCommand.size()+1];

      for(int i=0; i< indiserverCommand.size(); ++i)
      {
         drivers[i] = (indiserverCommand[i].data());
      }
      drivers[indiserverCommand.size()] = NULL;


      execvp("indiserver", (char * const*) drivers);

      std::cerr << "xindiserver: indiserver exited\n";
   
      delete[] drivers;
   }
   
   return 0;
}

int xindiserver::appLogic()
{
   return 0;
}

int xindiserver::appShutdown()
{
   return 0;
}

} //namespace app 
} //namespace MagAOX

#endif //xindiserver_hpp
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
