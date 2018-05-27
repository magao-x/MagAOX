/// $Id: Daemon.cpp,v 1.5 2007/09/20 18:06:48 pgrenz Exp $
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include "Daemon.hpp"
#include "Logger.hpp"
#include "Config.hpp"

using std::string;
using pcf::Daemon;
using pcf::Logger;
using pcf::Config;

////////////////////////////////////////////////////////////////////////////////

bool pcf::Daemon::sm_oQuitProcess = false;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor - initialize the internal timeval struct.

Daemon::Daemon()
{
  m_szProgName = "a.out";
  m_szVersion = "0.0";
  m_szLogFileName = "a.out.log";
  m_szConfigFileName = "a.out.conf";
  m_szOptions = "f:l:h?";
}

////////////////////////////////////////////////////////////////////////////////
/// standard constructor from program information
/// @param szProgName The name of the program for display.
/// @param szVersion The version of the program.
/// @param szConfigFileName The name of the config file.

Daemon::Daemon( const std::string &szProgName,
                const std::string &szVersion )
{
  m_szProgName = szProgName;
  m_szVersion = szVersion;
  m_szConfigFileName = szProgName + ".conf";
  m_szLogFileName = szProgName + ".log";
  m_szOptions = "f:l:h?";
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

Daemon::~Daemon()
{
  //  nothing to do.
}

////////////////////////////////////////////////////////////////////////////////
/// This variable and function are used to trap a 'SIGINT' to tell the program
/// to shutdown gracefully.

void Daemon::processSignalHandler( int nSignal )
{
  Logger msgInfo;
  msgInfo << Logger::enumInfo << "Caught signal #" << nSignal
          << ". Initiating an orderly shutdown." << std::endl;

  sm_oQuitProcess = true;
}

////////////////////////////////////////////////////////////////////////////////
/// The help message outputted to stderr.

void Daemon::displayHelp()
{
  std::cout << "'" << m_szProgName << "' (v" << m_szVersion
            << ") Provides an daemon for performing a task."
            << std::endl;
  std::cout << "The arguments are:" << std::endl;
  std::cout << "f    the full path to the config file (default: "
            << m_szConfigFileName << ")." << std::endl;
  std::cout << "h    show this message." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Will return 0 if we were successful, any other value if we were not.

bool Daemon::initialize( int argc, char **argv, const bool &oFork )
{
  // We need the current working directory as a default.
  char pcBuf[1024];
  memset( pcBuf, 0, 1024 );
  char *pcPath = getcwd( pcBuf, 1023 );

  int nErr = 0;
  bool oIsParent = true;
  bool oShowHelp = false;
  std::string szConfigFile( std::string( pcPath ) + "/" + m_szProgName + ".conf" );
  std::string szLogFile( std::string( pcPath ) + "/" + m_szProgName );

  // setup the signal handler.
  signal( SIGHUP, Daemon::processSignalHandler );
  signal( SIGINT, Daemon::processSignalHandler );
  signal( SIGTERM, Daemon::processSignalHandler );

  // Get our options.
  int nOption = getopt( argc, argv, m_szOptions.c_str() );
  while ( nOption != -1 )
  {
    switch ( nOption )
    {
      case 'f':
        szConfigFile = std::string( optarg );
        break;
      case 'l':
        szLogFile = std::string( optarg );
        break;
      case 'h': // fall-thru
      case '?': // fall-thru
      default:
        oShowHelp = true;
        break;
    }
    nOption = getopt( argc, argv, m_szOptions.c_str() );
  }

  if ( oShowHelp == true )
  {
    displayHelp();
  }
  else
  {
    // assume success.
    oIsParent = false;
    int nRetVal = 0;
    pid_t pid = -1;
    pid_t sid = -1;

    // fork the parent process - if it is bad, we failed to fork.
    if ( oFork == true )
      pid = fork();
    else
      pid = 0;

    if ( pid < 0 )
    {
      nRetVal = pid;
    }
    // if our pid is good, we can exit the parent.
    else if ( pid > 0 )
    {
      // we are the parent
      nRetVal = 0;
      oIsParent = true;
    }
    // pid = 0 - we are the new child.
    else
    {
      oIsParent = false;

      // change the file usage mask.
      umask( 0 );

      // make sure we can create a core file
      // this is not catastrophic if we can't.
      rlimit rlim;
      rlim.rlim_cur = rlim.rlim_max = RLIM_INFINITY;
      if ( setrlimit( RLIMIT_CORE, &rlim ) < 0 )
      {
        nErr = errno;
        std::cerr << "ERROR setrlimit - " << strerror( nErr ) << std::endl;
      }

      //  get a new session id.
      sid = setsid();

      // make the current working directory one that will always be there.
      chdir( "/" );

      // close the provided standard file descriptors.
      //close( STDIN_FILENO );
      //close( STDOUT_FILENO );
      //close( STDERR_FILENO );

      // Save the file names.
      m_szConfigFileName = szConfigFile;
      m_szLogFileName = szLogFile;

      // setup the logging mechanism.
      string szLPath;
      string szLFile = m_szLogFileName;
      size_t tLPos = m_szLogFileName.rfind( '/' );
      if ( tLPos != string::npos )
      {
        szLPath = m_szLogFileName.substr( 0, tLPos );
        szLFile = m_szLogFileName.substr( tLPos+1, m_szLogFileName.length() );
      }
      Logger::init( szLPath, szLFile );

      // write the version to the log.
      Logger msgInfo;
      if ( oFork == true )
      {
        msgInfo << Logger::enumInfo << "starting '" << getProgName()
                << "' (v " << getVersion() << " <forked>)...." << std::endl;
      }
      else
      {
        msgInfo << Logger::enumInfo << "starting '" << getProgName()
                << "' (v " << getVersion() << " <unforked>)...." << std::endl;
      }

      // reading the config file.
      string szCPath;
      string szCFile = m_szConfigFileName;
      size_t tCPos = m_szConfigFileName.rfind( '/' );
      if ( tCPos != string::npos )
      {
        szCPath = m_szConfigFileName.substr( 0, tCPos );
        szCFile = m_szConfigFileName.substr( tCPos+1, m_szConfigFileName.length() );
      }
      Config::init( szCPath, szCFile );
    }
  }

  // We only want to continue if we are the child.
  return !oIsParent;
}

////////////////////////////////////////////////////////////////////////////////
