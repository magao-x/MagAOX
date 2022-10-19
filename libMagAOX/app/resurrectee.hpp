/** \file resurrectee.hpp
  * \brief MagAO-X class wrapper to send hexbeats to resurrector
  * \author Brian T. Carcich (BrianTCarcich@gmail.com)
  *
  * History:
  * - 2022-10-17 created by TTC
  * 
  * \ingroup app_files
  */

#ifndef app_resurrectee_hpp
#define app_resurrectee_hpp

/** Build:
  *
  *     TBD
  *
  * Usage:
  *
  *     TBD
  *
  * This class will be instantiated in a process forked by resurrector,
  * and will perform several primary tasks:
  *
  * 1) Exit with logging on receipt of a SIGUSR2 signal
  * 2) Parse TOML config arguments to get the process name
  * 2) Open the named FIFO to resurrector
  * 4) Write a hexbeat (Note i) to a named FIFO (Note ii)
  *
  * Notes
  *
  * i) heartbeat representing some integer seconds in the future
  * ii) /.../<parentT->m_hexbeatName>.hb
  */
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>

#include "MagAOXApp.hpp"

namespace MagAOX
{
namespace app
{

template<class _parentT>
class resurrectee
{
public:

   ///The parent MagAOX app.
   typedef _parentT parentT;

protected:

   ///This object's parent class
   parentT * m_parent {nullptr};

public:
   /// Public c'tor
   /** Opens the FIFOs specified by parent.
     * If this fails, then m_good is set to false.
     * test this with good().
     */
   resurrectee( parentT * parent);
   ~resurrectee();

   bool good(){ return m_good;}

   virtual void execute(void);

private:

   static resurrectee * m_self; ///< Static pointer to this (set in constructor).  Used to test whether a a MagAOXApp is already instatiated (a fatal error) and used for getting out of static signal handlers.

   /// Flag to hold the status of this connection.
   bool m_good {true};

   // Save some static information e.g. to be logged on exit
   const char s_unknown[8]={"unknown"};  // default value
   std::string m_myname{(char*) s_unknown}; // Name after -n parsed
   int m_broken_pipes{0};                // Count of successive SIGPIPEs
   int m_broken_pipes_limit{2};          // Last SIGPIPE to log an error
   int m_mypid{-1};                      // PID of this process
   int m_fdhb{-1};                       // File Descriptor of named FIFO
   size_t m_time_offset{600};            // Resurrectee timeout

   /// Signal handler:  exit on any signal caught
   static void
   _sigusr2_handler(int sig, siginfo_t *si, void *unused);
   void
   sigusr2_handler(int sig
                  , siginfo_t *si __attribute__((unused))
                  , void *unused __attribute__((unused))
                  )
   {
       std::cerr
       << "Driver[" << m_myname << "]:  "
       << "PID=" << m_mypid
       << "; caught and exiting on [" << strsignal(sig) << "]"
       << "\n"
       ;
       exit(0);
   }

   /// Ignore some signals, establish handlers for others
   void setup_handle_SIGs()
   {
       int istat = -1;


       // Ignore SIGPIPE on bad write so we can handle it inline
       // Ignore SIGINT so ^C will stop parent process (resurrector) only
       for (auto isig : {SIGPIPE, SIGINT}) {
          struct sigaction oldsa;
          sigemptyset(&oldsa.sa_mask);
          oldsa.sa_flags = 0;
          oldsa.sa_handler = SIG_IGN;
          istat = sigaction(isig, NULL, &oldsa);

          if (!istat && SIG_IGN == oldsa.sa_handler) { continue; }

          struct sigaction sa;
          sigemptyset(&sa.sa_mask);
          sa.sa_flags = 0;
          sa.sa_handler = SIG_IGN;
          istat = sigaction(isig, &sa, NULL);
       }

       // Catch SIGUSR2 in sigusr2_handler(...) above
       struct sigaction sa;
       sigemptyset(&sa.sa_mask);
       sa.sa_flags = SA_SIGINFO;
       sa.sa_handler = nullptr;
       sa.sa_sigaction = _sigusr2_handler;
       errno = 0;
       for (auto isig : {SIGUSR2}) {
           istat = sigaction(isig, &sa, 0);
           if (istat < 0) {
               std::cerr
               << "Driver[" << m_myname << "]:  "
               << "sigaction("
               << strsignal(isig)
               << ")=" << istat
               << "; errno=" << errno
               << "[" << strerror(errno)
               << "]\n";
               perror("# sigaction/SIGPIPE");
               exit(1);
           }
       }
   }

}; // class resurrectee

//Set self pointer to null so app starts up uninitialized.
template<class parentT>
resurrectee<parentT> * resurrectee<parentT>::m_self = nullptr;

/// Write hexbeat timestamp to FIFO
template<class parentT>
void resurrectee<parentT>::execute()
{
    // Generate hexbeat timestamp
    char stimestamp[18];
    sprintf(stimestamp,"%9.9lx\n",time(0)+m_time_offset);

    // Write hexbeat to FIFO
    int irtn = write(m_fdhb,stimestamp,10);

    // On success, return
    if (irtn > 0)
    {
        // If previous read(s) had an error, then log recovery and reset
        if(m_broken_pipes)
        {
            std::cerr << "Driver[" << m_myname
                     << "]:  "
                     << "recovered\n";
            m_broken_pipes = 0;
        }
        return;
    }

    // Ignore successive errors after some number of them
    if (m_broken_pipes>m_broken_pipes_limit) { return; }

    // Log first three successive errors
    ++m_broken_pipes;
    char* pNL = strchr(stimestamp,'\n');
    if (pNL && (pNL-stimestamp)<10) { strcpy(pNL,"\\n"); }
    std::cerr << "Driver[" << m_myname
              << "]:  " << irtn
              << "=write(" << m_fdhb
              << ",[" << stimestamp
              << "],10); errno=" << errno
              << "[" << strerror(errno)
              << "]\n";
    errno = 0;
}

template<class parentT>
resurrectee<parentT>::resurrectee(parentT * parent)
{
   if( m_self != nullptr )
   {
      std::cerr << "Attempt to instantiate 2nd resurrectee.  Exiting immediately.\n";
      exit(-1);
   }
   
   m_self = this;

   // Save parent, name and PID to member attributes
   m_parent = parent;
   m_myname = m_parent->configName();
   m_mypid = getpid();

   // Set up signal handling
   setup_handle_SIGs();

   // Open resurrector FIFO for writing
   std::string fifoname{m_parent->resurrecteeFifoName()};
   int istat{-1};
   int fdrd{-1};
   int fdhb{-1};
   try {
      fdrd = open(fifoname.c_str(), O_RDONLY | O_NONBLOCK);
      if (fdrd < 0 && errno == ENOENT) {
         // File does not exist: create FIFO; open read-only
         mode_t prev_mode;
         errno = 0;
         prev_mode = umask(0);
         istat = mkfifo(fifoname.c_str(), S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP);
         prev_mode = umask(prev_mode);
         if (istat < 0) { throw 0; }
         fdrd = open(fifoname.c_str(), O_RDONLY | O_NONBLOCK);
      }
      if (fdrd < 0) { throw 0; }

      //  Ensure opened file is a FIFO
      struct stat fdstat;
      istat = fstat(fdrd,&fdstat);
      if (istat < 0) { throw 0; }
      if (!S_ISFIFO(fdstat.st_mode)) { errno = EEXIST; throw 0; }

      // Open hexbeat FIFO write-only option
      // - This will not block because fdrd is open for read
      fdhb = open(fifoname.c_str(),O_WRONLY);
      if (fdhb < 0) { throw 0; }

      istat = close(fdrd);
      if (istat < 0) { throw 0; }

      m_fdhb = fdhb;
   }
   catch (...)
   {
      int icleanup = errno;
      m_fdhb = -1;
      m_good = false;
      close(fdrd);
      close(fdhb);
      errno = icleanup;
   }
}

template<class parentT>
resurrectee<parentT>::~resurrectee()
{

   if (m_fdhb > -1)
   {
      int icleanup = errno;
      close(m_fdhb);
      m_fdhb = -1;
      errno  = icleanup;
   }
   resurrectee<parentT>::m_self = nullptr;
}

// Static SIGUSR2 handler; calls lone instance handler via m_self
template<class parentT>
void resurrectee<parentT>::_sigusr2_handler( int signum
                                           , siginfo_t *siginf
                                           , void *ucont
                                           )
{
   m_self->sigusr2_handler(signum,siginf,ucont);
}

} //namespace app
} //namespace MagAOX
#endif//app_resurrectee_hpp
#if 0
/** \file indiDriver.hpp
  * \brief MagAO-X INDI Driver Wrapper
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-05-26 created by JRM
  * 
  * \ingroup app_files
  */

#ifndef app_indiDriver_hpp
#define app_indiDriver_hpp

#include "../../INDI/libcommon/IndiDriver.hpp"
#include "../../INDI/libcommon/IndiElement.hpp"

#include "../../INDI/libcommon/IndiClient.hpp"

#include "MagAOXApp.hpp"

namespace MagAOX
{
namespace app
{

///Simple INDI Client class
class indiClient : public pcf::IndiClient
{

public:

   /// Constructor, which establishes the INDI client connection.
   indiClient( const std::string & clientName,
               const std::string & hostAddress,
               const int hostPort
             ) : pcf::IndiClient( clientName, "none", "1.7", hostAddress, hostPort)
   {
   }
   
   
   /// Implementation of the pcf::IndiClient interface, called by activate to actually begins the INDI event loop.
   /** This is necessary to detect server restarts.
     */
   void execute()
   {
      processIndiRequests(false);
   }
   
};   
   
template<class _parentT>
class indiDriver : public pcf::IndiDriver
{
public:

   ///The parent MagAOX app.
   typedef _parentT parentT;

protected:

   ///This objects parent class
   parentT * m_parent {nullptr};

   ///An INDI Client is used to send commands to other drivers.
   indiClient * m_outGoing {nullptr};

   ///The IP address of the server for the INDI Client connection
   std::string m_serverIPAddress {"127.0.01"};

   ///The port of the server for the INDI Client connection
   int m_serverPort {7624};

private:

   /// Flag to hold the status of this connection.
   bool m_good {true};

public:

   /// Public c'tor
   /** Call pcf::IndiDriver c'tor, and then opens the FIFOs specified
     * by parent.  If this fails, then m_good is set to false.
     * test this with good().
     */
   indiDriver( parentT * parent,
               const std::string &szName,
               const std::string &szDriverVersion,
               const std::string &szProtocolVersion
             );

   /// D'tor, deletes the IndiClient pointer.
   ~indiDriver();

   /// Get the value of the good flag.
   /**
     * \returns the value of m_good, true or false.
     */
   bool good(){ return m_good;}

   // override callbacks
   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );

   virtual void handleGetProperties( const pcf::IndiProperty &ipRecv );

   virtual void handleNewProperty( const pcf::IndiProperty &ipRecv );

   virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

   /// Define the execute virtual function.  This runs the processIndiRequests function in this thread, and does not return.
   virtual void execute(void);

   /// Define the update virt. func. here so the uptime message isn't sent
   virtual void update();

   /// Send a newProperty command to another INDI driver
   /** Uses the IndiClient member of this class, which is initialized the first time if necessary.
     *
     * \returns 0 on success
     * \returns -1 on any errors (which are logged).
     */
   virtual int sendNewProperty( const pcf::IndiProperty &ipRecv );

};

template<class parentT>
indiDriver<parentT>::indiDriver ( parentT * parent,
                                  const std::string &szName,
                                  const std::string &szDriverVersion,
                                  const std::string &szProtocolVersion
                                ) : pcf::IndiDriver(szName, szDriverVersion, szProtocolVersion)
{
   m_parent = parent;

   int fd;

   errno = 0;
   fd = open( parent->driverInName().c_str(), O_RDWR);
   if(fd < 0)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, errno, "Error opening input INDI FIFO."});
      m_good = false;
      return;
   }
   setInputFd(fd);

   errno = 0;
   fd = open( parent->driverOutName().c_str(), O_RDWR);
   if(fd < 0)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, errno, "Error opening output INDI FIFO."});
      m_good = false;
      return;
   }
   setOutputFd(fd);
   
   // Open the ctrl fifo and write a single byte to it to trigger a restart
   // of the xindidriver process.
   // This allows indiserver to refresh everything.
   errno = 0;
   fd = open( parent->driverCtrlName().c_str(), O_RDWR);
   if(fd < 0)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, errno, "Error opening control INDI FIFO."});
      m_good = false;
      return;
   }
   char c = 0;
   int wrno = write(fd, &c, 1);
   if(wrno < 0)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, errno, "Error writing to control INDI FIFO."});
      m_good = false;
   }
   
   
   close(fd);
}

template<class parentT>
indiDriver<parentT>::~indiDriver()
{
   if(m_outGoing) delete m_outGoing;

}
template<class parentT>
void indiDriver<parentT>::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
   if(m_parent) m_parent->handleDefProperty(ipRecv);
}

template<class parentT>
void indiDriver<parentT>::handleGetProperties( const pcf::IndiProperty &ipRecv )
{
   if(m_parent) m_parent->handleGetProperties(ipRecv);
}

template<class parentT>
void indiDriver<parentT>::handleNewProperty( const pcf::IndiProperty &ipRecv )
{
   if(m_parent) m_parent->handleNewProperty(ipRecv);
}

template<class parentT>
void indiDriver<parentT>::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
   if(m_parent) m_parent->handleSetProperty(ipRecv);
}

template<class parentT>
void indiDriver<parentT>::execute()
{
   processIndiRequests(false);
}

template<class parentT>
void  indiDriver<parentT>::update()
{
   return;
}

template<class parentT>
int  indiDriver<parentT>::sendNewProperty( const pcf::IndiProperty &ipRecv )
{
   //If there is an existing client, check if it has exited.
   if( m_outGoing != nullptr)
   {
      if(m_outGoing->getQuitProcess())
      {
         parentT::template log<logger::text_log>("INDI client disconnected.");
         m_outGoing->quitProcess();
         m_outGoing->deactivate();
         delete m_outGoing;
         m_outGoing = nullptr;
      }
   }
   
   //Connect if needed
   if( m_outGoing == nullptr)
   {
      try
      {
         m_outGoing = new indiClient(m_parent->configName()+"-client", m_serverIPAddress, m_serverPort);
         m_outGoing->activate();
      }
      catch(...)
      {
         parentT::template log<logger::software_error>({__FILE__, __LINE__, "Exception thrown while creating IndiClient connection"});
         return -1;
      }
      
      if(m_outGoing == nullptr)
      {
         parentT::template log<logger::software_error>({__FILE__, __LINE__, "Failed to allocate IndiClient connection"});
         return -1;
      }
      
      parentT::template log<logger::text_log>("INDI client connected and activated");
   }
   
   try
   {
      m_outGoing->sendNewProperty(ipRecv);
      if(m_outGoing->getQuitProcess())
      {
         parentT::template log<logger::software_error>({__FILE__, __LINE__, "INDI client appears to be disconnected -- NEW not sent."});
         return -1;
      }
      
      //m_outGoing->quitProcess();
      //delete m_outGoing;
      //m_outGoing = nullptr;
      return 0;
   }
   catch(std::exception & e)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, std::string("Exception from IndiClient: ") + e.what()});
      return -1;
   }
   catch(...)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, "Exception from IndiClient"});
      return -1;
   }

   //Should never get here, but we are exiting for some reason sometimes.
   parentT::template log<logger::software_error>({__FILE__, __LINE__, "fall through in sendNewProperty"});
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
#endif//0
