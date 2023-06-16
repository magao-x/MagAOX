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

   /// Configuration
   static void
   _setupConfig(mx::app::appConfigurator & _config);
   static void
   _loadConfig(mx::app::appConfigurator & _config);

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
   static size_t m_time_offset;          // Resurrectee timeout

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

//Set default timeout to 600
template<class parentT>
size_t resurrectee<parentT>::m_time_offset = 600;

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

   if (m_time_offset < 1) { m_time_offset = 600; }

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

template<class parentT>
void resurrectee<parentT>::_setupConfig( mx::app::appConfigurator & _config )
{
   _config.add("resurrectee.timeout", "", "resurrectee.timeout", mx::app::argType::Required, "resurrectee", "timeout", false, "int", "Resurrectee timeout, s");
}

template<class parentT>
void resurrectee<parentT>::_loadConfig( mx::app::appConfigurator & _config )
{
   _config(m_time_offset, "resurrectee.timeout");
}

} //namespace app
} //namespace MagAOX
#endif//app_resurrectee_hpp