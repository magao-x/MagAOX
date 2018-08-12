
#include <iostream>
#include <string>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>

#ifndef XINDID_BUFFSIZE
#define XINDID_BUFFSIZE (1024)
#endif

#ifndef XINDID_FIFODIR
#define XINDID_FIFODIR "."
#endif

#ifndef XINDID_COMPILEDNAME
#define XINDID_COMPILEDNAME "xindidriver"
#endif

std::string myName;
bool timeToDie;

/// The details of one driver FIFO, this is passed to xoverThread to start the rd/wr thread for the indicated FIFO.
struct driverFIFO
{
   std::string fileName;    ///< the name of the driver FIFO
   int fd {-1};             ///< file desriptor of the diver FIFO after opened.  Note that this must be < 0 on entry into xoverThread or it interprets it as an error.
   int stdfd {STDIN_FILENO}; ///< one of this processes file descriptor denoting which direction this FIFO is, must be either STDIN_FILENO or STDOUT_FILENO.

   /// Constructor to initialize the fileName and stdfd members.
   driverFIFO( const std::string & fn, ///< [in] the fileName to set
               int sfd ///< [in] one of this processes file descriptor, must be either STDIN_FILENO or STDOUT_FILENO
             ) : fileName(fn), stdfd(sfd)
   {
   }
};


/// Work function for the FIFO read/write threads.
void * xoverThread( void * vdf /**< [in] pointer to a driverFIFO struct */)
{
   driverFIFO * df = static_cast<driverFIFO *>(vdf);

   //Make sure the driverFIFO was initialized properly.
   if(df->stdfd != STDIN_FILENO && df->stdfd != STDOUT_FILENO)
   {
      std::cerr << " (" << XINDID_COMPILEDNAME << "): stdfd must be either STDIN_FILENO or STDOUT_FILENO.\n";
      return nullptr;
   }

   if(df->fd >= 0)
   {
      std::cerr << " (" << XINDID_COMPILEDNAME << "): file descriptor for " << df->fileName << " already initialized.\n";
      return nullptr;
   }

   //Open the controller's FIFO
   while(df->fd < 0 && !timeToDie)
   {
      errno = 0;
      df->fd = open( df->fileName.c_str(), O_RDWR);

      if(df->fd < 0)
      {
         if( errno == ENOENT ) //If it's just cuz the file doesn't exist, we'll be patient.
         {
            sleep(1);
         }
         else
         {
            std::cerr << " (" << XINDID_COMPILEDNAME << "): failed to open " << df->fileName << ".\n";
            return nullptr;
         }
      }
   }

   //The buffer for reading from the FIFO.
   char rdbuff[XINDID_BUFFSIZE];

   //Setup the file descriptors.
   int fdRead, fdWrite;
   if(df->stdfd == STDIN_FILENO)
   {
      fdRead = df->stdfd;
      fdWrite = df->fd;
   }
   else // (df->stdfd == STDOUT_FILENO)
   {
      fdRead = df->fd;
      fdWrite = df->stdfd;
   }



   //Now loop until told to stop.
   while(!timeToDie)
   {
      rdbuff[0] = 0;

      // Create and clear out the FD set, set to watch the reader.
      fd_set fdsRead;
      FD_ZERO( &fdsRead );
      FD_SET( fdRead, &fdsRead );

      // Set the timeout on the select call.
      timeval tv;
      tv.tv_sec = 1;
      tv.tv_usec = 0;

      /// \todo This doesn't seem to be doing anything -- since we open O_RDWR this returns immediately. Remove?
      int nRetval = ::select( fdRead + 1, &fdsRead, NULL, NULL, &tv );

      if(timeToDie) break;

      if ( nRetval == -1 )
      {
         //Select returned some error.  Report it, and sleep for a bit.  Then try again.
         if ( timeToDie == false )
         {
            std::cerr << " (" << XINDID_COMPILEDNAME << "): " << std::strerror( errno);
            std::cerr << " in " << __FILE__ << " at " << __LINE__ << "\n";
            sleep(1);
         }
      }
      else if ( FD_ISSET( fdRead, &fdsRead ) != 0 )
      {
         //Input is available.

         int rd = 1; //Initialzed > 0 for while loop.

         while(rd > 0 && !timeToDie)
         {
            rd = read(fdRead, rdbuff, XINDID_BUFFSIZE);

            if(timeToDie) break; //Woke up from a blocking read due to a signal

            if( rd < 0 && !timeToDie)
            {
               //An error on the read.  Report it, and go back around.
               std::cerr << " (" << XINDID_COMPILEDNAME << "): " << std::strerror( errno);
               std::cerr << " in " << __FILE__ << " at " << __LINE__ << "\n";
               break;
            }

            //We write until we have written all that was read.
            int totwr = 0;

            while(totwr != rd && !timeToDie)
            {
               int wr = write(fdWrite, rdbuff + totwr, rd - totwr);

               if( wr < 0 && !timeToDie)
               {
                  //An error on write.  Report it and go back around.
                  std::cerr << " (" << XINDID_COMPILEDNAME << "): " << std::strerror( errno);
                  std::cerr << " in " << __FILE__ << " at " << __LINE__ << "\n";
                  break;
               }

               totwr += wr;
            }
         }
      }
   }

   close(df->fd);

   return 0;
}

void sigHandler( int signum,
                 siginfo_t *siginf,
                 void *ucont
               )
{
   timeToDie = true;
}

int setSigTermHandler()
{
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = sigHandler;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGTERM, &act, 0) < 0 )
   {
      std::cerr << " (" << XINDID_COMPILEDNAME << "): error setting SIGTERM handler: " << strerror(errno) << "\n";
      return -1;
   }

   errno = 0;
   if( sigaction(SIGQUIT, &act, 0) < 0 )
   {
      std::cerr << " (" << XINDID_COMPILEDNAME << "): error setting SIGQUIT handler: " << strerror(errno) << "\n";
      return -1;
   }

   errno = 0;
   if( sigaction(SIGINT, &act, 0) < 0 )
   {
      std::cerr << " (" << XINDID_COMPILEDNAME << "): error setting SIGINT handler: " << strerror(errno) << "\n";
      return -1;
   }

   return 0;
}

void usage()
{
   std::cerr << "\n" << XINDID_COMPILEDNAME << ":\t";
   std::cerr << "an INDI driver passthrough to a device controller\n\t\twhich is the actual INDI driver. ";
   std::cerr << "See xindidriver.md for\n\t\tcomplete documentation.\n\n";
   std::cerr << "usage: linked_name [-h, --help]\n";
   std::cerr << "         - or -   \n";
   std::cerr << "       xindidriver drivername\n";
   std::cerr << "where linked_name is a symlink to xindidriver\n\n";
   std::cerr << "options:\n";
   std::cerr << "         -h,--help   print this message and exit.\n\n";

}

int main( int argc, char **argv)
{

   timeToDie = false;

   if (argc == 1 )
   {
      myName = basename( argv[0] );

      if(myName == XINDID_COMPILEDNAME)
      {
         std::cerr << XINDID_COMPILEDNAME << ": must call from linked path or with argument for driver name.\n\n";
         usage();
         return -1;
      }
   }
   else if (argc == 2)
   {
      myName = argv[1];

      if(myName == "-h" || myName == "--help")
      {
         usage();
         return 0;
      }
   }
   else
   {
      std::cerr << " (" << XINDID_COMPILEDNAME << "): too many arguments.\n\n";
      usage();
      return -1;
   }

   //Now that myName is known, install signal handler
   if( setSigTermHandler() < 0) return -1;

   std::string stdinFifo = std::string(XINDID_FIFODIR) + "/" + myName + ".in";
   std::string stdoutFifo = std::string(XINDID_FIFODIR) + "/" + myName + ".out";

   std::cerr << " (" << XINDID_COMPILEDNAME << "): starting with " << stdinFifo << " & " << stdoutFifo << std::endl;

   driverFIFO dfIn (stdinFifo, STDIN_FILENO);

   driverFIFO dfOut (stdoutFifo, STDOUT_FILENO);

   //Launch the read/write threads, one each for STDIN and STDOUT
   pthread_t stdIn_th = 0;
   pthread_create( &stdIn_th, NULL, xoverThread, &dfIn );

   pthread_t stdOut_th = 0;
   pthread_create( &stdOut_th, NULL, xoverThread, &dfOut );

   //Now loop until killed.
   int rv;
   while( !timeToDie )
   {
      rv = pthread_tryjoin_np(stdIn_th, 0);

      if(rv == 0)
      {
         std::cerr << " (" << XINDID_COMPILEDNAME << "): STDIN thread exited.\n";
         timeToDie = true;
      }

      rv = pthread_tryjoin_np(stdOut_th, 0);

      if(rv == 0)
      {
         std::cerr << " (" << XINDID_COMPILEDNAME << "): STDOUT thread exited.\n";
         timeToDie = true;
      }

      if(!timeToDie) sleep(1);
   }

   //Wake up each thread:
   pthread_kill(stdIn_th, SIGTERM);
   pthread_kill(stdOut_th, SIGTERM);

   //Give each thread a chance to cleanup.
   pthread_join(stdIn_th, 0);
   pthread_join(stdOut_th, 0);

   return 0;

}
