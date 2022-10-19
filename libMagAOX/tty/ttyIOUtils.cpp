/** \file ttyIOUtils.cpp
  * \brief Utilities for i/o on a file descriptor pointing to a tty device.
  * \author Jared R. Males (jaredmales@gmail.com)
  * 
  * \ingroup tty_files
  */

#include "ttyIOUtils.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>

#include <mx/sys/timeUtils.hpp>

#include "ttyErrors.hpp"



namespace MagAOX
{
namespace tty
{

int telnetCRLF( std::string & telnetStr,     // [out] the string with \\r an \\n converted to \\r\\n
                const std::string & inputStr // [in] the string to be converted
              )
{
   telnetStr.resize(inputStr.size()); 
      
   size_t N = inputStr.size();
   size_t j = 0;
   for(size_t i=0;i<N; ++i)
   {
      if(inputStr[i] != '\r' && inputStr[i] != '\n')
      {
         telnetStr[j] = inputStr[i];
      }
      else if(inputStr[i] == '\r')
      {
         telnetStr[j] = '\r';

         if(i < N-1)
         {
            if(inputStr[i+1] == '\n')
            {
               ++j;
               telnetStr[j] = '\n';
               ++i;
               ++j; //i is incremented on continue, but j is not
               continue;
            }
         }
         telnetStr.push_back(' ');
         ++j;
         telnetStr[j] = '\n';
      }
      else if(inputStr[i] == '\n')
      {
         telnetStr[j] = '\r';
         telnetStr.push_back(' ');
         ++j;
         telnetStr[j] = '\n';
      }
      ++j;
   }
   
   return 0;
}

int ttyOpenRaw( int & fileDescrip,        // [out] the file descriptor.  Set to 0 on an error.
                std::string & deviceName, // [in] the device path name, e.g. /dev/ttyUSB0
                speed_t speed             // [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
              )
{
   errno = 0;

   fileDescrip = ::open( deviceName.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);


   struct termios termopt;
   if( tcgetattr(fileDescrip, &termopt) < 0 )
   {
      close(fileDescrip);
      fileDescrip = 0;
      return TTY_E_TCGETATTR;
   }

   if( cfsetispeed(&termopt, speed) < 0 )
   {
      close(fileDescrip);
      fileDescrip = 0;
      return TTY_E_SETISPEED;
   }

   if( cfsetospeed(&termopt, speed) < 0 )
   {
      close(fileDescrip);
      fileDescrip = 0;
      return TTY_E_SETOSPEED;
   }

   cfmakeraw(&termopt);

   if( tcsetattr(fileDescrip, TCSANOW, &termopt) < 0 )
   {
      close(fileDescrip);
      fileDescrip = 0;
      return TTY_E_TCSETATTR;
   }

   return TTY_E_NOERROR;
}

bool isEndOfTrans( const std::string & strRead, // [in] The read buffer to check
                   const std::string & eot      // [in] The end-of-transmission string
                 )
{
   //If buffRead isn't long enough yet.
   if(eot.size() > strRead.size()) return false;

   //Now check from back, if any don't match it's false.
   for(size_t i=0; i < eot.size(); ++i)
   {
      if( strRead[strRead.size()-1-i] != eot[eot.size()-1-i] ) return false;
   }

   return true;
}

int ttyWrite( const std::string & buffWrite, // [in] The characters to write to the tty.
              int fd,                        // [in] The file descriptor of the open tty.
              int timeoutWrite               // [in] The timeout in milliseconds.
            )
{
   double t0;
   struct pollfd pfd;

   errno = 0;
   pfd.fd = fd;
   pfd.events = POLLOUT;

   t0 = mx::sys::get_curr_time();

   size_t totWritten = 0;
   while( totWritten < buffWrite.size())
   {
      int timeoutCurrent = timeoutWrite - (mx::sys::get_curr_time()-t0)*1000;
      if(timeoutCurrent < 0) return TTY_E_TIMEOUTONWRITE;

      int rv = poll( &pfd, 1, timeoutCurrent);
      if( rv == 0 ) return TTY_E_TIMEOUTONWRITEPOLL;
      else if( rv < 0 ) return TTY_E_ERRORONWRITEPOLL;

      rv = write(fd, buffWrite.c_str()+totWritten, buffWrite.size()-totWritten);
      if(rv < 0) return TTY_E_ERRORONWRITE;

      //sleep(1);
      #ifdef TTY_DEBUG
      std::cerr << "Wrote " << rv << " chars of " << buffWrite.size() << "\n";
      #endif

      totWritten += rv;

      if( ( mx::sys::get_curr_time()-t0)*1000 > timeoutWrite ) return TTY_E_TIMEOUTONWRITE;
   }

   return TTY_E_NOERROR;
}

int ttyReadRaw( std::vector<unsigned char> & vecRead, // [out] The buffer in which to store the output.
                int & readBytes,                      // [out] The number of bytes read.
                int fd,                               // [in] The file descriptor of the open tty.
                int timeoutRead                       // [in] The timeout in milliseconds.
              )
{
   int rv;

   struct pollfd pfd;

   errno = 0;

   pfd.fd = fd;
   pfd.events = POLLIN;


   rv = poll( &pfd, 1, timeoutRead);
   if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
   if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

   readBytes = 0;
   
   rv = read(fd, vecRead.data(), vecRead.size());
   if( rv < 0 ) return TTY_E_ERRORONREAD;


   readBytes = rv;
   

   return TTY_E_NOERROR;


}

int ttyRead( std::string & strRead,   // [out] The string in which to store the output.
             int bytes,               // [in] the number of bytes to read
             int fd,                  // [in] The file descriptor of the open tty.
             int timeoutRead          // [in] The timeout in milliseconds.
           )
{
   int rv;
   int timeoutCurrent;
   double t0;

   struct pollfd pfd;

   errno = 0;

   pfd.fd = fd;
   pfd.events = POLLIN;

   strRead.clear();
   char buffRead[TTY_BUFFSIZE];

   //Start timeout clock for reading.
   t0 = mx::sys::get_curr_time();
   timeoutCurrent = timeoutRead;

   //Now read the response up to the eot.
   strRead.clear();

   rv = poll( &pfd, 1, timeoutCurrent);
   if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
   if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

   rv = read(fd, buffRead, TTY_BUFFSIZE);
   if( rv < 0 ) return TTY_E_ERRORONREAD;

   strRead.append( buffRead, rv);

   int totBytes = rv;
   
   while( totBytes < bytes )
   {
      timeoutCurrent = timeoutRead - (mx::sys::get_curr_time()-t0)*1000;
      if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

      rv = poll( &pfd, 1, timeoutCurrent);
      if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
      if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

      rv = read(fd, buffRead, TTY_BUFFSIZE);
      if( rv < 0 ) return TTY_E_ERRORONREAD;
      buffRead[rv] ='\0';

      strRead.append( buffRead, rv);
      totBytes += rv;
      
      #ifdef TTY_DEBUG
      std::cerr << "ttyRead: read " << rv << " bytes. buffRead=" << buffRead << "\n";
      #endif
   }


   return TTY_E_NOERROR;


}

int ttyRead( std::string & strRead,   // [out] The string in which to store the output.
             const std::string & eot, // [in] A sequence of characters which indicates the end of transmission.
             int fd,                  // [in] The file descriptor of the open tty.
             int timeoutRead          // [in] The timeout in milliseconds.
           )
{
   int rv;
   int timeoutCurrent;
   double t0;

   struct pollfd pfd;

   errno = 0;

   pfd.fd = fd;
   pfd.events = POLLIN;

   strRead.clear();
   char buffRead[TTY_BUFFSIZE];

   //Start timeout clock for reading.
   t0 = mx::sys::get_curr_time();
   timeoutCurrent = timeoutRead;

   //Now read the response up to the eot.
   strRead.clear();

   rv = poll( &pfd, 1, timeoutCurrent);
   if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
   if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

   rv = read(fd, buffRead, TTY_BUFFSIZE);
   if( rv < 0 ) return TTY_E_ERRORONREAD;

   strRead.append( buffRead, rv);

   while( !isEndOfTrans(strRead, eot) )
   {
      timeoutCurrent = timeoutRead - (mx::sys::get_curr_time()-t0)*1000;
      if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

      rv = poll( &pfd, 1, timeoutCurrent);
      if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
      if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

      rv = read(fd, buffRead, TTY_BUFFSIZE);
      if( rv < 0 ) return TTY_E_ERRORONREAD;
      buffRead[rv] ='\0';

      strRead.append( buffRead, rv);
      #ifdef TTY_DEBUG
      std::cerr << "ttyRead: read " << rv << " bytes. buffRead=" << buffRead << "\n";
      #endif
   }


   return TTY_E_NOERROR;


}

int ttyWriteRead( std::string & strRead,        // [out] The string in which to store the output.
                  const std::string & strWrite, // [in] The characters to write to the tty.
                  const std::string & eot,      // [in] A sequence of characters which indicates the end of transmission.
                  bool swallowEcho,             // [in] If true, strWrite.size() characters are read after the write
                  int fd,                       // [in] The file descriptor of the open tty.
                  int timeoutWrite,             // [in] The write timeout in milliseconds.
                  int timeoutRead               // [in] The read timeout in milliseconds.
                )
{
   strRead.clear();

   int rv;

   //Write First
   rv = ttyWrite( strWrite, fd, timeoutWrite);
   if(rv != TTY_E_NOERROR) return rv;



   //Now read response from console
   int timeoutCurrent;
   double t0;


   //Start timeout clock for reading.
   t0 = mx::sys::get_curr_time();;

   if(swallowEcho)
   {
      struct pollfd pfd;
      pfd.fd = fd;
      pfd.events = POLLIN;
   
      size_t totrv = 0;
      char buffRead[TTY_BUFFSIZE];

      //First swallow the echo.
      while( totrv <= strWrite.size() )
      {
         timeoutCurrent = timeoutRead - (mx::sys::get_curr_time()-t0)*1000;
         if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

         rv = poll( &pfd, 1, timeoutCurrent);
         if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
         if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

         rv = read(fd, buffRead, TTY_BUFFSIZE);
         if( rv < 0 ) return TTY_E_ERRORONREAD;

         totrv += rv;
      }
   }

   timeoutCurrent = timeoutRead - (mx::sys::get_curr_time()-t0)*1000;
   if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

   //Now read the response up to the eot.
   return ttyRead(strRead, eot, fd, timeoutCurrent);
}



} //namespace tty
} //namespace MagAOX

