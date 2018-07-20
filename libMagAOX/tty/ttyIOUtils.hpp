/** \file ttyIOUtils.hpp
  * \brief Utilities for i/o on a file descriptor pointing to a tty device.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-01-15 created by JRM, starting with code imported from VisAO
  */

#ifndef tty_ttyIOUtils_hpp
#define tty_ttyIOUtils_hpp

//#include <string.h>

#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>

#include <string>

#include <mx/timeUtils.hpp>

#include "ttyErrors.hpp"

#ifndef TTY_BUFFSIZE
   #define TTY_BUFFSIZE (1024)
#endif

namespace MagAOX
{
namespace tty
{

/// Open a file as a raw-mode tty device
/**
  * \returns TTY_E_NOERROR on success.
  * \returns TTY_E_TCGETATTR on a error from tcgetattr.
  * \returns TTY_E_TCSETATTR on an error from tcsetattr.
  * \returns TTY_E_SETISPEED on a cfsetispeed error.
  * \returns TTY_E_SETOSPEED on a cfsetospeed error.
  *
  */
int ttyOpenRaw( int & fileDescrip,        ///< [out] the file descriptor.  Set to 0 on an error.
                std::string & deviceName, ///< [in] the device path name, e.g. /dev/ttyUSB0
                speed_t speed             ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
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

/// Check if the end of the buffer contains the end-of-transmission string
/**
  * \returns true if the last N chars of buffRead are equal to eot, where N is the length of eot.
  * \returns false otherwise.
  */
inline
bool isEndOfTrans( const std::string & strRead, ///< [in] The read buffer to check
                   const std::string & eot      ///< [in] The end-of-transmission string
                 )
{
   //If buffRead isn't long enough yet.
   if(eot.size() > strRead.size()) return false;

   //Now check from back, if any don't match it's false.
   for(int i=0; i < eot.size(); ++i)
   {
      if( strRead[strRead.size()-1-i] != eot[eot.size()-1-i] ) return false;
   }

   return true;
}


/// Write to the tty console indicated by a file descriptor.
/**
  *
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONWRITEPOLL if the poll times out.
  * \returns TTY_E_ERRORONWRITEPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONWRITE if a timeout occurs during the write.
  * \returns TTY_E_ERRORONWRITE if an error occurs writing to the file.
  */
inline
int ttyWrite( const std::string & buffWrite, ///< [in] The characters to write to the tty.
              int fd,                        ///< [in] The file descriptor of the open tty.
              int timeoutWrite               ///< [in] The timeout in milliseconds.
            )
{
   double t0;
   struct pollfd pfd;

   errno = 0;
   pfd.fd = fd;
   pfd.events = POLLOUT;

   t0 = mx::get_curr_time();

   int totWritten = 0;
   while( totWritten < buffWrite.size())
   {
      int timeoutCurrent = timeoutWrite - (mx::get_curr_time()-t0)*1000;
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

      if( ( mx::get_curr_time()-t0)*1000 > timeoutWrite ) return TTY_E_TIMEOUTONWRITE;
   }

   return TTY_E_NOERROR;
}

/// Read from a tty console indicated by a file-descriptor, until an end of transmission string is read.
/**
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
  * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
  * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
  */
inline
int ttyRead( std::string & strRead,   ///< [out] The string in which to store the output.
             const std::string & eot, ///< [in] A sequence of characters which indicates the end of transmission.
             int fd,                  ///< [in] The file descriptor of the open tty.
             int timeoutRead          ///< [in] The timeout in milliseconds.
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
   t0 = mx::get_curr_time();
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
      timeoutCurrent = timeoutRead - (mx::get_curr_time()-t0)*1000;
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

/// Write to a tty on an open file descriptor, then get the result.
/** The read is conducted until an end-of-transmission string is received.
  * Echo characters are swallowed if desired.
  *
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONWRITEPOLL if the poll times out.
  * \returns TTY_E_ERRORONWRITEPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONWRITE if a timeout occurs during the write.
  * \returns TTY_E_ERRORONWRITE if an error occurs writing to the file.
  * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
  * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
  * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
  */
inline
int ttyWriteRead( std::string & strRead,        ///< [out] The string in which to store the output.
                  const std::string & strWrite, ///< [in] The characters to write to the tty.
                  const std::string & eot,      ///< [in] A sequence of characters which indicates the end of transmission.
                  bool swallowEcho,             ///< [in] If true, strWrite.size() characters are read after the write
                  int fd,                       ///< [in] The file descriptor of the open tty.
                  int timeoutWrite,             ///< [in] The write timeout in milliseconds.
                  int timeoutRead               ///< [in] The read timeout in milliseconds.
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

   struct pollfd pfd;
   pfd.fd = fd;
   pfd.events = POLLIN;


   //Start timeout clock for reading.
   t0 = mx::get_curr_time();;

   if(swallowEcho)
   {
      int totrv = 0;
      char buffRead[TTY_BUFFSIZE];

      //First swallow the echo.
      while( totrv <= strWrite.size() )
      {
         timeoutCurrent = timeoutRead - (mx::get_curr_time()-t0)*1000;
         if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

         rv = poll( &pfd, 1, timeoutCurrent);
         if( rv == 0 ) return TTY_E_TIMEOUTONREADPOLL;
         if( rv < 0 ) return TTY_E_ERRORONREADPOLL;

         rv = read(fd, buffRead, TTY_BUFFSIZE);
         if( rv < 0 ) return TTY_E_ERRORONREAD;

         totrv += rv;
      }
   }

   timeoutCurrent = timeoutRead - (mx::get_curr_time()-t0)*1000;
   if(timeoutCurrent < 0) return TTY_E_TIMEOUTONREAD;

   //Now read the response up to the eot.
   return ttyRead(strRead, eot, fd, timeoutCurrent);
}



} //namespace tty
} //namespace MagAOX

#endif //tty_ttyIOUtils_hpp
