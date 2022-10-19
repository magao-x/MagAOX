/** \file ttyIOUtils.hpp
  * \brief Utilities for i/o on a file descriptor pointing to a tty device.
  * \author Jared R. Males (jaredmales@gmail.com)
  * 
  * \ingroup tty_files
  * History:
  * - 2018-01-15 created by JRM, starting with code imported from VisAO
  */

#ifndef tty_ttyIOUtils_hpp
#define tty_ttyIOUtils_hpp

#include <string>
#include <vector>

#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>


#ifndef TTY_BUFFSIZE
   #define TTY_BUFFSIZE (1024)
#endif

namespace MagAOX
{
namespace tty
{

/// Replace lone \\r and \\n with \\r\\n for telnet-ness.
/** Do it all at once instead of during char-by-char transmission
  * cuz some devices (I'm looking at you SDG) get impatient and 
  * stop paying attention.
  * 
  * \returns 0 on success
  * \returns -1 on error (nothing yet)
  * 
  */
int telnetCRLF( std::string & telnetStr,     ///< [out] the string with \\r an \\n converted to \\r\\n
                const std::string & inputStr ///< [in] the string to be converted
              );
  
/// Open a file as a raw-mode tty device
/**
  * \returns TTY_E_NOERROR on success.
  * \returns TTY_E_TCGETATTR on a error from tcgetattr.
  * \returns TTY_E_TCSETATTR on an error from tcsetattr.
  * \returns TTY_E_SETISPEED on a cfsetispeed error.
  * \returns TTY_E_SETOSPEED on a cfsetospeed error.
  *
  * \ingroup tty 
  */
int ttyOpenRaw( int & fileDescrip,        ///< [out] the file descriptor.  Set to 0 on an error.
                std::string & deviceName, ///< [in] the device path name, e.g. /dev/ttyUSB0
                speed_t speed             ///< [in] indicates the baud rate (see http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html)
              );

/// Check if the end of the buffer contains the end-of-transmission string
/**
  * \returns true if the last N chars of buffRead are equal to eot, where N is the length of eot.
  * \returns false otherwise.
  * 
  * \ingroup tty 
  */
bool isEndOfTrans( const std::string & strRead, ///< [in] The read buffer to check
                   const std::string & eot      ///< [in] The end-of-transmission string
                 );

/// Write to the tty console indicated by a file descriptor.
/**
  *
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONWRITEPOLL if the poll times out.
  * \returns TTY_E_ERRORONWRITEPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONWRITE if a timeout occurs during the write.
  * \returns TTY_E_ERRORONWRITE if an error occurs writing to the file.
  * 
  * \ingroup tty 
  */
int ttyWrite( const std::string & buffWrite, ///< [in] The characters to write to the tty.
              int fd,                        ///< [in] The file descriptor of the open tty.
              int timeoutWrite               ///< [in] The timeout in milliseconds.
            );

/// Read from a tty console indicated by a file-descriptor, up to a given number of bytes.
/** Polls before attempting to read, but does not wait for all bytes to be ready.
  * 
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
  * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
  * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
  * 
  * \ingroup tty 
  */
int ttyReadRaw( std::vector<unsigned char> & vecRead, ///< [out] The buffer in which to store the output.
                int & readBytes,                      ///< [out] The number of bytes read.
                int fd,                               ///< [in] The file descriptor of the open tty.
                int timeoutRead                       ///< [in] The timeout in milliseconds.
              );

/// Read from a tty console indicated by a file-descriptor, until a given number of bytes are read.
/**
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
  * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
  * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
  * 
  * \ingroup tty 
  */
int ttyRead( std::string & strRead,   ///< [out] The string in which to store the output.
             int bytes,               ///< [in] the number of bytes to read
             int fd,                  ///< [in] The file descriptor of the open tty.
             int timeoutRead          ///< [in] The timeout in milliseconds.
           );

/// Read from a tty console indicated by a file-descriptor, until an end of transmission string is read.
/**
  * \returns TTY_E_NOERROR on success
  * \returns TTY_E_TIMEOUTONREADPOLL if the poll times out.
  * \returns TTY_E_ERRORONREADPOLL if an error is returned by poll.
  * \returns TTY_E_TIMEOUTONREAD if a timeout occurs during the read.
  * \returns TTY_E_ERRORONREAD if an error occurs reading from the file.
  * 
  * \ingroup tty 
  */
int ttyRead( std::string & strRead,   ///< [out] The string in which to store the output.
             const std::string & eot, ///< [in] A sequence of characters which indicates the end of transmission.
             int fd,                  ///< [in] The file descriptor of the open tty.
             int timeoutRead          ///< [in] The timeout in milliseconds.
           );

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
  * 
  * \ingroup tty 
  */
int ttyWriteRead( std::string & strRead,        ///< [out] The string in which to store the output.
                  const std::string & strWrite, ///< [in] The characters to write to the tty.
                  const std::string & eot,      ///< [in] A sequence of characters which indicates the end of transmission.
                  bool swallowEcho,             ///< [in] If true, strWrite.size() characters are read after the write
                  int fd,                       ///< [in] The file descriptor of the open tty.
                  int timeoutWrite,             ///< [in] The write timeout in milliseconds.
                  int timeoutRead               ///< [in] The read timeout in milliseconds.
                );

} //namespace tty
} //namespace MagAOX

#endif //tty_ttyIOUtils_hpp
