/** \file netSerial.cpp
  * \brief Managing a connection to a serial device over a network socket.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * This code is taken from the LBTAO/MagAO supervisor source code, modifications
  * mainly for c++.
  * 
  * \ingroup tty_files
  */

#include "netSerial.hpp"

#include <thread>
#include <chrono>

#include <cstring>
#include <cerrno>
#include <sys/time.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>


namespace MagAOX
{
namespace tty
{



int netSerial::serialInit( const char *address, 
                           int port
                         )
{
   struct sockaddr_in servaddr;

   serialClose();
   
   m_sockfd = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP);
   
   if(m_sockfd == -1)
   {
      return NETSERIAL_E_NETWORK;
   }
      
   struct hostent *h = gethostbyname(address);

   servaddr.sin_family=AF_INET;
   servaddr.sin_port=htons(port);
   memcpy( &servaddr.sin_addr, h->h_addr_list[0], h->h_length);

   if(connect( m_sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0)
   {
      return NETSERIAL_E_CONNECT;
   }

   return NETSERIAL_E_NOERROR;
}
   
int netSerial::serialClose(void)
{
   shutdown( m_sockfd, 2);
   close(m_sockfd);
   
   return NETSERIAL_E_NOERROR;
}


int netSerial::serialOut( const char *buf, 
                          int len
                        )
{
   int i=0;
      
   while (i<len)
   {
      int stat = send( m_sockfd, buf+i, len-i, 0);
      if (stat >0)
      {
         i+= stat;
      }
      if ((stat<0) && (stat != EAGAIN))
      {
         return NETSERIAL_E_COMM;
      }
   }

   return NETSERIAL_E_NOERROR;
}

int netSerial::serialIn( char *buffer, 
                         int len, 
                         int timeout
                       )
{
   int res=0;

   memset( buffer, 0, len);

   while (res < len)
   {
      fd_set rdfs;
      struct timeval tv;
      int retval;

      memset( &tv, 0, sizeof( struct timeval));
      FD_ZERO( &rdfs);
      FD_SET( m_sockfd, &rdfs);
      tv.tv_sec = timeout / 1000;
      tv.tv_usec = (timeout-(timeout/1000)*1000)*1000;

      retval = select( m_sockfd+1, &rdfs, NULL, NULL, &tv);
      if (!retval)
         return res;

      res += recv( m_sockfd, buffer+res, len-res, 0);

      std::this_thread::sleep_for(std::chrono::milliseconds(3));
   }
   
   return res;
}

int netSerial::serialInString( char *buffer, 
                               int len, 
                               int timeout, 
                               char terminator
                             )
{
   int res=0;
   struct timeval tv0, tv1;
   double t0, t1;
   
   memset( buffer, 0, len);

   #ifdef DEBUG
      printf("initial timeout = %i\n", timeout);
   #endif
      
   gettimeofday(&tv0, 0);
   t0 = ((double)tv0.tv_sec + (double)tv0.tv_usec/1e6);

   gettimeofday(&tv1, 0);
   t1 = ((double)tv1.tv_sec + (double)tv1.tv_usec/1e6);

   while (res < len && ((t1-t0)*1000. < timeout))
   {
      fd_set rdfs;
      struct timeval tv;
      int retval;

      memset( &tv, 0, sizeof( struct timeval));
      FD_ZERO( &rdfs);
      FD_SET( m_sockfd, &rdfs);
      tv.tv_sec = timeout / 1000;
      tv.tv_usec = (timeout-(timeout/1000)*1000)*1000;

      #ifdef DEBUG
         printf("Selecting . . .\n");
      #endif
         
      /* Making this a signal-safe call to select*/
      /*JRM: otherwise, signals will cause select to return
          causing this loop to never timeout*/
         
      int signaled = 1;

      retval = 0;
      
      while(signaled && ((t1-t0)*1000. < timeout))
      {
         errno = 0;
         signaled = 0;
         retval = select(m_sockfd+1, &rdfs, NULL, NULL, &tv);
         if(retval < 0)
         {
            //This means select was interrupted by a signal, so keep going.
            if(errno == EINTR)
            {
               #ifdef DEBUG
                  printf("EINTR\n");
               #endif
                  
               signaled = 1;
               gettimeofday(&tv1, 0);
               t1 = ((double)tv1.tv_sec + (double)tv1.tv_usec/1e6) ;
               //Reset for smaller timeout
               timeout = timeout - (t1-t0)*1000.;

               #ifdef DEBUG
                  printf("t1-t0 = %f\n", (t1-t0)*1000);
                  printf("timeout = %i\n", timeout);
               #endif
               
               tv.tv_sec = timeout / 1000;
               tv.tv_usec = (timeout-(timeout/1000)*1000)*1000;
               if(tv.tv_sec < 0) tv.tv_sec = 0;
               if(tv.tv_usec < 0) tv.tv_usec = 0;
                  
            }
         }
      }

      #ifdef DEBUG
         printf("select returned %i . . .\n", retval);
      #endif
      
      if (retval <= 0) //this means we timed out or had an error
            return res;

      #ifdef DEBUG
         printf("Starting read . . .\n");
      #endif
            
      res += recv( m_sockfd, buffer+res, len-res, 0);
      
      #ifdef DEBUG
         printf("Read %i bytes. . .\n", res);
      #endif

      buffer[res] =0;
         
      #ifdef DEBUG
         printf("SerialInString received %d bytes: %s\n", res, buffer);
      #endif
         
      if (strchr( buffer, terminator)) return res;

      gettimeofday(&tv1, 0);
      t1 = ((double)tv1.tv_sec + (double)tv1.tv_usec/1e6);

      #ifdef DEBUG
         printf("t1-t0 = %f\n", (t1-t0)*1000.);
         printf("timeout = %i\n", timeout);
      #endif

      std::this_thread::sleep_for(std::chrono::milliseconds(3));
   }

   
   #ifdef DEBUG
      int i;
      printf("SerialIn(): received %d characters:\n", res);
   
      for ( i=0; i<res; i++) printf("0x%02X ", (unsigned char) buffer[i]);
      printf("\n");
   #endif

   return res;
}

int netSerial::getSocketFD(void)
{
   return m_sockfd;
}

} // namespace tty
} // namespace MagAOX

