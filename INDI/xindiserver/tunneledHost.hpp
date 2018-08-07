/** \file tunneledHost.hpp
  * \brief Managing specifics of a remote INDI host to be tunneled over ssh.
  *
  * \ingroup xindiserver_files
  */

#include <cstdlib>
#include <string>
#include <iostream>

#ifndef tunneledHost_hpp
#define tunneledHost_hpp

#ifndef INDI_DEFAULT_PORT
#define INDI_DEFAULT_PORT (7624)
#endif

///Contains the details of a tunneled-host specification and provides parsing.
class tunneledHost
{
   
protected:
   std::string m_name; ///< The host name or ip address
   int m_remotePort {INDI_DEFAULT_PORT}; ///< The remote port on the host
   int m_localPort {0}; ///< The local port which will be forwarded

public:
   ///Get the host name
   /**
     * \returns the current value of m_name
     */
   std::string name() const
   {
      return m_name;
   }

   ///Set the host name.
   /**
     * \returns 0 on sucess.
     * \returns -1 on error.
     */ 
   int name(const std::string & nn /**< [in] the new name */)
   {
      m_name = nn;
      return 0;
   }
      
   ///Get the remote port
   /**
     * \returns the current value of m_remotePort
     */
   int remotePort() const
   {
      return m_remotePort;
   }

   ///Set the remote port
   /**
     * \returns 0 on sucess.
     * \returns -1 on error.
     */ 
   int remotePort(const int & rp /**< [in] the new remote port*/)
   {
      m_remotePort = rp;
      return 0;
   }
   
   ///Get the localPort
   /**
     * \returns the current value of m_localPort
     */
   int localPort() const
   {
      return m_localPort;
   }

   ///Set the local port
   /**
     * \returns 0 on sucess.
     * \returns -1 on error.
     */ 
   int localPort(const int & lp /**< [in] the new local port*/)
   {
      m_localPort = lp;
      return 0;
   }
   
   ///Parse a host specification string into the component parts.
   /**
     * Expects a string in the form 'hostname[:remotePort]:localPort'.
     * hostname and localPort are required.  remotePort is optional, as the
     * default INDI port of 7624 is assumed.
     * Whitespace anywhere in the string is ignored.  The ports are converted
     * to integers.
     *
     * \returns 0 on success
     * \returns -1 on any error, in which case none of the values should be used.
     */
   int parseName( const std::string & cnws)
   {
      //Remove whitespace
      std::string cn = cnws;
      cn.erase(std::remove_if(cn.begin(), cn.end(), ::isspace), cn.end());

      //Find first colon
      size_t first = cn.find(':', 0);

      if(first == 0) //too early
      {
         std::cerr << "No host found.\n";
         return -1;
      }

      if(first == cn.size()-1) //too late
      {
         std::cerr << "Localport can't be empty.\n";
         return -1;
      }

      if(first == std::string::npos) //not there
      {
         std::cerr << "Must specify localport.\n";
         return -1;
      }



      //Look for a 2nd :
      size_t second = cn.find(':', first+1);

      //There is a 2nd
      if(second != std::string::npos )
      {
         if(second == cn.size()-1) //too early
         {
            std::cerr << "Localport can't be empty.\n";
            return -1;
         }

         m_name = cn.substr(0, first);
         ++first;
      
         //Only change default for a non-empty remote port, not for "::"
         if(second-first > 1)
         {
            m_remotePort = atoi( cn.substr(first, second - first).c_str());
         }

         m_localPort  = atoi( cn.substr(second+1, cn.size()-second-1).c_str());
      }
      else //Just one colon
      {
         m_name = cn.substr(0, first);
         ++first;

         m_localPort  = atoi( cn.substr(first, cn.size()-first).c_str());
      }

      return 0;
   }

};

#endif //tunneledHost_hpp
