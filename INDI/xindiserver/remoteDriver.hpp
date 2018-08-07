/** \file remoteDriver.hpp
  * \brief Managing specifics of a remote INDI driver specification
  *
  * \ingroup xindiserver_files
  */

#include <cstdlib>
#include <string>
#include <iostream>

#ifndef remoteDriver_hpp
#define remoteDriver_hpp

#ifndef INDI_DEFAULT_PORT
#define INDI_DEFAULT_PORT (7624)
#endif

///Contains the details of a remote driver specification and provides parsing.
/**
  * A remote driver is specified in terms of its name, host, and (optionally) a port
  * on that host.
  */ 
class remoteDriver
{
   
protected:
   std::string m_name; ///< The driver name
   std::string m_host; ///< The remote host
   int m_port {INDI_DEFAULT_PORT}; ///< The port on the remote host
  
public:
   ///Get the driver name
   /**
     * \returns the current value of m_name
     */
   std::string name() const
   {
      return m_name;
   }

   ///Set the driver name.
   /** Checks for @ and : characters in the name, which are errors.
     * 
     * \returns 0 on sucess.
     * \returns -1 on error.
     */ 
   int name(const std::string & nn /**< [in] the new name */)
   {
      if(nn.find('@', 0) != std::string::npos)
      {
         std::cerr << "No @ allowed in driver name.  Use parse instead.\n";
         return -1;
      }
      
      if(nn.find(':', 0) != std::string::npos)
      {
         std::cerr << "No : allowed in driver name.  Use parse instead.\n";
         return -1;
      }
      
      m_name = nn;
      return 0;
   }
      
   ///Get the host
   /**
     * \returns the current value of m_host
     */
   std::string host() const
   {
      return m_host;
   }

   ///Set the host
   /**
     * \returns 0 on sucess.
     * \returns -1 on error.
     */ 
   int host(const std::string & nh /**< [in] the new host*/)
   {
      if(nh.find('@', 0) != std::string::npos)
      {
         std::cerr << "No @ allowed in host.  Use parse instead.\n";
         return -1;
      }
      
      if(nh.find(':', 0) != std::string::npos)
      {
         std::cerr << "No : allowed in  host.  Use parse instead.\n";
         return -1;
      }
      
      m_host = nh;
      return 0;
   }
   
   ///Get the port
   /**
     * \returns the current value of m_port
     */
   int port() const
   {
      return m_port;
   }

   ///Set the port
   /**
     * \returns 0 on sucess.
     * \returns -1 on error.
     */ 
   int port(const int & np /**< [in] the new port*/)
   {
      m_port = np;
      return 0;
   }
   
   ///Parse a remote driver specification string into the component parts.
   /**
     * Expects a string in the form 'name@host[:port]'.
     * Both name and host are required, port is optional.
     * Whitespace is ignored.
     * 
     * \returns 0 on success
     * \returns -1 on any error, in which case none of the values should be used.
     */
   int parse( const std::string & cnws)
   {
      //Remove whitespace
      std::string cn = cnws;
      cn.erase(std::remove_if(cn.begin(), cn.end(), ::isspace), cn.end());

      //Find the @
      size_t first = cn.find('@', 0);

      if(first == 0) //too early
      {
         std::cerr << "Name can't be empty.\n";
         return -1;
      }

      if(first == cn.size()-1) //too late
      {
         std::cerr << "Host can't be empty.\n";
         return -1;
      }

      if(first == std::string::npos) //not there
      {
         std::cerr << "Host can't be empty, no @ found.\n";
         return -1;
      }

      size_t second = cn.find(':',0);
      if(second != std::string::npos)
      {
         if(second < first) //port in wrong spot
         {
            std::cerr << "Host can't be empty, : found before @.\n";
            return -1;
         }
         
         if(second < cn.size()-1)
         {
            m_port = atoi(cn.substr(second+1, cn.size() - second-1).c_str());
         }
         //Otherwise we don't change anything since 0 isn't a valid port.
         
      }
      else
      {
         second=cn.size();
      }

      m_name = cn.substr(0, first);
      ++first;

      m_host = cn.substr(first, second-first);
      

      return 0;
   }

};

#endif //remoteDriver_hpp
