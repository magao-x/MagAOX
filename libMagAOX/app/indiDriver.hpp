/** \file indiDriver.hpp
  * \brief MagAO-X INDI Driver Wrapper
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-05-26 created by JRM
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
   pcf::IndiClient * m_outGoing {nullptr};

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
      m_good = false;
      return;
   }
   setInputFd(fd);

   errno = 0;
   fd = open( parent->driverOutName().c_str(), O_RDWR);
   if(fd < 0)
   {
      m_good = false;
      return;
   }
   setOutputFd(fd);
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
   if( m_outGoing == nullptr)
   {
      try
      {
         m_outGoing = new pcf::IndiClient(m_serverIPAddress, m_serverPort);
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
   }

   try
   {
      m_outGoing->sendNewProperty(ipRecv);
   }
   catch(...)
   {
      parentT::template log<logger::software_error>({__FILE__, __LINE__, "Exception from IndiClient::sendNewProperty"});
      return -1;
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
