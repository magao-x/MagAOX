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
