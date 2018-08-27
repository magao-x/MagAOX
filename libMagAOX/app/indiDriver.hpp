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

#include "MagAOXApp.hpp"

namespace MagAOX
{
namespace app
{

template<class _parentT>
class indiDriver : public pcf::IndiDriver
{
public:
   typedef _parentT parentT;

protected:

   parentT * m_parent {nullptr};

private:
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

   virtual void execute(void);

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

} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
