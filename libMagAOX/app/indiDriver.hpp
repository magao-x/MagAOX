/** \file indiDriver.hpp 
  * \brief MagAO-X INDI Driver Wrapper
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2018-05-26 created by JRM
  */ 

#ifndef app_indiDriver_hpp
#define app_indiDriver_hpp

#include <indi/IndiDriver.hpp>
#include <indi/IndiElement.hpp>

#include "MagAOXApp.hpp"

namespace MagAOX 
{
namespace app 
{

//class MagAOXApp;

template<class _parentT>
class indiDriver : public pcf::IndiDriver
{
public:
   typedef _parentT parentT;
   
protected:

   parentT * m_parent {nullptr};

public:
   
   indiDriver ( parentT * parent,
                const std::string &szName, 
                const std::string &szDriverVersion, 
                const std::string &szProtocolVersion
              );
   
   // override callbacks
   virtual void handleGetProperties( const pcf::IndiProperty &ipRecv );

   virtual void handleNewProperty( const pcf::IndiProperty &ipRecv );

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
void indiDriver<parentT>::execute()
{
   processIndiRequests(false);
}

} //namespace app 
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp

