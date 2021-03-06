
#include "indiDictionary.hpp"

indiDictionary::indiDictionary() : rtimvDictionaryInterface()
{
   connect(&m_connTimer, SIGNAL(timeout()), this, SLOT(checkConnection()));
   m_connTimer.start(1000); ///\todo make timeout a config-able variable
}

indiDictionary::~indiDictionary()
{
   if(m_client) delete m_client;
}

int indiDictionary::attachDictionary( dictionaryT * dict,
                                  mx::app::appConfigurator & config
                                )
{
   m_dict = dict;
   
   config.configUnused(m_ipAddress, mx::app::iniFile::makeKey("indi", "ipAddress"));
   std::cerr << "indi ip: " << m_ipAddress << "\n";
   
   config.configUnused(m_port, mx::app::iniFile::makeKey("indi", "port"));
   std::cerr << "indi port: " << m_port << "\n";
   
   return 0;
}
      
void indiDictionary::checkConnection()
{
   if(!m_client)
   {
      m_client = new rtimvIndiClient("rtimvIndiClient", "1.7", "1.7", m_ipAddress, m_port, m_dict);
      m_client->activate();
   
      pcf::IndiProperty ipSend;
      m_client->sendGetProperties( ipSend );
    
      return;
   }
   else if(m_client->getQuitProcess())
   {
      m_client->quitProcess();
      m_client->deactivate();
      delete m_client;
      m_client = nullptr;
      
      //std::cerr << "indiDictionary disconnected\n";
      
      return;
   }
   
}
