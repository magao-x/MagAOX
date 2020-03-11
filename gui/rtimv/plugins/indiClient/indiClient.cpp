
#include "indiClient.hpp"

indiClient::indiClient() : rtimvDictionaryInterface()
{
   connect(&m_connTimer, SIGNAL(timeout()), this, SLOT(checkConnection()));
   m_connTimer.start(1000); ///\todo make timeout a config-able variable
}

indiClient::~indiClient()
{
   if(m_client) delete m_client;
}

int indiClient::attachDictionary(std::unordered_map<std::string, rtimvDictBlob> * dict)
{
   std::cerr << "indiClient attached\n";
   
   m_dict = dict;
   
   return 0;
}
      
void indiClient::checkConnection()
{
   if(!m_client)
   {
      m_client = new rtimvIndiClient("rtimvIndiClient", "1.7", "1.7", "127.0.0.1", 7624, m_dict);
      m_client->activate();
   
      pcf::IndiProperty ipSend;
      m_client->sendGetProperties( ipSend );
    
      std::cerr << "indiClient connected\n";
      return;
   }
   else if(m_client->getQuitProcess())
   {
      m_client->quitProcess();
      m_client->deactivate();
      delete m_client;
      m_client = nullptr;
      
      std::cerr << "indiClient disconnected\n";
      
      return;
   }
   
}
