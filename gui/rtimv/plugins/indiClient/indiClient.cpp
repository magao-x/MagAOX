
#include "indiClient.hpp"

indiClient::indiClient() : rtimvDictionaryInterface()
{
}

indiClient::~indiClient()
{
   if(m_client) delete m_client;
}

int indiClient::attachDictionary(std::unordered_map<std::string, rtimvDictBlob> * dict)
{
   std::cerr << "indiClient attached\n";
   
   if(m_client) delete m_client;
   
   m_client = new rtimvIndiClient("rtimvIndiClient", "1.7", "1.7", "127.0.0.1", 7624, dict);
   
   m_client->activate();
   
   pcf::IndiProperty ipSend;
   m_client->sendGetProperties( ipSend );
      
   return 0;
}
      
