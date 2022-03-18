
#include "indiDictionary.hpp"

class rtimvIndiClient: public pcf::IndiClient
{
protected:
   bool m_shutdown {false};
   bool m_connectionLost{false};

   dictionaryT * m_dict {nullptr};
   
   
public:
   rtimvIndiClient( const std::string &szName,
                    const std::string &szVersion,
                    const std::string &szProtocolVersion,
                    const std::string & ipAddress,
                    const int port,
                    dictionaryT * dict
                  ) : pcf::IndiClient(szName, szVersion, szProtocolVersion, ipAddress, port)  //"127.0.0.1", 7624)
   {
      m_dict = dict;
   }

   ~rtimvIndiClient()
   {
      quitProcess();
      deactivate();
   }
   
   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv )
   {
      if(m_dict == nullptr) return;
      
      std::string key = ipRecv.createUniqueKey();
      
      auto elIt = ipRecv.getElements().begin();

      while(elIt != ipRecv.getElements().end())
      {
         std::string elKey = key + "." + elIt->second.getName();

         std::string val;
         
         if(ipRecv.getType() == pcf::IndiProperty::Switch)
         {
            if(ipRecv[elIt->second.getName()].getSwitchState() == pcf::IndiElement::On) val = "on";
            else if (ipRecv[elIt->second.getName()].getSwitchState() == pcf::IndiElement::Off) val = "off";
            else val = "unk";
         }
         else
         {
            val = ipRecv[elIt->second.getName()].get();
         }
         
         (*m_dict)[elKey].setBlob(val.c_str(), val.size()+1);
         
         //val = (char *) (*m_dict)[elKey].m_blob;
         
         //std::cout << "elKey: " << elKey << " " <<  val << "\n";
         
         ++elIt;
      }
   }

   virtual void handleDelProperty( const pcf::IndiProperty &ipRecv )
   {
      if(m_dict == nullptr) return;
      
      static_cast<void>(ipRecv);
   }

   virtual void handleMessage( const pcf::IndiProperty &ipRecv )
   {
      static_cast<void>(ipRecv);
   }

   virtual void handleSetProperty( const pcf::IndiProperty &ipRecv )
   {
      handleDefProperty(ipRecv);
   }
   
   virtual void execute()
   {
      processIndiRequests(false);
   }

};



indiDictionary::indiDictionary() : rtimvDictionaryInterface()
{
   
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
   
   config.configUnused(m_port, mx::app::iniFile::makeKey("indi", "port"));
   
   config.configUnused(m_checkTimeout, mx::app::iniFile::makeKey("indi", "checkTimeout"));

   if(m_ipAddress == "" || m_port <= 0)
   {
      std::cerr << "INDI Dictionary: no connection specified. INDI disabled.\n";
      m_enabled = false;
   }
   else
   {
      std::cerr << "INDI Dictionary: enabling for " << m_ipAddress << ":" << m_port << "\n";
      m_enabled = true;
      checkConnection();
      connect(&m_connTimer, SIGNAL(timeout()), this, SLOT(checkConnection()));
      m_connTimer.start(m_checkTimeout);
   }

   return 0;
}
      
void indiDictionary::checkConnection()
{
   if(!m_enabled) return;

   if(!m_client)
   {
      try
      {
         m_client = new rtimvIndiClient("rtimvIndiClient", "1.7", "1.7", m_ipAddress, m_port, m_dict);
      }   
      catch(...)
      {
         //This means failed to connect, often b/c tunnel not open.  m_client will still be nullptr.
         //just go on and try again
         return;
      }

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
      return;
   }
   
}

