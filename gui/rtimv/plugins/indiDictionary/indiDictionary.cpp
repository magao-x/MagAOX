#include <set>

#include "indiDictionary.hpp"


class rtimvIndiClient: public pcf::IndiClient
{
protected:
    bool m_shutdown{false};
    bool m_connectionLost{false};

public:
    std::set<std::string> m_subscribed;

protected:
    dictionaryT *m_dict{nullptr};

    float m_northAngle{0};

public:
    rtimvIndiClient(const std::string &szName,
                    const std::string &szVersion,
                    const std::string &szProtocolVersion,
                    const std::string &ipAddress,
                    const int port,
                    dictionaryT *dict) : pcf::IndiClient(szName, szVersion, szProtocolVersion, ipAddress, port) //"127.0.0.1", 7624)
    {
        m_dict = dict;
    }

    ~rtimvIndiClient()
    {
        quitProcess();
        deactivate();
    }

    virtual void handleDefProperty(const pcf::IndiProperty &ipRecv)
    {
        if (m_dict == nullptr)
            return;

        std::string key = ipRecv.createUniqueKey();

        if (m_subscribed.count(key) == 0)
            return;

        auto elIt = ipRecv.getElements().begin();

        while(elIt != ipRecv.getElements().end())
        {
            std::string elKey = key + "." + elIt->second.name();

            std::string val;

            if(ipRecv.getType() == pcf::IndiProperty::Switch)
            {
                if(ipRecv[elIt->second.name()].getSwitchState() == pcf::IndiElement::On)
                    val = "on";
                else if(ipRecv[elIt->second.name()].getSwitchState() == pcf::IndiElement::Off)
                    val = "off";
                else
                    val = "unk";
            }
            else
            {
                val = ipRecv[elIt->second.name()].get();
            }

            (*m_dict)[elKey].setBlob(val.c_str(), val.size() + 1);

            if(elKey == "tcsi.teldata.pa")
            {
                m_northAngle = ipRecv[elIt->second.name()].get<float>();
                (*m_dict)["rtimv.north.angle"].setBlob(&m_northAngle, sizeof(float));
            }

            ++elIt;
        }
    }

    virtual void handleDelProperty(const pcf::IndiProperty &ipRecv)
    {
        if(m_dict == nullptr)
        {
            return;
        }

        static_cast<void>(ipRecv);
    }

    virtual void handleMessage(const pcf::IndiProperty &ipRecv)
    {
        static_cast<void>(ipRecv);
    }

    virtual void handleSetProperty(const pcf::IndiProperty &ipRecv)
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
      m_enabled = false;
      return 1; 
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

   std::lock_guard<std::mutex> lock(m_clientMutex);

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
   }
   else if(m_client->getQuitProcess())
   {
      m_client->quitProcess();
      m_client->deactivate();
      delete m_client;
      m_client = nullptr;
      return;
   }

   if(!m_client) return;

   //Well if we're here we're connected, so now check if we're listening to our props
   for(auto it=m_dict->begin(); it != m_dict->end(); ++it)
   {
      std::string elKey = it->first;

      size_t np = elKey.find('.', 0);
      std::string dev = elKey.substr(0, np);
      
      size_t ap = elKey.find('.', np+1);
      std::string prop = elKey.substr(np+1, ap-(np+1));

      std::string key = dev + "." + prop;
      
      auto res = m_client->m_subscribed.insert(key);
      if(res.second == true) //If we have inserted it, we snoop it
      {
         pcf::IndiProperty ipSend;
         ipSend.setDevice(dev);
         ipSend.setName(prop);
         m_client->sendGetProperties(ipSend);
      }
      
   }
   
}

std::vector<std::string> indiDictionary::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back("INDI dictionary: " + m_ipAddress + ":" + std::to_string(m_port));
    if(m_client) vinfo[0] += " [connected]";
    else vinfo[0] += " [not connected]";

    return vinfo;
}
