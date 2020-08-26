#ifndef indiDictionary_hpp
#define indiDictionary_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QObject>
#include <QtPlugin>
#include <QTimer>

#include <iostream>

#include <IndiClient.hpp>

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

class indiDictionary : public QObject,
                   public rtimvDictionaryInterface
{
   Q_OBJECT
   Q_PLUGIN_METADATA(IID "rtimv.dictionaryInterface/1.0")
   Q_INTERFACES(rtimvDictionaryInterface)
    
   std::string m_ipAddress {"127.0.0.1"};
   int m_port {7624};
   
   protected:
      
      dictionaryT * m_dict {nullptr};
      
      rtimvIndiClient * m_client {nullptr};  
      
      QTimer m_connTimer;
      
   public:
      indiDictionary();
      
      virtual ~indiDictionary();

      virtual int attachDictionary( dictionaryT * dict,
                                    mx::app::appConfigurator & config
                                  ); 

   public slots:
      void checkConnection();
   

};

#endif //indiDictionary_hpp
