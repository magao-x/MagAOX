#ifndef indiDictionary_hpp
#define indiDictionary_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QObject>
#include <QtPlugin>
#include <QTimer>

#include <iostream>

#include <IndiClient.hpp>

//Forward decl:
class rtimvIndiClient;

/// rtimv dictionary using the INDI protocol
/**
  * \todo indiDictionary should only subscribe to specified properties 
  */ 
class indiDictionary : public QObject, public rtimvDictionaryInterface
{
   Q_OBJECT
   Q_PLUGIN_METADATA(IID "rtimv.dictionaryInterface/1.1")
   Q_INTERFACES(rtimvDictionaryInterface)
    
protected:
   std::string m_ipAddress {""}; ///< The IP address of the INDI server
   int m_port {0};               ///< The port of the INDI server
   int m_checkTimeout {1000};    ///< The timeout for checking the INDI connection, in msec
      
   dictionaryT * m_dict {nullptr};
      
   rtimvIndiClient * m_client {nullptr};  
      
   bool m_enabled {false};

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
