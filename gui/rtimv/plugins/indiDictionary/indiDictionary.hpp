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
  * 
  */ 
class indiDictionary : public rtimvDictionaryInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "rtimv.dictionaryInterface/1.2")
    Q_INTERFACES(rtimvDictionaryInterface)
    
protected:
    std::string m_ipAddress {""}; ///< The IP address of the INDI server
    int m_port {0};               ///< The port of the INDI server
    int m_checkTimeout {1000};    ///< The timeout for checking the INDI connection, in msec
      
    dictionaryT * m_dict {nullptr}; ///< The rtimv INDI dictionary.
   
    rtimvIndiClient * m_client {nullptr}; ///< The INDI client, which is destroyed each time connection is lost

    std::mutex m_clientMutex; ///< Protection for access to client.
      
    bool m_enabled {false}; ///< Whether or not this plugin is enabled

    QTimer m_connTimer; ///< Time for checking the connection.
      
public:

    /// c'tor
    indiDictionary();
      
    /// d'tor
    virtual ~indiDictionary();

    /// Attach this plugin to rtimv
    virtual int attachDictionary( dictionaryT * dict,               ///< [in] pointer to the rtimv dictionary, a std::map
                                  mx::app::appConfigurator & config ///< [in] app configurator from which to config the connection
                                ); 

public slots:

    /// Check the status of the INDI connection.    
    void checkConnection();
   
public:
    virtual std::vector<std::string> info();

};

#endif //indiDictionary_hpp
