/** \file indiDictionary.cpp
 * \brief Implements the rtimv INDI-backed dictionary plugin.
 *
 * \author Jared Males
 */

#include <set>
#include <utility>
#include <vector>

#include "indiDictionary.hpp"

namespace
{
bool dictionaryElementKeyToPropertyKey( const std::string &elKey, std::string &device, std::string &property )
{
    size_t np = elKey.find( '.', 0 );
    if( np == std::string::npos )
    {
        return false;
    }

    size_t ap = elKey.find( '.', np + 1 );
    if( ap == std::string::npos )
    {
        return false;
    }

    device   = elKey.substr( 0, np );
    property = elKey.substr( np + 1, ap - ( np + 1 ) );

    return true;
}
} // namespace

class rtimvIndiClient : public pcf::IndiClient
{
  public:
    std::set<std::string> m_subscribed;
    std::mutex            m_subscribedMutex; ///< Protects the subscribed-property set.

  protected:
    dictionaryT *m_dict{ nullptr };      ///< Shared rtimv dictionary populated from INDI updates.
    std::mutex  *m_dictMutex{ nullptr }; ///< Protects dictionary structure while keys are inserted or snapshotted.

  public:
    rtimvIndiClient( const std::string &szName,
                     const std::string &szVersion,
                     const std::string &szProtocolVersion,
                     const std::string &ipAddress,
                     const int          port,
                     dictionaryT       *dict,
                     std::mutex        *dictMutex );

    ~rtimvIndiClient();

    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );

    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv );

    virtual void handleMessage( const pcf::IndiProperty &ipRecv );

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

    virtual void execute();
};

rtimvIndiClient::rtimvIndiClient( const std::string &szName,
                                  const std::string &szVersion,
                                  const std::string &szProtocolVersion,
                                  const std::string &ipAddress,
                                  const int          port,
                                  dictionaryT       *dict,
                                  std::mutex        *dictMutex )
    : pcf::IndiClient( szName, szVersion, szProtocolVersion, ipAddress, port )
{
    m_dict      = dict;
    m_dictMutex = dictMutex;
}

rtimvIndiClient::~rtimvIndiClient()
{
    quitProcess();
    deactivate();
}

void rtimvIndiClient::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    if( m_dict == nullptr || m_dictMutex == nullptr )
    {
        return;
    }

    std::string key = ipRecv.createUniqueKey();

    {
        std::lock_guard<std::mutex> guard( m_subscribedMutex );
        if( m_subscribed.count( key ) == 0 )
        {
            return;
        }
    }

    std::vector<std::pair<std::string, std::string>> updates;
    updates.reserve( ipRecv.getElements().size() );

    bool  updateNorthAngle{ false };
    float northAngle{ 0 };

    auto elIt = ipRecv.getElements().begin();

    while( elIt != ipRecv.getElements().end() )
    {
        std::string elName = elIt->second.getName();
        std::string elKey  = key + "." + elName;

        std::string val;

        if( ipRecv.getType() == pcf::IndiProperty::Switch )
        {
            if( ipRecv[elName].getSwitchState() == pcf::IndiElement::On )
            {
                val = "on";
            }
            else if( ipRecv[elName].getSwitchState() == pcf::IndiElement::Off )
            {
                val = "off";
            }
            else
            {
                val = "unk";
            }
        }
        else
        {
            val = ipRecv[elName].get();
        }

        updates.emplace_back( elKey, val );

        if( elKey == "tcsi.teldata.pa" )
        {
            northAngle       = ipRecv[elName].get<float>();
            updateNorthAngle = true;
        }

        ++elIt;
    }

    std::lock_guard<std::mutex> guard( *m_dictMutex );

    for( const auto &update : updates )
    {
        // Subscribed properties can expose dynamic elements. Keep those
        // element keys populated so overlays can discover active presets.
        ( *m_dict )[update.first].setBlob( update.second.c_str(), update.second.size() + 1 );
    }

    if( updateNorthAngle )
    {
        auto nit = m_dict->find( "rtimv.north.angle" );
        if( nit != m_dict->end() )
        {
            nit->second.setBlob( &northAngle, sizeof( float ) );
        }
    }
}

void rtimvIndiClient::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv );
}

void rtimvIndiClient::handleMessage( const pcf::IndiProperty &ipRecv )
{
    static_cast<void>( ipRecv );
}

void rtimvIndiClient::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    handleDefProperty( ipRecv );
}

void rtimvIndiClient::execute()
{
    processIndiRequests( false );
}

indiDictionary::indiDictionary() : rtimvDictionaryInterface()
{
}

indiDictionary::~indiDictionary()
{
    m_connTimer.stop();
    QObject::disconnect( &m_connTimer, SIGNAL( timeout() ), this, SLOT( checkConnection() ) );

    std::lock_guard<std::mutex> lock( m_clientMutex );
    if( m_client )
        delete m_client;
    m_client = nullptr;
}

int indiDictionary::attachDictionary( dictionaryT *dict, mx::app::appConfigurator &config )
{
    m_dict = dict;

    config.configUnused( m_ipAddress, mx::app::iniFile::makeKey( "indi", "ipAddress" ) );

    config.configUnused( m_port, mx::app::iniFile::makeKey( "indi", "port" ) );

    config.configUnused( m_checkTimeout, mx::app::iniFile::makeKey( "indi", "checkTimeout" ) );

    if( m_ipAddress == "" || m_port <= 0 )
    {
        pluginLogInfo( "not configured" );

        m_enabled = false;
        return 1;
    }
    else
    {
        pluginLogInfo( std::format( "enabling for {}:{}", m_ipAddress, m_port ) );

        m_enabled = true;
        checkConnection();
        connect( &m_connTimer, SIGNAL( timeout() ), this, SLOT( checkConnection() ) );
        m_connTimer.start( m_checkTimeout );

        if( m_dict )
        {
            std::lock_guard<std::mutex> guard( m_dictMutex );
            ( *m_dict )["tcsi.teldata.pa"].setBlob( nullptr, 0 );
            ( *m_dict )["rtimv.north.angle"].setBlob( nullptr, 0 );
        }
    }

    return 0;
}

void indiDictionary::checkConnection()
{
    if( !m_enabled )
        return;

    std::lock_guard<std::mutex> lock( m_clientMutex );

    if( !m_client )
    {
        try
        {
            m_client =
                new rtimvIndiClient( "rtimvIndiClient", "1.7", "1.7", m_ipAddress, m_port, m_dict, &m_dictMutex );
        }
        catch( ... )
        {
            // This means failed to connect, often b/c tunnel not open.  m_client will still be nullptr.
            // just go on and try again
            return;
        }

        m_client->activate();
    }
    else if( m_client->getQuitProcess() )
    {
        m_client->quitProcess();
        m_client->deactivate();
        delete m_client;
        m_client = nullptr;
        return;
    }

    if( !m_client )
        return;

    std::set<std::pair<std::string, std::string>> propertyKeys;

    {
        std::lock_guard<std::mutex> guard( m_dictMutex );

        if( m_dict == nullptr )
        {
            return;
        }

        // Well if we're here we're connected, so now check if we're listening to our props.
        for( auto it = m_dict->begin(); it != m_dict->end(); ++it )
        {
            std::string dev;
            std::string prop;

            if( !dictionaryElementKeyToPropertyKey( it->first, dev, prop ) )
            {
                continue;
            }

            propertyKeys.insert( std::make_pair( dev, prop ) );
        }
    }

    for( const auto &propertyKey : propertyKeys )
    {
        bool shouldSnoop = false;
        {
            std::lock_guard<std::mutex> guard( m_client->m_subscribedMutex );
            auto res    = m_client->m_subscribed.insert( propertyKey.first + "." + propertyKey.second );
            shouldSnoop = res.second;
        }

        if( shouldSnoop == true ) // If we have inserted it, we snoop it
        {
            pcf::IndiProperty ipSend;
            ipSend.setDevice( propertyKey.first );
            ipSend.setName( propertyKey.second );
            m_client->sendGetProperties( ipSend );
        }
    }
}

std::vector<std::string> indiDictionary::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back( "INDI dictionary: " + m_ipAddress + ":" + std::to_string( m_port ) );
    if( m_client )
        vinfo[0] += " [connected]";
    else
        vinfo[0] += " [not connected]";

    return vinfo;
}
