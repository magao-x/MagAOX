/** \file indiDictionary.cpp
 * \brief Implements the rtimv INDI-backed dictionary plugin.
 *
 * \author Jared Males
 */

#include <cmath>
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

/// INDI receive thread; dictionary and property-cache mutations are queued to the GUI thread.
class rtimvIndiClient : public pcf::IndiClient
{
  public:
    std::set<std::string> m_subscribed; ///< Requested property keys.
    std::mutex m_subscribedMutex; ///< Guards subscriptions shared with the GUI.
  protected:
    indiDictionary *m_owner; ///< Plugin owns this client and outlives its receive thread.
    quint64 m_generation; ///< Generation captured when this client connected.

    /// Queue a complete property without changing shared dictionary structures.
    void receive(const pcf::IndiProperty &property, /**< [in] Incoming vector. */
                 bool definition, /**< [in] True for DefProperty. */
                 bool removed = false /**< [in] True for DelProperty. */);
  public:
    /// Create the single connection used for both dictionary updates and commands.
    rtimvIndiClient(const std::string &address, /**< [in] Configured INDI address. */
                   int port, /**< [in] Configured INDI port. */
                   indiDictionary *owner, /**< [in] Owning GUI plugin. */
                   quint64 generation /**< [in] Connection generation. */);
    /// Stop the receive thread before releasing the client.
    ~rtimvIndiClient();
    /// Queue an incoming definition.
    void handleDefProperty(const pcf::IndiProperty &property /**< [in] Definition. */) override;
    /// Queue a property or device deletion.
    void handleDelProperty(const pcf::IndiProperty &property /**< [in] Deleted property/device. */) override;
    /// Ignore unsolicited log messages.
    void handleMessage(const pcf::IndiProperty &property /**< [in] Server message. */) override;
    /// Queue an incoming value update.
    void handleSetProperty(const pcf::IndiProperty &property /**< [in] Values. */) override;
    /// Execute the INDI receive loop.
    void execute() override;
};

rtimvIndiClient::rtimvIndiClient(const std::string &address, int port, indiDictionary *owner, quint64 generation)
    : pcf::IndiClient("rtimvIndiClient", "1.7", "1.7", address, port), m_owner(owner), m_generation(generation)
{
}
rtimvIndiClient::~rtimvIndiClient() { quitProcess(); deactivate(); }
void rtimvIndiClient::receive(const pcf::IndiProperty &property, bool definition, bool removed)
{
    if(!removed)
    { //mutex scope
        std::lock_guard<std::mutex> guard(m_subscribedMutex);
        if(!m_subscribed.count(property.createUniqueKey())) return;
    }
    QMetaObject::invokeMethod(m_owner, [owner=m_owner, generation=m_generation, property, definition, removed]()
    {
        owner->receiveProperty(property, generation, definition, removed);
    }, Qt::QueuedConnection);
}
void rtimvIndiClient::handleDefProperty(const pcf::IndiProperty &property) { receive(property, true); }
void rtimvIndiClient::handleDelProperty(const pcf::IndiProperty &property) { receive(property, false, true); }
void rtimvIndiClient::handleMessage(const pcf::IndiProperty &property) { static_cast<void>(property); }
void rtimvIndiClient::handleSetProperty(const pcf::IndiProperty &property) { receive(property, false); }
void rtimvIndiClient::execute() { processIndiRequests(false); }

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
                new rtimvIndiClient( m_ipAddress, m_port, this, ++m_generation );
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
        invalidateConnection();
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

QString indiDictionary::commandProtocol() const { return "indi"; }

bool indiDictionary::commandConnected() const
{
    return m_client && !m_client->getQuitProcess();
}

void indiDictionary::writeCommand(const pcf::IndiProperty &property)
{
    m_client->sendNewProperty(property);
}

rtimvPropertySnapshot indiDictionary::propertySnapshot(const QString &device, const QString &property) const
{
    if(!commandConnected()) return {};
    return m_properties.value(device + "." + property);
}

void indiDictionary::invalidateConnection()
{
    ++m_generation;
    if(m_dict)
    {
        for(auto it = m_properties.begin(); it != m_properties.end(); ++it)
        {
            const std::string prefix = it.key().toStdString() + ".";
            for(auto entry = m_dict->lower_bound(prefix); entry != m_dict->end() && entry->first.starts_with(prefix); ++entry)
                entry->second.setBlob(nullptr, 0);
            if(it.key() == "tcsi.teldata") (*m_dict)["rtimv.north.angle"].setBlob(nullptr, 0);
        }
    }
    m_properties.clear();
}

void indiDictionary::receiveProperty(const pcf::IndiProperty &property, quint64 generation, bool definition, bool removed)
{
    if(generation != m_generation || !commandConnected()) return;
    QString device = QString::fromStdString(property.getDevice());
    QString key = QString::fromStdString(property.createUniqueKey());
    if(removed)
    {
        QString prefix = property.getName().empty() ? device + "." : key + ".";
        for(auto it = m_properties.begin(); it != m_properties.end(); )
        {
            if(it.key() == key || it.key().startsWith(prefix)) it = m_properties.erase(it);
            else ++it;
        }
        if(m_dict) for(auto &entry : *m_dict)
            if(QString::fromStdString(entry.first).startsWith(prefix)) entry.second.setBlob(nullptr, 0);
        if(m_client)
        { //mutex scope
            std::lock_guard<std::mutex> guard(m_client->m_subscribedMutex);
            for(auto it = m_client->m_subscribed.begin(); it != m_client->m_subscribed.end(); )
            {
                if(*it == key.toStdString() || QString::fromStdString(*it).startsWith(prefix)) it = m_client->m_subscribed.erase(it);
                else ++it;
            }
        }
        return;
    }
    rtimvPropertySnapshot snapshot = definition ? rtimvPropertySnapshot{} : m_properties.value(key);
    if(definition)
    {
        snapshot.m_valid = true;
        snapshot.m_writable = property.getPerm() == pcf::IndiProperty::ReadWrite || property.getPerm() == pcf::IndiProperty::WriteOnly;
        if(property.getType() == pcf::IndiProperty::Number) snapshot.m_type = rtimvPropertyType::number;
        else if(property.getType() == pcf::IndiProperty::Switch) snapshot.m_type = rtimvPropertyType::toggle;
        else if(property.getType() == pcf::IndiProperty::Text) snapshot.m_type = rtimvPropertyType::text;
        else snapshot.m_valid = false;
    }
    snapshot.m_generation = generation;
    snapshot.m_complete = snapshot.m_valid;
    for(auto it = snapshot.m_elements.begin(); it != snapshot.m_elements.end(); ++it)
        if(!property.find(it.key().toStdString())) snapshot.m_complete = false;
    if((snapshot.m_type == rtimvPropertyType::number && property.getType() != pcf::IndiProperty::Number) ||
       (snapshot.m_type == rtimvPropertyType::toggle && property.getType() != pcf::IndiProperty::Switch) ||
       (snapshot.m_type == rtimvPropertyType::text && property.getType() != pcf::IndiProperty::Text))
        snapshot.m_valid = snapshot.m_complete = false;
    snapshot.m_revision = ++m_revision;
    switch(property.getState())
    {
        case pcf::IndiProperty::Ok: snapshot.m_state = "ok"; break;
        case pcf::IndiProperty::Busy: snapshot.m_state = "busy"; break;
        case pcf::IndiProperty::Alert: snapshot.m_state = "alert"; break;
        default: snapshot.m_state = "idle";
    }
    for(const auto &entry : property.getElements())
    {
        QString name = QString::fromStdString(entry.first);
        std::string value = entry.second.get();
        if(property.getType() == pcf::IndiProperty::Switch)
        {
            auto state = entry.second.getSwitchState();
            value = state == pcf::IndiElement::On ? "on" : state == pcf::IndiElement::Off ? "off" : "unk";
            snapshot.m_elements[name] = state == pcf::IndiElement::UnknownSwitchState ? QVariant() : QVariant(state == pcf::IndiElement::On);
        }
        else if(property.getType() == pcf::IndiProperty::Number)
        {
            bool ok;
            double number = QString::fromStdString(value).toDouble(&ok);
            snapshot.m_elements[name] = ok && std::isfinite(number) ? QVariant(number) : QVariant();
        }
        else snapshot.m_elements[name] = QString::fromStdString(value);
        if(m_dict)
        {
            (*m_dict)[property.createUniqueKey() + "." + entry.first].setBlob(value.c_str(), value.size()+1);
            if(key == "tcsi.teldata" && name == "pa")
            {
                float angle = snapshot.m_elements[name].toFloat();
                (*m_dict)["rtimv.north.angle"].setBlob(&angle, sizeof(angle));
            }
        }
    }
    m_properties[key] = snapshot;
}

bool indiDictionary::sendCommands(const QVector<rtimvCommand> &commands, QString &error)
{
    std::lock_guard<std::mutex> guard(m_clientMutex);
    if(!commandConnected()) { error = "INDI disconnected"; return false; }
    std::vector<pcf::IndiProperty> outgoing;
    for(const auto &command : commands)
    {
        auto snapshot = propertySnapshot(command.m_device, command.m_property);
        if(!snapshot.m_valid || !snapshot.m_writable || snapshot.m_type != command.m_type || command.m_elements.empty() || command.m_type == rtimvPropertyType::text)
        { error = "INDI property unavailable, read-only, or wrong type"; return false; }
        pcf::IndiProperty property(command.m_type == rtimvPropertyType::number ? pcf::IndiProperty::Number : pcf::IndiProperty::Switch,
                                   command.m_device.toStdString(), command.m_property.toStdString());
        for(auto it = command.m_elements.begin(); it != command.m_elements.end(); ++it)
        {
            if(!snapshot.m_elements.contains(it.key())) { error = "Unknown INDI element"; return false; }
            std::string name = it.key().toStdString();
            property.add(pcf::IndiElement(name));
            if(command.m_type == rtimvPropertyType::number)
            {
                bool ok;
                double number = it.value().toDouble(&ok);
                const int type = it.value().userType();
                if(!ok || !std::isfinite(number) ||
                   (type != QMetaType::Double && type != QMetaType::Float && type != QMetaType::Int &&
                    type != QMetaType::UInt && type != QMetaType::LongLong && type != QMetaType::ULongLong))
                { error = "Invalid INDI number"; return false; }
                property[name] = number;
            }
            else
            {
                if(it.value().userType() != QMetaType::Bool) { error = "Invalid INDI switch"; return false; }
                property[name].setSwitchState(it.value().toBool() ? pcf::IndiElement::On : pcf::IndiElement::Off);
            }
        }
        outgoing.push_back(property);
    }
    try
    {
        for(const auto &property : outgoing) writeCommand(property);
        return true;
    }
    catch(const std::exception &e) { error = QString::fromUtf8(e.what()); }
    catch(...) { error = "INDI send failed"; }
    invalidateConnection();
    if(m_client) m_client->quitProcess();
    return false;
}
