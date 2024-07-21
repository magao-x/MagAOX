/** \file instGraph.hpp
  * \brief The MagAO-X Instrument Graph header file
  *
  * \ingroup instGraph_files
  */

#ifndef instGraph_hpp
#define instGraph_hpp

#include <instGraph/instGraph.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch

/** \defgroup instGraph
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup instGraph_files
  * \ingroup instGraph
  */


class xigNode : public ingr::instNode 
{
public:

    std::set<std::string> m_keys;

    void key( const std::string & nkey )
    {
        m_keys.insert(nkey);
    }

    virtual void handleSetProperty( const pcf::IndiProperty & ipRecv ) = 0;
};

class pwrOnOffNode : public xigNode
{

public:

    std::string m_pwrKey;
    
    virtual void handleSetProperty( const pcf::IndiProperty & ipRecv )
    {
        if(m_keys.size() != 1) return;

        if(m_keys.count(ipRecv.createUniqueKey()) == 0) return;

        if(!ipRecv.find("state")) return;

        if(ipRecv["state"].get<std::string>() == "On") return toggleOn();
        else return toggleOff();
    }

    void toggleOn()
    {
        std::cerr << m_name << " toggle on\n";
        for(auto && iput : m_inputs)
        {
            iput.second->state(ingr::putState::on);
        } 

        for(auto && oput : m_outputs)
        {
            oput.second->state(ingr::putState::on);
        } 

    }

    void toggleOff()
    {
        std::cerr << m_name << " toggle off\n";

        for(auto && iput : m_inputs)
        {
            iput.second->state(ingr::putState::off);
        } 

        for(auto && oput : m_outputs)
        {
            oput.second->state(ingr::putState::off);
        } 

    }
};

class instGraph;

class instGraphIndiClient : public pcf::IndiClient
{
public: 

    instGraph * m_parent {nullptr};

    instGraphIndiClient( const std::string & name,
                         const std::string & version,
                         const std::string & something,
                         const std::string & ipAddr,
                         int port,
                         instGraph * parent 
                      ) : IndiClient(name, version, something, ipAddr, port), m_parent(parent)
    {

    }

    void execute()
    {
        std::cerr << __LINE__ << "\n";
        processIndiRequests(false);
    }

    virtual void handleSetProperty(const pcf::IndiProperty & ipRecv);
    
    virtual void handleDefProperty(const pcf::IndiProperty & ipRecv);

};


/// The MagAO-X xxxxxxxx
/** 
  * \ingroup instGraph
  */
class instGraph : public mx::app::application,  public ingr::instGraph
{

   //Give the test harness access.
   friend class instGraph_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}

    ingr::instBeam * m_beam_fwcamsim2camsim {nullptr};

    std::map<std::string, xigNode *> m_nodes;
    std::multimap<std::string, xigNode *> m_nodeHandleSets;

    instGraphIndiClient * m_client {nullptr};

public:
   /// Default c'tor.
   instGraph();

   /// D'tor, declared and defined for noexcept.
   ~instGraph() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for instGraph.
   /** 
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int execute();

   /// Shutdown the app.
   /** 
     *
     */
   virtual int appShutdown();

   virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the set property message.*/);

};

instGraph::instGraph() 
{
   
   return;
}

void instGraph::setupConfig()
{
    std::cerr << __LINE__ << "\n";
}

int instGraph::loadConfigImpl( mx::app::appConfigurator & _config )
{
   static_cast<void>(_config);
   
   return 0;
}

void instGraph::loadConfig()
{
   loadConfigImpl(config);
   std::cerr << __LINE__ << "\n";
}

std::string deviceFromKey(const std::string & key)
{
    size_t dot = key.find('.');
    if(dot == std::string::npos) return "";

    return key.substr(0, dot);
}

std::string nameFromKey(const std::string & key)
{
    size_t dot = key.find('.');
    if(dot == std::string::npos) return "";

    return key.substr(dot+1);
}

int instGraph::appStartup()
{
    std::cerr << __LINE__ << "\n";

    pwrOnOffNode * nn = new pwrOnOffNode;
    nn->name("fwcamsim");
    nn->key("pdu0.fwcamsim");
    m_nodes.insert({nn->name(), nn});

    nn = new pwrOnOffNode;
    nn->name("camsim");
    nn->key("pdu0.camsim");
    m_nodes.insert({nn->name(), nn});

    /*for(auto it=m_nodes.begin(); it != m_nodes.end(); ++it)
    {
        for(auto kit=it->second->m_keys.begin(); kit != it->second->m_keys.end(); ++kit)
        {
            m_nodeHandleSets.insert({*kit, it->second});
            pcf::IndiProperty * ip = new pcf::IndiProperty;
            std::string devname = deviceFromKey(*kit);
            std::string propname = nameFromKey(*kit);
            ip->setDevice(devname);
            ip->setName(propname);
            registerIndiPropertySet( *ip, devname, propname, st_setCallBack_nodeProperties);
        }
    }*/

    m_beam_fwcamsim2camsim = new ingr::instBeam;
    m_beam_fwcamsim2camsim->name("fwcamsim2camsim");
    m_beams.insert({m_beam_fwcamsim2camsim->name(), m_beam_fwcamsim2camsim});

    ingr::instIOPut * newput = new ingr::instIOPut({m_nodes["fwcamsim"],ingr::ioDir::output,"out",ingr::putType::light,m_beam_fwcamsim2camsim});
    m_nodes["fwcamsim"]->addIOPut(newput);

    newput = new ingr::instIOPut({m_nodes["camsim"],ingr::ioDir::input,"in",ingr::putType::light,m_beam_fwcamsim2camsim});
    m_nodes["camsim"]->addIOPut(newput);


/*    m_node_fwtelsim = new pwrOnOffNode;
    m_node_fwtelsim->name("fwtelsim");

    m_node_source->m_device = "pdu0";
    m_node_source->m_property = "source";

    REG_INDI_SETPROP(m_indiP_sourcePower, "pdu0", "source");

    m_node_fwtelsim->m_device = "usbdu0";
    m_node_fwtelsim->m_property = "fwtelsim";

    REG_INDI_SETPROP(m_indiP_fwtelsimPower, "usbdu0", "fwtelsim");

    m_beam_source2fwtelsim = new ingr::instBeam;
    m_beam_source2fwtelsim->name("source2fwtelsim");

    ingr::instIOPut * newput = new ingr::instIOPut({m_node_source,ingr::ioDir::output,"out",ingr::putType::light,m_beam_source2fwtelsim});
    m_node_source->addIOPut(newput);

    newput = new ingr::instIOPut({m_node_fwtelsim,ingr::ioDir::input,"in",ingr::putType::light,m_beam_source2fwtelsim});
    m_node_fwtelsim->addIOPut(newput);
*/

    
    return 0;
}

int instGraph::execute()
{
    appStartup();

    //std::cout << ingr::beamState2String(m_beam_source2fwtelsim->state()) << "\n";

    std::cerr << __LINE__ << "\n";

    bool m_shutdown = false;
    bool m_connected = false;

    while(!m_shutdown)
    {
        while(!m_connected && !m_shutdown)
        {
            try 
            {
                m_client = new instGraphIndiClient("instGraph", "1.7", "1,7", "127.0.0.1", 7624, this);
            }
            catch(...) 
            {
                sleep(1);
                continue;
            }

            m_client->activate();

            sleep(2);

            if(m_client->getQuitProcess())
            {
                m_client->quitProcess();
                m_client->deactivate();
                sleep(2);
                delete m_client;
                m_client = nullptr;
                continue;
            }

            for(auto it=m_nodes.begin(); it != m_nodes.end(); ++it)
            {
                for(auto kit=it->second->m_keys.begin(); kit != it->second->m_keys.end(); ++kit)
                {
                    m_nodeHandleSets.insert({*kit, it->second});
                    std::string devname = deviceFromKey(*kit);
                    std::string propname = nameFromKey(*kit);
                    
                    pcf::IndiProperty ip;
                    ip.setDevice(devname);
                    ip.setName(propname);
                    m_client->sendGetProperties( ip );

                    if(m_client->getQuitProcess())
                    {
                        break;
                    }
                }

                if(m_client->getQuitProcess())
                {
                    break;
                }
            }

            if(m_client->getQuitProcess())
            {
                m_client->quitProcess();
                m_client->deactivate();
                sleep(2);
                delete m_client;
                m_client = nullptr;
                continue;
            }

            m_connected = true;
        }

        while(m_connected && !m_shutdown)
        {
            std::cout << "fwcamsim2camsim state   = " << ingr::beamState2String(beam("fwcamsim2camsim")->state()) << "\n  ";

            mx::sys::milliSleep(1000);

            if(m_client->getQuitProcess())
            {
                m_connected = false;
                m_client->quitProcess();
                m_client->deactivate();
                sleep(2);
                delete m_client;
                m_client = nullptr;
            }
        }
    }

    return 0;
}

int instGraph::appShutdown()
{
   return 0;
}

void instGraph::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    auto range = m_nodeHandleSets.equal_range(ipRecv.createUniqueKey());

    for(auto it=range.first; it != range.second; ++it)
    {
        it->second->handleSetProperty(ipRecv);
    }
}


void instGraphIndiClient::handleSetProperty(const pcf::IndiProperty & ipRecv)
{
    std::cerr << __LINE__ << "\n";
    if(m_parent)
    {
        m_parent->handleSetProperty(ipRecv);
    }
}

void instGraphIndiClient::handleDefProperty(const pcf::IndiProperty & ipRecv)
{
    handleSetProperty(ipRecv);
}

#endif //instGraph_hpp
