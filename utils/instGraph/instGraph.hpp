/** \file instGraph.hpp
  * \brief The MagAO-X Instrument Graph header file
  *
  * \ingroup instGraph_files
  */

#ifndef xInstGraph_hpp
#define xInstGraph_hpp

#include "/usr/local/include/instGraph/instGraphXML.hpp"
using namespace ingr;

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


class xigNode
{
public:

    std::string m_name;

    std::set<std::string> m_keys;

    instGraphXML * m_parentGraph {nullptr};

    void key( const std::string & nkey )
    {
        m_keys.insert(nkey);
    }

    ingr::instNode * m_node {nullptr};

    virtual void handleSetProperty( const pcf::IndiProperty & ipRecv ) = 0;
};

class pwrOnOffNode : public xigNode
{

public:

    std::string m_pwrKey;
    int m_pwrState {-1};

    virtual void handleSetProperty( const pcf::IndiProperty & ipRecv )
    {
        if( ipRecv.createUniqueKey() != m_pwrKey ) return;

        if(!ipRecv.find("state")) return;

        std::cerr << "got: " << ipRecv.createUniqueKey() << "\n";
        if(ipRecv["state"].get<std::string>() == "On") return toggleOn();
        else return toggleOff();
    }

    virtual void toggleOn()
    {
        m_pwrState = 1;

        togglePutsOn();
        
    }

    virtual void togglePutsOn()
    {
        if(m_node == nullptr) return;

        for(auto && iput : m_node->inputs())
        {
            iput.second->state(ingr::putState::on);
        } 

        for(auto && oput : m_node->outputs())
        {
            oput.second->state(ingr::putState::on);
        } 
    }

    virtual void toggleOff()
    {
        m_pwrState = 1;

        togglePutsOff();
    }

    virtual void togglePutsOff()
    {
        if(m_node == nullptr) return;

        for(auto && iput : m_node->inputs())
        {
            iput.second->state(ingr::putState::off);
        } 

        for(auto && oput : m_node->outputs())
        {
            oput.second->state(ingr::putState::off);
        }
    }
};

class stdMotionNode : public pwrOnOffNode
{

public:

    std::string m_device;
    std::string m_presetPrefix {"preset"};
    
    std::string m_curVal;

    std::vector<std::string> m_presetPutName {"out"}; 

    ingr::ioDir m_presetDir {ingr::ioDir::output};

    virtual void handleSetProperty( const pcf::IndiProperty & ipRecv )
    {
        pwrOnOffNode::handleSetProperty(ipRecv);

        if(ipRecv.createUniqueKey() ==  (m_device + "." + m_presetPrefix + "Name") )
        {
            std::cerr << "got " << ipRecv.createUniqueKey() << "\n";

            if(m_node != nullptr)
            {
                for(auto && it : ipRecv.getElements() )
                {
                    if(it.second.getSwitchState() == pcf::IndiElement::On)
                    {
                        bool changed;
                        if(m_curVal != it.second.getName())
                        {
                            changed = true;
                        }

                        m_curVal = it.second.getName();
                        
                        if(changed) 
                        {
                            if(m_curVal == "none")
                            {
                                togglePutsOff();
                            }
                            else 
                            {   
                                togglePutsOn();
                            }
                        }
                    }
                }
            
                /*else 
                {
                    if(m_node->auxDataValid())
                    {
                        if(m_presetPutName.size() == 1)
                        {
                            m_parentGraph->valuePut(m_name, m_presetPutName[0], m_presetDir, "off");
                        }
                        else 
                        {
                        }
                    }
                }*/
                
            }
        }
    }

    virtual void togglePutsOn()
    {
        std::cerr << "toggle on\n";
        if(m_pwrState == 1)
        {
            if(m_presetPutName.size() == 1)
            {
                if(m_node->auxDataValid())
                {
                    if(m_parentGraph)
                    {
                        m_parentGraph->valuePut(m_name, m_presetPutName[0], m_presetDir, m_curVal);
                    }
                }
                pwrOnOffNode::togglePutsOn();
            }
            else 
            {
                for(auto s : m_presetPutName)
                {
                    ingr::instIOPut * pptr;
                    
                    try 
                    {
                        if(m_presetDir == ingr::ioDir::input )
                        {
                            pptr = m_node->input(s);
                        }
                        else 
                        {
                            pptr = m_node->output(s);
                        }
                    } 
                    catch(...)
                    {
                        return;
                    }

                    if(s == m_curVal)
                    {
                        pptr->state(ingr::putState::on);
                    }
                    else 
                    {
                        pptr->state(ingr::putState::off);
                    }
                }
                std::cerr << "changing state\n";
                m_parentGraph->stateChange();
            } Indent: Namespace Contents
        }

        return; //we don't automatically toggle puts on upon power on.
    }

    virtual void togglePutsOff()
    {
        std::cerr << "toggle off\n";
        if(m_node != nullptr)
        {
            if(m_node->auxDataValid())
            {
                if(m_presetPutName.size() == 1)
                {
                    m_parentGraph->valuePut(m_name, m_presetPutName[0], m_presetDir, "off");
                }
                else 
                {
                }
            }
        }
            
        pwrOnOffNode::togglePutsOff();
    }

};


class xInstGraph;

class xInstGraphIndiClient : public pcf::IndiClient
{
public: 

    xInstGraph * m_parent {nullptr};

    xInstGraphIndiClient( const std::string & name,
                          const std::string & version,
                          const std::string & something,
                          const std::string & ipAddr,
                          int port,
                          xInstGraph * parent 
                        ) : IndiClient(name, version, something, ipAddr, port), m_parent(parent)
    {

    }

    void execute()
    {
        processIndiRequests(false);
    }

    virtual void handleSetProperty(const pcf::IndiProperty & ipRecv);
    
    virtual void handleDefProperty(const pcf::IndiProperty & ipRecv);

};


/// The MagAO-X xxxxxxxx
/** 
  * \ingroup instGraph
  */
class xInstGraph : public mx::app::application
{

   //Give the test harness access.
   friend class instGraph_test;

protected:

   /** \name Configurable Parameters
     *@{
     */
   
   //here add parameters which will be config-able at runtime
   
   ///@}

    ingr::instGraphXML m_graph;

    std::map<std::string, xigNode *> m_nodes;
    std::multimap<std::string, xigNode *> m_nodeHandleSets;

    xInstGraphIndiClient * m_client {nullptr};

public:
   /// Default c'tor.
   xInstGraph();

   /// D'tor, declared and defined for noexcept.
   ~xInstGraph() noexcept
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

xInstGraph::xInstGraph() 
{
   
   return;
}

void xInstGraph::setupConfig()
{
}

int xInstGraph::loadConfigImpl( mx::app::appConfigurator & _config )
{
   static_cast<void>(_config);
   
   return 0;
}

void xInstGraph::loadConfig()
{
   loadConfigImpl(config);
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

int xInstGraph::appStartup()
{
    std::string emsg;
    if( m_graph.loadXMLFile(emsg, "magaox.drawio") < 0 )
    {
        std::cerr << emsg << "\n";
        return -1;
    }

    pwrOnOffNode * nn = new pwrOnOffNode;
    nn->m_name = "source";
    nn->m_pwrKey = "pdu0.source";
    nn->key(nn->m_pwrKey);
    nn->m_node = m_graph.node(nn->m_name);
    m_nodes.insert({nn->m_node->name(), nn});

    stdMotionNode * n2 = new stdMotionNode;
    n2->m_parentGraph = &m_graph;
    n2->m_name = "fwtelsim";
    n2->m_pwrKey = "dcdu1.fwtelsim";
    n2->key(n2->m_pwrKey);
    n2->m_device = "fwtelsim";
    n2->m_presetPrefix = "filter";
    n2->key(n2->m_device + "." + n2->m_presetPrefix + "Name");
    n2->m_node = m_graph.node("fwtelsim");
    m_nodes.insert({n2->m_node->name(), n2});

    stdMotionNode * n3 = new stdMotionNode;
    n3->m_parentGraph = &m_graph;
    n3->m_name = "pickoff";
    n3->m_pwrKey = "pdu2.stagepickoff";
    n3->key(n3->m_pwrKey);
    n3->m_device = "stagepickoff";
    n3->m_presetPrefix = "preset";
    n3->key(n3->m_device + "." + n3->m_presetPrefix + "Name");
    n3->m_node = m_graph.node("pickoff");
    n3->m_presetDir = ingr::ioDir::input;
    n3->m_presetPutName = {"lab", "tel"};
    m_nodes.insert({n3->m_node->name(), n3});

    return 0;
}

int xInstGraph::execute()
{
    appStartup();

    //std::cout << ingr::beamState2String(m_beam_source2fwtelsim->state()) << "\n";

    bool m_shutdown = false;
    bool m_connected = false;

    while(!m_shutdown)
    {
        while(!m_connected && !m_shutdown)
        {
            try 
            {
                m_client = new xInstGraphIndiClient("instGraph", "1.7", "1,7", "127.0.0.1", 7624, this);
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

int xInstGraph::appShutdown()
{
   return 0;
}

void xInstGraph::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    auto range = m_nodeHandleSets.equal_range(ipRecv.createUniqueKey());

    for(auto it=range.first; it != range.second; ++it)
    {
        it->second->handleSetProperty(ipRecv);
    }
}


void xInstGraphIndiClient::handleSetProperty(const pcf::IndiProperty & ipRecv)
{
    if(m_parent)
    {
        m_parent->handleSetProperty(ipRecv);
    }
}

void xInstGraphIndiClient::handleDefProperty(const pcf::IndiProperty & ipRecv)
{
    handleSetProperty(ipRecv);
}

#endif //xInstGraph_hpp
