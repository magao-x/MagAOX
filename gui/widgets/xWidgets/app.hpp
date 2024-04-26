#ifndef xqt_app_hpp
#define xqt_app_hpp

#include <QApplication>
#include <QFile>
#include <QTextStream>

#include <mx/app/application.hpp>
#include <mx/sys/environment.hpp>

#include "../../../libMagAOX/common/environment.hpp"
#include "../../../libMagAOX/common/paths.hpp"


#include "xWidget.hpp"
#include "multiIndiManager.hpp"

namespace xqt 
{

template<class widgetT>
class app : public xWidget, public mx::app::application
{
public:

    std::string m_styleSheet {":/magaox.qss"};

    std::string m_deviceName;
    std::string m_ipAddress {"127.0.0.1"};
    int m_port {7624};

    app()
    {
        std::string basePath = mx::sys::getEnv(MAGAOX_env_path);
        if(basePath == "")
        {
            basePath = MAGAOX_path;
        }
        
        std::string configDir = mx::sys::getEnv(MAGAOX_env_config);
        if(configDir == "")
        {
            configDir = MAGAOX_configRelPath;
        }

        m_configPathCLBase = basePath += "/" + configDir + "/";

        //Lack of config files is ignored
        m_requireConfigPathGlobal = false;
        m_requireConfigPathUser = false;
        m_requireConfigPathLocal = false;

        //Have to keep config for later processing by device
        m_preserveConfig = true;
    }


    /// Set up the configuration system
    /** After specifying the "user config" of mx::application this calls mx::application::setup(argc,argv).
      */
    virtual void setup( int argc,    ///< [in] standard command line result specifying number of argumetns in argv
                        char ** argv ///< [in] standard command line result containing the arguments.
                      )
    {
        ///\todo We should config search path regardless of number of arguments.
        if(argc == 2)
        {
            m_deviceName = argv[1];
            if(m_deviceName.find(".conf") != std::string::npos || m_deviceName.find('/') != std::string::npos)
            {
                throw std::invalid_argument("Can only pass device name, not config file, as non-option argument.  Use -c <config_file> instead.");
            }

            m_configPathGlobal = m_configPathCLBase + "/guis.conf";
            m_configPathUser = m_configPathCLBase + "/gui_" + m_deviceName + ".conf";
        }
            
        application::setup(argc, argv);

    }

    virtual void setupConfig()
    {
        config.add("gui.stylesheet", "", "gui.styleSheet", mx::app::argType::Required, "gui", "styleSheet", false, "string", "Path to a Qt style sheet to replace the default.");
        
        config.add("indi.device", "d", "indi.device", mx::app::argType::Required, "indi", "device", false, "string", "INDI device name.");
        config.add("indi.ip", "a", "indi.ip", mx::app::argType::Required, "indi", "ip", false, "string", "INDI IP address or host name.  Default is 127.0.0.1.");
        config.add("indi.port", "p", "indi.port", mx::app::argType::Required, "indi", "port", false, "int", "INDI IP port.  Default is 7624.");

       widgetT::setupConfig(config);
    }

    virtual void loadConfig()
    {
        config(m_styleSheet, "gui.styleSheet");

        config(m_deviceName, "indi.device");
        config(m_ipAddress, "indi.ip");
        config(m_port, "indi.port");
    }

    virtual int execute()
    {
        if(m_deviceName == "")
        {
            throw std::invalid_argument("No device name specified.  Either call this with a single argument specifying it, or use -c config_file.conf with '[indi] device='");
        }

        // set stylesheet
        QFile file(m_styleSheet.c_str());
        file.open(QFile::ReadOnly | QFile::Text);
        QTextStream stream(&file);
        qApp->setStyleSheet(stream.readAll());

        multiIndiManager mgr(m_deviceName, m_ipAddress, m_port);

        widgetT device(m_deviceName);

        device.loadConfig(config);
        config.clear(); //m_preserveConfig was set to get to this point, now we don't need it.

        mgr.addSubscriber(&device);

        mgr.activate();
      
        device.show();
   
        int rv = qApp->exec();
   
        return rv;
    }
};

}

#endif //xqt_app_hpp
