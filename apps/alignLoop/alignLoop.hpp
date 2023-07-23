/** \file alignLoop.hpp
  * \brief The MagAO-X XXXXXX header file
  *
  * \ingroup alignLoop_files
  */

#ifndef alignLoop_hpp
#define alignLoop_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup alignLoop
  * \brief The XXXXXX application to do YYYYYYY
  *
  * <a href="../handbook/operating/software/apps/XXXXXX.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup alignLoop_files
  * \ingroup alignLoop
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X xxxxxxxx
/** 
  * \ingroup alignLoop
  */
class alignLoop : public MagAOXApp<true>, public dev::shmimMonitor<alignLoop>
{
   //Give the test harness access.
   friend class alignLoop_test;

   friend class dev::shmimMonitor<alignLoop>;

public:
   //The base shmimMonitor type
   typedef dev::shmimMonitor<alignLoop> shmimMonitorT;

protected:

   /** \name Configurable Parameters
     *@{
     */

   std::vector<std::string> m_ctrlDevices;
   std::vector<std::string> m_ctrlProperties;
   std::vector<std::string> m_ctrlCurrents;
   std::vector<std::string> m_ctrlTargets;

   std::vector<float> m_currents;

   std::vector<pcf::IndiProperty> m_indiP_ctrl;

   std::string m_intMatFile;

   std::vector<float> m_defaultGains;

   std::string m_upstreamDevice;
   std::string m_upstreamProperty {"loop_state"};
   bool m_upstreamFollowClosed {false};

   
   ///@}

   mx::improc::eigenImage<float> m_intMat;

   mx::improc::eigenImage<float> m_measurements;
   mx::improc::eigenImage<float> m_commands;

   float m_ggain {0};

   std::vector<float> m_gains;

   bool m_ctrlEnabled {false};

public:
   /// Default c'tor.
   alignLoop();

   /// D'tor, declared and defined for noexcept.
   ~alignLoop() noexcept
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

   /// Implementation of the FSM for alignLoop.
   /** 
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.
   /** 
     *
     */
   virtual int appShutdown();

   int toggleLoop( bool onoff );

   // shmimMonitor interface:
   int allocate( const dev::shmimT &);
   
   int processImage( void* curr_src,
                     const dev::shmimT &
                    );

   int sendCommands(std::vector<float> & commands);
   
   //INDI 

   pcf::IndiProperty m_indiP_ggain;
   pcf::IndiProperty m_indiP_ctrlEnabled;

   INDI_NEWCALLBACK_DECL(alignLoop, m_indiP_ggain);
   INDI_NEWCALLBACK_DECL(alignLoop, m_indiP_ctrlEnabled);

   static int st_setCallBack_ctrl( void * app, const pcf::IndiProperty &ipRecv)
   {
      return static_cast<alignLoop *>(app)->setCallBack_ctrl(ipRecv);
   }

   int setCallBack_ctrl(const pcf::IndiProperty &ipRecv);

   int m_upstreamState {0};
   pcf::IndiProperty m_indiP_upstream; ///< Property used to report the loop state
   
   INDI_SETCALLBACK_DECL(alignLoop, m_indiP_upstream);
};

alignLoop::alignLoop() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   
   return;
}

void alignLoop::setupConfig()
{
   shmimMonitorT::setupConfig(config);

   config.add("ctrl.devices", "", "ctrl.devices", argType::Required, "ctrl", "devices", false, "string", "Device names of the controller(s) (one per element).");
   config.add("ctrl.properties", "", "ctrl.properties", argType::Required, "ctrl", "properties", false, "string", "Properties of the ctrl devices to which to give the commands. One per element");
   config.add("ctrl.currents", "", "ctrl.currents", argType::Required, "ctrl", "currents", false, "vector<string>", "current elements of the properties on which base the commands.");
   config.add("ctrl.targets", "", "ctrl.targets", argType::Required, "ctrl", "targets", false, "vector<string>", "target elements of the properties to which to send the commands.");

   config.add("loop.intMat", "", "loop.intMat", argType::Required, "loop", "intMat", false, "string", "file name of the interaction matrix.");
   config.add("loop.gains", "", "loop.gains", argType::Required, "loop", "gains", false, "vector<float>", "default loop gains.  If single number, it is applied to all axes.");
   config.add("loop.upstream", "", "loop.upstream", argType::Required, "loop", "upstream", false, "string", "Upstream loop device name.  This loop will open, and optionally close, with the upstream loop.  Default none.");
   config.add("loop.upstreamProperty", "", "loop.upstreamProperty", argType::Required, "loop", "upstreamProperty", false, "string", "Property of upstream loop device to follow.  Must be a toggle.  Default is loop_state.");

}

int alignLoop::loadConfigImpl( mx::app::appConfigurator & _config )
{
   shmimMonitorT::loadConfig(_config);

   _config(m_ctrlDevices, "ctrl.devices");
   _config(m_ctrlProperties, "ctrl.properties");
   _config(m_ctrlCurrents, "ctrl.currents");
   _config(m_ctrlTargets, "ctrl.targets");
   _config(m_intMatFile, "loop.intMat");
   _config(m_defaultGains, "loop.gains");
   _config(m_upstreamDevice, "loop.upstream");
   _config(m_upstreamProperty, "loop.upstreamProperty");
   return 0;
}

void alignLoop::loadConfig()
{
   loadConfigImpl(config);
}

int alignLoop::appStartup()
{
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(m_ctrlTargets.size() != m_ctrlDevices.size())
   {
      return log<software_error, -1>({__FILE__, __LINE__, "ctrl.Targets and ctrl.devices are not the same size"});
   }

   if(m_ctrlTargets.size() != m_ctrlProperties.size())
   {
      return log<software_error, -1>({__FILE__, __LINE__, "ctrl.Targets and ctrl.properties are not the same size"});
   }

   if(m_ctrlTargets.size() != m_ctrlCurrents.size())
   {
      return log<software_error, -1>({__FILE__, __LINE__, "ctrl.Currents and ctrl.properties are not the same size"});
   }

   if(m_ctrlTargets.size() != m_defaultGains.size())
   {
      if(m_defaultGains.size()==1)
      {
         float g = m_defaultGains[0];
         m_defaultGains.resize(m_ctrlTargets.size(), g);
         log<text_log>("Setting loop.gains gains to be same size as ctrl.Targets", logPrio::LOG_NOTICE);
      }
      else
      {
         return log<software_error, -1>({__FILE__, __LINE__, "ctrl.Targets and loop.gains are not the same size"});
      }
   }

   m_gains.resize(m_defaultGains.size());
   for(size_t n=0; n < m_defaultGains.size(); ++n) m_gains[n] = m_defaultGains[n];

   createStandardIndiNumber<unsigned>( m_indiP_ggain, "loop_gain", 0, 1, 0, "%0.2f");
   m_indiP_ggain["current"] = m_ggain;
   m_indiP_ggain["target"] = m_ggain;  
   if( registerIndiPropertyNew( m_indiP_ggain, INDI_NEWCALLBACK(m_indiP_ggain)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   createStandardIndiToggleSw( m_indiP_ctrlEnabled, "loop_state");  
   if( registerIndiPropertyNew( m_indiP_ctrlEnabled, INDI_NEWCALLBACK(m_indiP_ctrlEnabled)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   m_currents.resize(m_ctrlDevices.size(), -1e15);

   m_indiP_ctrl.resize(m_ctrlDevices.size());

   for(size_t n=0; n< m_ctrlDevices.size();++n)
   {
      registerIndiPropertySet( m_indiP_ctrl[n], m_ctrlDevices[n], m_ctrlProperties[n], &st_setCallBack_ctrl);
   }
   std::string intMatPath = m_calibDir + "/" + m_intMatFile; 

   m_commands.resize(m_ctrlTargets.size(), m_ctrlTargets.size()) ;
   m_commands.setZero();

   m_intMat.resize(m_ctrlTargets.size(), m_ctrlTargets.size()) ;
   m_intMat.setZero();
   m_intMat(1,0) = 0.926;
   m_intMat(1,1) = -0.370;
   m_intMat(0,0) = 0.185;
   m_intMat(0,1) = 0.926;

/*
   mx::fits::fitsFile<float> ff;
   try
   {
      ff.read(m_intMat, intMatPath);
   }
   catch(...)
   {
      return log<software_error, -1>({__FILE__, __LINE__, "error reading loop.intMatFile.  Does it exist?"});
   }

   if(m_intMat.rows() != m_ctrlTargets.size())
   {
      return log<software_error, -1>({__FILE__, __LINE__, "interaction matrix wrong size: rows do not match sensor Targets"});
   }

   if(m_intMat.cols() != m_correctorTargets.size())
   {
      return log<software_error, -1>({__FILE__, __LINE__, "interaction matrix wrong size: cols do not match corrector Targets"});
   }
*/

   //Get the loop state for managing offloading
   if(m_upstreamDevice != "")
   {
      REG_INDI_SETPROP(m_indiP_upstream, m_upstreamDevice, m_upstreamProperty);
   }

   return 0;
}

int alignLoop::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   state(stateCodes::OPERATING);

   return 0;
}

int alignLoop::appShutdown()
{
   shmimMonitorT::appShutdown();

   return 0;
}

int alignLoop::toggleLoop(bool onoff)
{
   if(!m_ctrlEnabled && onoff) //not enabled so change
   {      
      m_ctrlEnabled = true;
      log<loop_closed>();
      updateSwitchIfChanged(m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::On, INDI_OK);
      return 0;
   }

   if(m_ctrlEnabled && !onoff)
   {
      m_ctrlEnabled = false;
      log<loop_open>();
      updateSwitchIfChanged(m_indiP_ctrlEnabled, "toggle", pcf::IndiElement::Off, INDI_IDLE);
   
      return 0;
   }

   return 0;
}

inline
int alignLoop::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy);
   
   std::lock_guard<std::mutex> guard(m_indiMutex);

   std::cerr << shmimMonitorT::m_width << shmimMonitorT::m_height << "\n";
   m_measurements.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
   
   return 0;
}

inline
int alignLoop::processImage( void* curr_src,
                            const dev::shmimT & dummy
                           )
{
   static_cast<void>(dummy);
   
   for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
   {
      m_measurements.data()[nn] = ((float*)curr_src) [nn];
   }



   std::cout << "measurements: ";
   for(int cc = 0; cc < m_measurements.rows(); ++cc)
   {
      std::cout << m_measurements(cc,0) << " ";
   }
   std::cout << "\n";

   if(m_currents[0] < -1e14) return 0;

   m_commands.matrix() = m_intMat.matrix() * m_measurements.matrix();

   std::cout << "delta commands:    ";
   for(int cc = 0; cc < m_measurements.rows(); ++cc)
   {
      std::cout << -m_commands(cc,0) << " ";
   }
   std::cout << "\n";

   std::vector<float> commands;
   commands.resize(m_measurements.rows());

   std::cout << "commands:    ";
   for(int cc = 0; cc < m_measurements.rows(); ++cc)
   {
      commands[cc] = m_currents[cc] - m_ggain*m_gains[cc]*m_commands(cc,0);
      std::cout << commands[cc] << " ";
   }
   std::cout << "\n";
  
   //And send commands.
   if(m_ctrlEnabled)
   {
      return sendCommands(commands);
   }
   else 
   {
      return 0;
   }
}

inline
int alignLoop::sendCommands(std::vector<float> & commands)
{
   for(size_t n=0; n < m_ctrlDevices.size(); ++n)
   {
      pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
      ip.setDevice(m_ctrlDevices[n]);
      ip.setName(m_ctrlProperties[n]);
      ip.add(pcf::IndiElement(m_ctrlTargets[n]));
      ip[m_ctrlTargets[n]] = commands[n];
   
      sendNewProperty(ip);
   }

   return 0;
}

INDI_NEWCALLBACK_DEFN(alignLoop, m_indiP_ggain)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_ggain.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_ggain, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_ggain = target;
   
   updateIfChanged(m_indiP_ggain, "current", m_ggain);
   updateIfChanged(m_indiP_ggain, "target", m_ggain);
   
   log<text_log>("set global gain to " + std::to_string(m_ggain), logPrio::LOG_NOTICE);
   
   return 0;
}

INDI_NEWCALLBACK_DEFN(alignLoop, m_indiP_ctrlEnabled)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_ctrlEnabled.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }

   //switch is toggled to on
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
   {
      return toggleLoop(true);
   }

   //switch is toggle to off
   if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      return toggleLoop(false);
   }
   
   return 0;
}

inline
int alignLoop::setCallBack_ctrl(const pcf::IndiProperty &ipRecv)
{
   for(size_t n = 0; n < m_ctrlDevices.size(); ++n)
   {
      if( ipRecv.getDevice() == m_ctrlDevices[n])
      {
         if(ipRecv.getName() == m_ctrlProperties[n] && ipRecv.find(m_ctrlCurrents[n]))
         {
            m_currents[n] = ipRecv[m_ctrlCurrents[n]].get<float>();
         }
      }
   }

   return 0;
}

INDI_SETCALLBACK_DEFN(alignLoop, m_indiP_upstream)(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_upstream.getName())
   {
      return log<software_error>({__FILE__,__LINE__, "wrong INDI property received"});
   }

   if(!ipRecv.find("toggle")) return 0;

   if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On && m_upstreamFollowClosed)
   {
      std::cerr << "upstream on\n";
      return toggleLoop(true);
   }
   else if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
   {
      std::cerr << "upstream off\n";
      return toggleLoop(false);
   }

   return 0;
}


} //namespace app
} //namespace MagAOX

#endif //alignLoop_hpp
