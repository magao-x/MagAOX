/** \file picoMotorCtrl.hpp
  * \brief The MagAO-X Pico Motor Controller header file
  *
  * \ingroup picoMotorCtrl_files
  */

#ifndef picoMotorCtrl_hpp
#define picoMotorCtrl_hpp

#include <map>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup picoMotorCtrl
  * \brief The Pico Motor Controller application.
  *
  * Controls a multi-channel Newport pico motor controller.  Each motor gets its own thread.
  * 
  * <a href="../handbook/operating/software/apps/picoMotorCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup picoMotorCtrl_files
  * \ingroup picoMotorCtrl
  */



namespace MagAOX
{
namespace app
{

/** MagAO-X application to control a multi-channel Newport Picomotor Controller.
  *
  * \todo need to recognize signals in tty polls and not return errors, etc.
  * \todo need to implement an onDisconnect() to update values to unknown indicators.
  * \todo need a frequency-dependent max amp facility.
  * \todo convert to ioDevice
  * \todo need telnet device, with optional username/password.
  * 
  */
class picoMotorCtrl : public MagAOXApp<>, public dev::ioDevice, public dev::telemeter<picoMotorCtrl>
{
   
   friend class dev::telemeter<picoMotorCtrl>;
   
   typedef dev::telemeter<picoMotorCtrl> telemeterT;

   typedef long posT;
   
   struct motorChannel
   {
      picoMotorCtrl * m_parent {nullptr}; ///< A pointer to this for thread starting.
      
      std::string m_name; ///< The name of this channel, from the config section
      
      std::vector<std::string> m_presetNames;
      std::vector<posT> m_presetPositions;
      
      int m_channel {-1}; ///< The number of this channel, where the motor is plugged in
      
      posT m_currCounts {0}; ///< The current counts, the cumulative position
      
      bool m_doMove {false}; ///< Flag indicating that a move is requested.
      bool m_moving {false}; ///< Flag to indicate that we are actually moving
      
      pcf::IndiProperty m_property;
      pcf::IndiProperty m_indiP_presetName;
      
      std::thread * m_thread {nullptr}; ///< Thread for managing this channel.  A pointer to allow copying, but must be deleted in d'tor of parent.
       
      bool m_threadInit {true}; ///< Thread initialization flag.  
      
      pid_t m_threadID {0}; ///< The ID of the thread.
   
      pcf::IndiProperty m_threadProp; ///< The property to hold the thread details.
   
      motorChannel( picoMotorCtrl * p /**< [in] The parent point to set */) : m_parent(p)
      {
         m_thread = new std::thread;
      }
      
      motorChannel( picoMotorCtrl * p,     ///< [in] The parent point to set
                    const std::string & n, ///< [in] The name of this channel
                    int ch                 ///< [in] The number of this channel
                  ) : m_parent(p), m_name(n), m_channel(ch)
      {
         m_thread = new std::thread;
      }
      
   };
   
   typedef std::map<std::string, motorChannel> channelMapT;
   
   /** \name Configurable Parameters
     * @{
     */

   std::string m_deviceAddr; ///< The device address
   std::string m_devicePort {"23"}; ///< The device port
   
   int m_nChannels {4}; ///< The number of motor channels total on the hardware.  Number of attached motors inferred from config.
   
   ///@}
   
   channelMapT m_channels; ///< Map of motor names to channel.
   
   tty::telnetConn m_telnetConn; ///< The telnet connection manager

   ///Mutex for locking telnet communications.
   std::mutex m_telnetMutex;
   
   public:

   /// Default c'tor.
   picoMotorCtrl();

   /// D'tor, declared and defined for noexcept.
   ~picoMotorCtrl() noexcept;

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   /// load the configuration system results (called by MagAOXApp::setup())
   virtual void loadConfig();

   /// Startup functions
   /** Setsup the INDI vars.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM
   /** 
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Implementation of the on-power-off FSM logic
   virtual int onPowerOff();

   /// Implementation of the while-powered-off FSM
   virtual int whilePowerOff();

   /// Do any needed shutdown tasks. 
   virtual int appShutdown();
   
   /// Read the current channel counts from disk at startup
   /** Reads the counts from the file with the specified name in this apps sys directory.
     * Returns the file contents as a posT.
     */ 
   posT readChannelCounts(const std::string & chName);
   
   int writeChannelCounts( const std::string & chName,
                           posT counts 
                         );
   
   /// Channel thread starter function
   static void channelThreadStart( motorChannel * mc /**< [in] the channel to start controlling */);
   
   /// Channel thread execution function
   /** Runs until m_shutdown is true.
     */
   void channelThreadExec( motorChannel * mc );
   
/** \name INDI
     * @{
     */ 
protected:

   //declare our properties
   std::vector<pcf::IndiProperty> m_indiP_counts;
   
   
public:
   /// The static callback function to be registered for relative position requests
   /** Dispatches to the handler, which then signals the relavent thread.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_picopos( void * app, ///< [in] a pointer to this, will be static_cast-ed to this
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                    );
   
   /// The handler function for relative position requests, called by the static callback
   /** Signals the relavent thread.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_picopos( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// The static callback function to be registered for position presets
   /** Dispatches to the handler, which then signals the relavent thread.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_presetName( void * app, ///< [in] a pointer to this, will be static_cast-ed to this
                                      const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                    );
   
   /// The handler function for position presets, called by the static callback
   /** Signals the relavent thread.
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_presetName( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   ///@}
   
   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_pico * );
   
   int recordPico( bool force = false );
   ///@}
   
};

picoMotorCtrl::picoMotorCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{   
   m_powerMgtEnabled = true;
   m_telnetConn.m_prompt = "\r\n";
   return;
}

picoMotorCtrl::~picoMotorCtrl() noexcept
{
   //Wait for each channel thread to exit, then delete it.
   for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
   {
      if(it->second.m_thread != nullptr)
      {
         if(it->second.m_thread->joinable()) it->second.m_thread->join();
         delete it->second.m_thread;
      }
   }
}


void picoMotorCtrl::setupConfig()
{
   config.add("device.address", "", "device.address", argType::Required, "device", "address", false, "string", "The controller IP address.");
   config.add("device.nChannels", "", "device.nChannels", argType::Required, "device", "nChannels", false, "int", "Number of motoro channels.  Default is 4.");
   
   dev::ioDevice::setupConfig(config);
   
   telemeterT::setupConfig(config);
}

#define PICOMOTORCTRL_E_NOMOTORS   (-5)
#define PICOMOTORCTRL_E_BADCHANNEL (-6)
#define PICOMOTORCTRL_E_DUPMOTOR   (-7)
#define PICOMOTORCTRL_E_INDIREG    (-20)

int picoMotorCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   //Standard config parsing
   _config(m_deviceAddr, "device.address");
   _config(m_nChannels, "device.nChannels");
 

   // Parse the unused config options to look for motors
   std::vector<std::string> sections;

   _config.unusedSections(sections);

   if( sections.size() == 0 )
   {
      log<text_log>("No motors found in config.", logPrio::LOG_CRITICAL);

      return PICOMOTORCTRL_E_NOMOTORS;
   }

   //Now see if any unused sections have a channel keyword
   for(size_t i=0;i<sections.size(); ++i)
   {
      int channel = -1;
      _config.configUnused(channel, mx::app::iniFile::makeKey(sections[i], "channel" ) );
      if( channel == -1 )
      {
         //not a channel
         continue;
      }
      
      if(channel < 1 || channel > m_nChannels)
      {
         log<text_log>("Bad channel specificiation: " + sections[i] + " " + std::to_string(channel), logPrio::LOG_CRITICAL);

         return PICOMOTORCTRL_E_BADCHANNEL;
      }

      //Ok, valid channel.  Insert into map and check for duplicates.
      std::pair<channelMapT::iterator, bool> insert = m_channels.insert(std::pair<std::string, motorChannel>(sections[i], motorChannel(this,sections[i],channel)));
      
      if(insert.second == false)
      {
         log<text_log>("Duplicate motor specificiation: " + sections[i] + " " + std::to_string(channel), logPrio::LOG_CRITICAL);
         return PICOMOTORCTRL_E_DUPMOTOR;
      }
      else
      {
         _config.configUnused(insert.first->second.m_presetNames, mx::app::iniFile::makeKey(sections[i], "names" ));
         _config.configUnused(insert.first->second.m_presetPositions, mx::app::iniFile::makeKey(sections[i], "positions" ));
      }
      
      log<pico_channel>({sections[i], (uint8_t) channel});
   }
   
   return 0;
}
  
void picoMotorCtrl::loadConfig()
{
   if( loadConfigImpl(config) < 0)
   {
      log<text_log>("Error during config", logPrio::LOG_CRITICAL);
      m_shutdown = true;
   }
   
   if(dev::ioDevice::loadConfig(config) < 0)
   {
      log<text_log>("Error during ioDevice config", logPrio::LOG_CRITICAL);
      m_shutdown = true;
   }
   
   if(telemeterT::loadConfig(config) < 0)
   {
      log<text_log>("Error during telemeter config", logPrio::LOG_CRITICAL);
      m_shutdown = true;
   }
}

int picoMotorCtrl::appStartup()
{
   ///\todo read state from disk to get current counts.
   
   for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
   {
      it->second.m_currCounts = readChannelCounts(it->second.m_name);
      
      
      createStandardIndiNumber( it->second.m_property, it->first+"_pos", std::numeric_limits<posT>::lowest(), std::numeric_limits<posT>::max(), static_cast<posT>(1), "%d", "Position", it->first);
      it->second.m_property["current"].set(it->second.m_currCounts);
      it->second.m_property["target"].set(it->second.m_currCounts);
      it->second.m_property.setState(INDI_IDLE);
      
      if( registerIndiPropertyNew( it->second.m_property, st_newCallBack_picopos) < 0)
      {
         #ifndef PICOMOTORCTRL_TEST_NOLOG
         log<software_error>({__FILE__,__LINE__});
         #endif
         return PICOMOTORCTRL_E_INDIREG;
      }
      
      if(it->second.m_presetNames.size() > 0)
      {
         if(createStandardIndiSelectionSw( it->second.m_indiP_presetName, it->first, it->second.m_presetNames) < 0)
         {
            log<software_critical>({__FILE__, __LINE__});
            return -1;
         }
         if( registerIndiPropertyNew( it->second.m_indiP_presetName, st_newCallBack_presetName) < 0)
         {
            log<software_error>({__FILE__,__LINE__});
            return -1;
         }
      }
      
      //Here we start each channel thread, with 0 R/T prio.
      threadStart( *it->second.m_thread, it->second.m_threadInit, it->second.m_threadID, it->second.m_threadProp, 0, "", it->second.m_name, &it->second, channelThreadStart);
   }
   
   //Install empty signal handler for USR1, which is used to interrupt sleeps in the channel threads.
   struct sigaction act;
   sigset_t set;

   act.sa_sigaction = &sigUsr1Handler;
   act.sa_flags = SA_SIGINFO;
   sigemptyset(&set);
   act.sa_mask = set;

   errno = 0;
   if( sigaction(SIGUSR1, &act, 0) < 0 )
   {
      std::string logss = "Setting handler for SIGUSR1 failed. Errno says: ";
      logss += strerror(errno);

      log<software_error>({__FILE__, __LINE__, errno, 0, logss});

      return -1;
   }
   
   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   return 0;
}

int picoMotorCtrl::appLogic()
{
   if( state() == stateCodes::POWERON)
   {
      if(!powerOnWaitElapsed()) return 0;
      
      state(stateCodes::NOTCONNECTED);
   }
   
   if(state() == stateCodes::NOTCONNECTED || state() == stateCodes::ERROR)
   {
      int rv = m_telnetConn.connect(m_deviceAddr, m_devicePort);
      
      if(rv == 0)
      {
         state(stateCodes::CONNECTED);
         m_telnetConn.noLogin();
      }
      else
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;

         if(!stateLogged())
         {
            log<text_log>("Failed to connect on " + m_deviceAddr + ":" + m_devicePort);
         }
         
         return 0;
      }
      
   }
   
   if(state() == stateCodes::CONNECTED)
   {
         
      std::unique_lock<std::mutex> lock(m_telnetMutex);
      int rv = m_telnetConn.write("*IDN?\r\n", m_writeTimeout);
      if(rv != TTY_E_NOERROR)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
         log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
         state(stateCodes::ERROR);
         return 0;
      }

      rv = m_telnetConn.read("\r\n", m_readTimeout, true);
      if(rv != TTY_E_NOERROR)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
         log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
         state(stateCodes::ERROR);
         return 0;
      }

      if(m_telnetConn.m_strRead.find("New_Focus") != std::string::npos)
      {
         log<text_log>("Connected to " + m_telnetConn.m_strRead);
      }
      else
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
         
         log<software_error>({__FILE__, __LINE__, "wrong response to IDN query"});
         state(stateCodes::ERROR);
         return 0;
      }
      
      //Do a motor scan
      rv = m_telnetConn.write("MC\r\n", m_writeTimeout);
      if(rv != TTY_E_NOERROR)
      {
         if(powerState() != 1 || powerStateTarget() != 1) return 0;
         log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
         state(stateCodes::ERROR);
         return 0;
      }

      sleep(1); //wtf is this here?
      
      //Now check for each motor attached
      for(auto it=m_channels.begin(); it!=m_channels.end();++it)
      {
         std::string query = std::to_string(it->second.m_channel) + "QM?";

         rv = m_telnetConn.write(query + "\r\n", m_writeTimeout); 
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
            return 0;
         }

         rv = m_telnetConn.read("\r\n", m_readTimeout, true);
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
            return 0;
         }

         int moType = std::stoi(m_telnetConn.m_strRead);
         if(moType == 0)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<text_log>("No motor connected on channel " + std::to_string(it->second.m_channel) + " [" + it->second.m_name + "]", logPrio::LOG_CRITICAL);
            state(stateCodes::FAILURE);
            return -1;
         }
         else if (moType != 3)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<text_log>("Wrong motor type connected on channel " + std::to_string(it->second.m_channel) + " [" + it->second.m_name + "]", logPrio::LOG_CRITICAL);
            state(stateCodes::FAILURE);
            return -1;
         }
      }
      
         
      state(stateCodes::READY);
      
      
      return 0;
   }
   
   if(state() == stateCodes::READY || state() == stateCodes::OPERATING)
   {
      //check connection      
      {
         std::unique_lock<std::mutex> lock(m_telnetMutex);

         int rv = m_telnetConn.write("*IDN?\r\n", m_writeTimeout);
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
            return 0;
         }

         rv = m_telnetConn.read("\r\n", m_readTimeout, true);
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
            return 0;
         }

         if(m_telnetConn.m_strRead.find("New_Focus") == std::string::npos)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
         
            log<software_error>({__FILE__, __LINE__, "wrong response to IDN query"});
            state(stateCodes::ERROR);
            return 0;
         }
      }
      
      //Now check state of motors
      bool anymoving = false;
      
      //This is where we'd check for moving
      for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
      {
         std::unique_lock<std::mutex> lock(m_telnetMutex);
      
         std::string query = std::to_string(it->second.m_channel) + "MD?";
         
         int rv = m_telnetConn.write(query + "\r\n", m_writeTimeout);
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
            return 0;
         }

         rv = m_telnetConn.read("\r\n", m_readTimeout, true);
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) return 0;
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
            return 0;
         }

         //The check for moving here. With power off detection
         if(std::stoi(m_telnetConn.m_strRead) == 0) 
         {
            anymoving = true;
            it->second.m_moving = true;
         }
         else
         {
            it->second.m_moving = false;
         }
         
         if(it->second.m_moving == false && it->second.m_doMove == true)
         {
            it->second.m_currCounts = it->second.m_property["target"].get<long>();
            log<text_log>("moved " + it->second.m_name + " to " + std::to_string(it->second.m_currCounts) + " counts");
            it->second.m_doMove = false;
            recordPico(true);
         }
      }
   
      if(anymoving == false) state(stateCodes::READY);
      else state(stateCodes::OPERATING);
      
      for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
      {
         std::unique_lock<std::mutex> lock(m_indiMutex);
         if(it->second.m_moving) updateIfChanged(it->second.m_property, "current", it->second.m_currCounts, INDI_BUSY);
         else updateIfChanged(it->second.m_property, "current", it->second.m_currCounts, INDI_IDLE);
         
         for(size_t n=0; n < it->second.m_presetNames.size(); ++n)
         {
            bool changed = false;
            if( it->second.m_currCounts == it->second.m_presetPositions[n])
            {
               if(it->second.m_indiP_presetName[it->second.m_presetNames[n]] == pcf::IndiElement::Off) changed = true;
               it->second.m_indiP_presetName[it->second.m_presetNames[n]] = pcf::IndiElement::On;
            }
            else
            {
               if(it->second.m_indiP_presetName[it->second.m_presetNames[n]] == pcf::IndiElement::On) changed = true;
               it->second.m_indiP_presetName[it->second.m_presetNames[n]] = pcf::IndiElement::Off;
            }
            
            if(changed) m_indiDriver->sendSetProperty(it->second.m_indiP_presetName);
         }
         
         if(writeChannelCounts(it->second.m_name, it->second.m_currCounts) < 0)
         {
            log<software_error>({__FILE__, __LINE__});
         }
      }
      
      if(telemeterT::appLogic() < 0)
      {
         log<software_error>({__FILE__, __LINE__});
         return 0;
      }
      
      return 0;
   }
   
   
   return 0;
}

int picoMotorCtrl::onPowerOff()
{
   return 0;
}

int picoMotorCtrl::whilePowerOff()
{
   return 0;
}

int picoMotorCtrl::appShutdown()
{
   //Shutdown and join the threads
   for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
   {
      if(it->second.m_thread->joinable())
      {
         pthread_kill(it->second.m_thread->native_handle(), SIGUSR1);
         try
         {
            it->second.m_thread->join(); //this will throw if it was already joined
         }
         catch(...)
         {
         }
      }
   }
   
   telemeterT::appShutdown();

   return 0;
}

picoMotorCtrl::posT picoMotorCtrl::readChannelCounts(const std::string & chName)
{
   std::string statusDir = sysPath;
   statusDir += "/";
   statusDir += m_configName;

   std::string fileName = statusDir + "/" + chName;
   
   std::ifstream posIn;
   posIn.open( fileName );
   
   if(!posIn.good())
   {
      log<text_log>("no position file for " + chName + " found.  initializing to 0.");
      return 0;
   }
   
   long pos;
   posIn >> pos;
   
   posIn.close();
   
   log<text_log>("initializing " + chName + " to " + std::to_string(pos));
   
   return pos;
}

int picoMotorCtrl::writeChannelCounts( const std::string & chName,
                                       posT counts 
                                     )
{
   std::string statusDir = sysPath;
   statusDir += "/";
   statusDir += m_configName;

   std::string fileName = statusDir + "/" + chName;
   
   elevatedPrivileges ep(this);
   
   std::ofstream posOut;
   posOut.open( fileName );
   
   if(!posOut.good())
   {
      log<text_log>("could not open counts file for " + chName + " -- can not store position.", logPrio::LOG_ERROR);
      return -1;
   }
   
   posOut << counts;
   
   posOut.close();
   
   return 0;
}

void picoMotorCtrl::channelThreadStart( motorChannel * mc )
{
   mc->m_parent->channelThreadExec(mc);
}
   
void picoMotorCtrl::channelThreadExec( motorChannel * mc)
{
   //Get the thread PID immediately so the caller can return.
   mc->m_threadID = syscall(SYS_gettid);
   
   //Wait for initialization to complete.
   while( mc->m_threadInit == true && m_shutdown == 0)
   {
      sleep(1);
   }
   
   //Now begin checking for state change request.
   while(!m_shutdown)
   {
      //If told to move and not moving, start a move
      if(mc->m_doMove && !mc->m_moving && (state() == stateCodes::READY || state() == stateCodes::OPERATING))
      {
         long dr = mc->m_property["target"].get<long>() - mc->m_currCounts;
         
         recordPico(true);
         std::unique_lock<std::mutex> lock(m_telnetMutex);
         state(stateCodes::OPERATING);
         mc->m_moving = true;
         log<text_log>("moving " + mc->m_name + " by " + std::to_string(dr) + " counts");

         std::string comm = std::to_string(mc->m_channel) + "PR" + std::to_string(dr);
                  
         int rv = m_telnetConn.write(comm + "\r\n", m_writeTimeout);
         if(rv != TTY_E_NOERROR)
         {
            if(powerState() != 1 || powerStateTarget() != 1) //about to get POWEROFF
            {
               sleep(1);
               continue;
            }
            log<software_error>({__FILE__, __LINE__, tty::ttyErrorString(rv)});
            state(stateCodes::ERROR);
         }
      }
      else if( !(state() == stateCodes::READY || state() == stateCodes::OPERATING))
      {
         mc->m_doMove = false; //In case a move is requested when not able to move
      }
      
      sleep(1);
   }
   
   
}


int picoMotorCtrl::st_newCallBack_picopos( void * app,
                                           const pcf::IndiProperty &ipRecv
                                         )
{
   return static_cast<picoMotorCtrl*>(app)->newCallBack_picopos(ipRecv);
}

int picoMotorCtrl::newCallBack_picopos( const pcf::IndiProperty &ipRecv )
{
   
   //Search for the channel
   std::string propName = ipRecv.getName();
   size_t nend = propName.rfind("_pos");
   
   if(nend == std::string::npos)
   {
      log<software_error>({__FILE__, __LINE__, "Channel without _pos received"});
      return -1;
   }
   
   std::string chName = propName.substr(0, nend);
   channelMapT::iterator it = m_channels.find(chName);

   if(it == m_channels.end())
   {
      log<software_error>({__FILE__, __LINE__, "Unknown channel name received"});
      return -1;
   }

   if(it->second.m_doMove == true)
   {
      log<text_log>("channel " + it->second.m_name + " is already moving", logPrio::LOG_WARNING);
      return 0;
   }
   
   //Set the target element, and the doMove flag, and then signal the thread.
   {//scope for mutex
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      long counts; //not actually used
      if(indiTargetUpdate( it->second.m_property, counts, ipRecv, true) < 0)
      {
         return log<software_error,-1>({__FILE__,__LINE__});
      }
   }
   
   it->second.m_doMove= true;
   
   pthread_kill(it->second.m_thread->native_handle(), SIGUSR1);
   
   return 0;
}

int picoMotorCtrl::st_newCallBack_presetName( void * app,
                                             const pcf::IndiProperty &ipRecv
                                            )
{
   return static_cast<picoMotorCtrl*>(app)->newCallBack_presetName (ipRecv);
}

int picoMotorCtrl::newCallBack_presetName( const pcf::IndiProperty &ipRecv )
{
   channelMapT::iterator it = m_channels.find(ipRecv.getName());

   if(it == m_channels.end())
   {
      log<software_error>({__FILE__, __LINE__, "Unknown channel name received"});
      return -1;
   }

   if(it->second.m_doMove == true)
   {
      log<text_log>("channel " + it->second.m_name + " is already moving", logPrio::LOG_WARNING);
      return 0;
   }
   
   long counts = -1e10;
   
   size_t i;
   for(i=0; i< it->second.m_presetNames.size(); ++i) 
   {
      if(!ipRecv.find(it->second.m_presetNames[i])) continue;
      
      if(ipRecv[it->second.m_presetNames[i]].getSwitchState() == pcf::IndiElement::On)
      {
         if(counts != -1e10)
         {
            log<text_log>("More than one preset selected", logPrio::LOG_ERROR);
            return -1;
         }
         
         counts = it->second.m_presetPositions[i];
         std::cerr << "selected: " << it->second.m_presetNames[i] << " " << counts << "\n";
      }
   }
   
   //Set the target element, and the doMove flag, and then signal the thread.
   {//scope for mutex
      std::unique_lock<std::mutex> lock(m_indiMutex);
      
      it->second.m_property["target"].set(counts);
   }
   
   it->second.m_doMove= true;
   
   pthread_kill(it->second.m_thread->native_handle(), SIGUSR1);
   
   return 0;
}

int picoMotorCtrl::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_pico());
}
   
int picoMotorCtrl::recordTelem( const telem_pico * )
{
   return recordPico(true);
}

int picoMotorCtrl::recordPico( bool force )
{
   static std::vector<int64_t> lastpos(m_nChannels, std::numeric_limits<long>::max());
   
   bool changed = false;
   for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
   {
      if(it->second.m_currCounts != lastpos[it->second.m_channel-1]) changed = true;
   }
   
   if( changed || force )
   {
      for(channelMapT::iterator it = m_channels.begin(); it != m_channels.end(); ++ it)
      {
         lastpos[it->second.m_channel-1] = it->second.m_currCounts;
      }
   
      telem<telem_pico>(lastpos);
   }

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //picoMotorCtrl_hpp
