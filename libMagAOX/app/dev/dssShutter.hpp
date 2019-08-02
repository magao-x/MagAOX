/** \file dssShutter.hpp
  * \brief Uniblitz DSS shutter interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef dssShutter_hpp
#define dssShutter_hpp


namespace MagAOX
{
namespace app
{
namespace dev 
{
   
/// MagAO-X Uniblitz DSS Shutter interface
/**
  * 
  * The derived class `derivedT` must be a MagAOXApp\<true\>, and should declare this class a friend like so: 
   \code
    friend class dev::dssShutter<derivedT>;
   \endcode
  *
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`
  * `onPowerOff`, `whilePowerOff`, functions must be placed in the derived class's functions of the same name.
  *
  * 
  * \ingroup appdev
  */
template<class derivedT>
class dssShutter
{
protected:

   /** \name Configurable Parameters
    * @{
    */

   std::string m_powerDevice;    ///< The device controlling this shutter's power
   std::string m_powerChannel;   ///< The channel controlling this shutter's power
   
   std::string m_dioDevice;      ///< The device controlling this shutter's digital I/O.
   std::string m_sensorChannel;  ///< The channel reading this shutter's sensor
   std::string m_triggerChannel; ///< The channel sending this shutter's trigger
      
   ///@}
   
   int m_powerState {-1};  ///< The current power state, -1 is unknown, 0 is off, 1 is on.
   
   int m_sensorState {-1}; ///< The current sensor state, -1 is unknown, 0 is shut, 1 is open.
    
   int m_triggerState {-1}; ///< The current trigger state.  -1 is unknown, 0 is low, 1 is high.
   
public:

   
   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       dssShutter<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       dssShutter<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       dssShutter<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// application logic
   /** This should be called in `derivedT::appLogic` as
     * \code
       dssShutter<derivedT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// applogic shutdown
   /** This should be called in `derivedT::appShutdown` as
     * \code
       dssShutter<derivedT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
   /// Actions on power off
   /** This should be called in `derivedT::appPowerOff` as
     * \code
       dssShutter<derivedT>::appPowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int onPowerOff();

   /// Actions while powered off
   /** This should be called in `derivedT::whilePowerOff` as
     * \code
       dssShutter<derivedT>::whilePowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int whilePowerOff();
   
   /// Open the shutter
   /** Do not lock the mutex before calling this.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int open();
   
   /// Shut the shutter
   /** Do not lock the mutex before calling this.
     * 
     * \returns 0 on success
     * \returns -1 on error
     */
   int shut();
   
protected:
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   pcf::IndiProperty m_indiP_powerChannel; ///< Property used to monitor the shutter's power state
   pcf::IndiProperty m_indiP_sensorChannel; ///< Property used to monitor the shutter's hall sensor
   pcf::IndiProperty m_indiP_triggerChannel; ///< Property used to monitor and set the shutter's trigger
   
   pcf::IndiProperty m_indiP_state; ///< Property used to report the current shutter state (unkown, off, shut, open)
   
   
   
public:

   /// Update the INDI properties for this device controller
   /** You should call this once per main loop, with the INDI mutex locked.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   
   /// The static callback function to be registered for shutter power channel changes
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_setCallBack_powerChannel( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int setCallBack_powerChannel( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// The static callback function to be registered for shutter sensor channel changes
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_setCallBack_sensorChannel( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int setCallBack_sensorChannel( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// The static callback function to be registered for shutter trigger channel changes
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_setCallBack_triggerChannel( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                           const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                         );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int setCallBack_triggerChannel( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   
   /// The static callback function to be registered for shutter state change requests.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_state( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                    const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                  );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_state( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   

   ///@}
   
private:
   
   /// Access the derived class.
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};


template<class derivedT>
void dssShutter<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   config.add("shutter.powerDevice", "", "shutter.powerDevice", argType::Required, "shutter", "powerDevice", false, "string", "The device controlling this shutter's power");
   config.add("shutter.powerChannel", "", "shutter.powerChannel", argType::Required, "shutter", "powerChannel", false, "string", "The channel controlling this shutter's power");
   
   config.add("shutter.dioDevice", "", "shutter.dioDevice", argType::Required, "shutter", "dioDevice", false, "string", "The device controlling this shutter's digital I/O.");
   
   config.add("shutter.sensorChannel", "", "shutter.sensorChannel", argType::Required, "shutter", "sensorChannel", false, "string", "The channel reading this shutter's sensor.");
   
   config.add("shutter.triggerChannel", "", "shutter.triggerChannel", argType::Required, "shutter", "triggerChannel", false, "string", "The channel sending this shutter's trigger.");
}

template<class derivedT>
void dssShutter<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_powerDevice, "shutter.powerDevice");
   config(m_powerChannel, "shutter.powerChannel");
   config(m_dioDevice, "shutter.dioDevice");
   config(m_sensorChannel, "shutter.sensorChannel");
   config(m_triggerChannel, "shutter.triggerChannel");

}
   

template<class derivedT>
int dssShutter<derivedT>::appStartup()
{
   //Register the powerChannel property for updates
   if( derived().registerIndiPropertySet( m_indiP_powerChannel, m_powerDevice, m_powerChannel, st_setCallBack_powerChannel) < 0)
   {
      #ifndef DSSSHUTTER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the sensorChannel property for updates
   if( derived().registerIndiPropertySet( m_indiP_sensorChannel, m_dioDevice, m_sensorChannel, st_setCallBack_sensorChannel) < 0)
   {
      #ifndef DSSSHUTTER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the triggerChannel property for updates
   if( derived().registerIndiPropertySet( m_indiP_triggerChannel, m_dioDevice, m_triggerChannel, st_setCallBack_triggerChannel) < 0)
   {
      #ifndef DSSSHUTTER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Register the shmimName INDI property
   m_indiP_state = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_state.setDevice(derived().configName());
   m_indiP_state.setName("shutter");
   m_indiP_state.setPerm(pcf::IndiProperty::ReadWrite); 
   m_indiP_state.setState(pcf::IndiProperty::Idle);
   m_indiP_state.add(pcf::IndiElement("current"));
   m_indiP_state["current"] = "";
   m_indiP_state.add(pcf::IndiElement("target"));
   m_indiP_state["target"] = "";
   
   if( derived().registerIndiPropertyNew( m_indiP_state, st_newCallBack_state) < 0)
   {
      #ifndef DSSSHUTTER_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   
   
   return 0;

}

template<class derivedT>
int dssShutter<derivedT>::appLogic()
{
   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::appShutdown()
{
   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::onPowerOff()
{
   return updateINDI();
}

template<class derivedT>
int dssShutter<derivedT>::whilePowerOff()
{
   return updateINDI();
}

template<class derivedT>
int dssShutter<derivedT>::open()
{
   if(m_powerState != 1) return 0;
   
   int startss = m_sensorState;
   
   if(startss) return 0; //already open
   
   //First try:
   int startts = m_triggerState;
   
   derived().sendNewProperty (m_indiP_triggerChannel, "target", (int) !m_triggerState);
   
   //--invert
   
   //--wait for state change checking frequently
   
   //if(startss) return; //and log 
   
   //timeout: send original state
   
   //--wait for state change checking frequently
   
   //if(startss) return; //and log 
   
   //timeout: failed
   
   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::shut()
{
   if(m_powerState != 1) return 0;
   
   int startss = m_sensorState;
   
   if(!startss) return 0; //Already shut
   
   //First try:
   int startts = m_triggerState;
   
   derived().sendNewProperty (m_indiP_triggerChannel, "target", (int) !m_triggerState);
   
   //--invert
   
   //--wait for state change checking frequently
   
   //if(startss) return; //and log 
   
   //timeout: send original state
   
   //--wait for state change checking frequently
   
   //if(startss) return; //and log 
   
   //timeout: failed
   
   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::updateINDI()
{
   if(m_powerState !=0 && m_powerState != 1)
   {
      indi::updateIfChanged(m_indiP_state, "current", std::string("UNKNOWN"), derived().m_indiDriver);
      indi::updateIfChanged(m_indiP_state, "target", std::string(""), derived().m_indiDriver);
   
      return 0;
   }
   
   if(m_powerState == 0)
   {
      indi::updateIfChanged(m_indiP_state, "current", std::string("OFF"), derived().m_indiDriver);
      indi::updateIfChanged(m_indiP_state, "target", std::string(""), derived().m_indiDriver);
   
      return 0;
   }
   
   if(m_sensorState == 0)
   {
      indi::updateIfChanged(m_indiP_state, "current", std::string("SHUT"), derived().m_indiDriver);
      //indi::updateIfChanged(m_indiP_state, "target", std::string(""), derived().m_indiDriver);
   
      return 0;
   }
   
   if(m_sensorState == 1)
   {
      indi::updateIfChanged(m_indiP_state, "current", std::string("OPEN"), derived().m_indiDriver);
      //indi::updateIfChanged(m_indiP_state, "target", std::string(""), derived().m_indiDriver);
   
      return 0;
   }
   
   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::st_setCallBack_powerChannel( void * app,
                                                       const pcf::IndiProperty &ipRecv
                                                      )
{
   return static_cast<derivedT *>(app)->setCallBack_powerChannel(ipRecv);
}

template<class derivedT>
int dssShutter<derivedT>::setCallBack_powerChannel( const pcf::IndiProperty &ipRecv )
{
   std::string ps;

   std::cerr << "In shutter power callback\n";
   
   if(ipRecv.getName() != m_indiP_powerChannel.getName()) return 0;
   
   m_indiP_powerChannel = ipRecv;
   
   if(!ipRecv.find("state")) return 0;
   
   ps = ipRecv["state"].get<std::string>();
   
   if(ps == "On")
   {
      m_powerState = 1;
   }
   else if (ps == "Off")
   {
      m_powerState = 0;
   }
   else
   {
      m_powerState = -1;
   }

   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::st_setCallBack_sensorChannel( void * app,
                                                        const pcf::IndiProperty &ipRecv
                                                      )
{
   return static_cast<derivedT *>(app)->setCallBack_sensorChannel(ipRecv);
}

template<class derivedT>
int dssShutter<derivedT>::setCallBack_sensorChannel( const pcf::IndiProperty &ipRecv )
{
   
   if(ipRecv.getName() != m_indiP_sensorChannel.getName()) return 0;
   
   m_indiP_sensorChannel = ipRecv;
   
   if(!ipRecv.find("current")) return 0;
   
   int ss = ipRecv["current"].get<int>(); 

   if(ss == 1)
   {
      m_sensorState = 1;
   }
   else if (ss == 0)
   {
      m_sensorState = 0;
   }
   else
   {
      m_sensorState = -1;
   }

   return 0;
}


template<class derivedT>
int dssShutter<derivedT>::st_setCallBack_triggerChannel( void * app,
                                                         const pcf::IndiProperty &ipRecv
                                                       )
{
   return static_cast<derivedT *>(app)->setCallBack_triggerChannel(ipRecv);
}

template<class derivedT>
int dssShutter<derivedT>::setCallBack_triggerChannel( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName() != m_indiP_triggerChannel.getName()) return 0;
   
   m_indiP_triggerChannel = ipRecv;
   
   if(!ipRecv.find("current")) return 0;
   
   int ts = ipRecv["current"].get<int>(); 

   std::cerr << "\n*************\ntrigger: " << ts << "\n*****************\n";
   
   if(ts == 1)
   {
      m_triggerState = 1;
   }
   else if (ts == 0)
   {
      m_triggerState = 0;
   }
   else
   {
      m_triggerState = -1;
   }

   return 0;
}

template<class derivedT>
int dssShutter<derivedT>::st_newCallBack_state( void * app,
                                              const pcf::IndiProperty &ipRecv
                                            )
{
   return static_cast<derivedT *>(app)->newCallBack_state(ipRecv);
}

template<class derivedT>
int dssShutter<derivedT>::newCallBack_state( const pcf::IndiProperty &ipRecv )
{
   if (ipRecv.getName() == m_indiP_state.getName())
   {
      std::string current;
      std::string target;
      
      if(ipRecv.find("curent")) current = ipRecv["current"].get();
      
      if(ipRecv.find("target")) target = ipRecv["target"].get();
      
      if(target == "") target = current;
      
      target = mx::ioutils::toUpper(target);
      
      if(target != "OPEN" && target != "SHUT")
      {
         return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "invalid shutter request"});
      }
      else
      {
         //Get a lock
         std::unique_lock<std::mutex> lock(derived().m_indiMutex);
         indi::updateIfChanged(m_indiP_state, "target", target, derived().m_indiDriver);
      }
      
      if(target == "OPEN") return open();
      if(target == "SHUT") return shut();
      
      return -1; //never get here
   }
   return derivedT::template log<software_error,-1>({__FILE__, __LINE__, "wrong INDI-P in callback"});
}
   


} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //dssShutter_hpp
