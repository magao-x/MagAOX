/** \file stdMotionStage.hpp
  * \brief Standard motion stage interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef stdMotionStage_hpp
#define stdMotionStage_hpp


namespace MagAOX
{
namespace app
{
namespace dev 
{


/// MagAO-X standard motion stage interface
/** Implements the standard interface to a MagAO-X motion stage.
  * This includes the mcbl filter wheels, the zaber stages. 
  * 
  * The required interface to be implemented in derivedT is
  * \code
   
    int stop(); //Note that the INDI mutex will not be locked on this call 
    
    int startHoming(); //INDI mutex will be locked on this call.
    
    double presetNumber();
    
    int moveTo(double); //INDI mutex will be locked on this call.
    \endcode 
  *
  * In addition the derived class is responsible for setting m_moving and m_preset. m_preset_target should also be set if the wheel
  * is moved via a low-level position command.
  * 
  * The derived class `derivedT` must be a MagAOXApp\<true\>, and should declare this class a friend like so: 
   \code
    friend class dev::stdMotionStage<derivedT>;
   \endcode
  *
  *
  *
  * Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`
  * `onPowerOff`, and `whilePowerOff`,  must be placed in the derived class's functions of the same name.
  *
  * \ingroup appdev
  */
template<class derivedT>
class stdMotionStage
{
protected:

   /** \name Configurable Parameters
    * @{
    */
   
   bool m_powerOnHome {false}; ///< If true, then the motor is homed at startup (by this software or actual power on)

   std::vector<std::string> m_presetNames; ///< The names of each position on the stage.
   
   std::vector<double> m_presetPositions; ///< The positions, in arbitrary units, of each preset.  If 0, then the integer position number (starting from 1) is used to calculate.
   
   ///@}
   
   std::string m_presetNotation {"preset"}; ///< Notation used to refer to a preset, should be singular, as in "preset" or "filter".
   
   int m_moving {0}; ///< Whether or not the stage is moving.
   
   double m_preset {0}; ///< The current numerical preset position [1.0 is index 0 in the preset name vector]
   double m_preset_target {0}; ///< The target numerical preset position [1.0 is index 0 in the preset name vector]
   
public:

   ///Destructor
   ~stdMotionStage() noexcept;
   
   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       stdMotionStage<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       stdMotionStage<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       stdMotionStage<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// Application logic 
   /** Checks the stdMotionStage thread
     * 
     * This should be called from the derived's appLogic() as in
     * \code
       stdMotionStage<derivedT>::appLogic();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appLogic();

   /// Actions on power off
   /**
     * This should be called from the derived's onPowerOff() as in
     * \code
       stdMotionStage<derivedT>::onPowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int onPowerOff();

   /// Actions while powered off
   /**
     * This should be called from the derived's whilePowerOff() as in
     * \code
       stdMotionStage<derivedT>::whilePowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int whilePowerOff();
   
   /// Application the shutdown 
   /** Shuts down the stdMotionStage thread
     * 
     * \code
       stdMotionStage<derivedT>::appShutdown();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appShutdown();
   
protected:
   
   
    /** \name INDI 
      *
      *@{
      */ 
protected:
   //declare our properties
   
   ///The position of the stage in presets
   pcf::IndiProperty m_indiP_preset;

   ///The name of the nearest preset for this position
   pcf::IndiProperty m_indiP_presetName;

   ///Command the stage to home. .
   pcf::IndiProperty m_indiP_home;
   
   ///Command the stage to halt. 
   pcf::IndiProperty m_indiP_stop;

public:

   /// The static callback function to be registered for stdMotionStage properties
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_stdMotionStage( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                             const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                           );
   
   /// Callback to process a NEW preset position request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_preset( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW preset name request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_presetName( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW home request switch toggle
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_home( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW stop request switch toggle
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_stop( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

   ///@}
   
private:
   derivedT & derived()
   {
      return *static_cast<derivedT *>(this);
   }
};

template<class derivedT>
stdMotionStage<derivedT>::~stdMotionStage() noexcept
{
   return;
}



template<class derivedT>
void stdMotionStage<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   static_cast<void>(config);
   
   config.add("motor.powerOnHome", "", "motor.powerOnHome", argType::Required, "motor", "powerOnHome", false, "bool", "If true, home at startup/power-on.  Default=false.");
   
   config.add(m_presetNotation + "s.names", "", m_presetNotation + "s.names",  argType::Required, m_presetNotation+"s", "names", false, "vector<string>", "The names of the " + m_presetNotation+ "s.");
   config.add(m_presetNotation + "s.positions", "", m_presetNotation + "s.positions",  argType::Required, m_presetNotation+"s", "positions", false, "vector<double>", "The positions of the " + m_presetNotation + "s.  If omitted or 0 then order is used.");
   
}

template<class derivedT>
void stdMotionStage<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_powerOnHome, "motor.powerOnHome");
   
   config(m_presetNames, m_presetNotation + "s.names");
   m_presetPositions.resize(m_presetNames.size(), 0);
   for(size_t n=0;n<m_presetPositions.size();++n) m_presetPositions[n] = n+1;
   config(m_presetPositions, m_presetNotation + "s.positions");
   for(size_t n=0;n<m_presetPositions.size();++n) if(m_presetPositions[n] == 0) m_presetPositions[n] = n+1;
}
   


template<class derivedT>
int stdMotionStage<derivedT>::appStartup()
{
 
   derived().createStandardIndiNumber( m_indiP_preset, m_presetNotation, 1.0, (double) m_presetNames.size(), 0.0, "%0.3d");
   if( derived().registerIndiPropertyNew( m_indiP_preset, st_newCallBack_stdMotionStage) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   derived().createStandardIndiSelectionSw( m_indiP_presetName, m_presetNotation + "Name", m_presetNames);
   if( derived().registerIndiPropertyNew( m_indiP_presetName, st_newCallBack_stdMotionStage) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   derived().createStandardIndiRequestSw( m_indiP_home, "home");
   if( derived().registerIndiPropertyNew( m_indiP_home, st_newCallBack_stdMotionStage) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }

   derived().createStandardIndiRequestSw( m_indiP_stop, "stop");
   if( derived().registerIndiPropertyNew( m_indiP_stop, st_newCallBack_stdMotionStage) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   return 0;
}

template<class derivedT>
int stdMotionStage<derivedT>::appLogic()
{
   return 0;

}

template<class derivedT>
int stdMotionStage<derivedT>::onPowerOff()
{
   if( !derived().m_indiDriver ) return 0;
   return 0;
}

template<class derivedT>
int stdMotionStage<derivedT>::whilePowerOff()
{
   return 0;
}

template<class derivedT>
int stdMotionStage<derivedT>::appShutdown()
{
   return 0;
}


template<class derivedT>
int stdMotionStage<derivedT>::st_newCallBack_stdMotionStage( void * app,
                                                   const pcf::IndiProperty &ipRecv
                                                 )
{
   std::string name = ipRecv.getName();
   derivedT * _app = static_cast<derivedT *>(app);
   
   if(name == "stop") return _app->newCallBack_stop(ipRecv); //Check this first to make sure it 
   if(name == "home") return _app->newCallBack_home(ipRecv);
   if(name == _app->m_presetNotation) return _app->newCallBack_preset(ipRecv);
   if(name == _app->m_presetNotation + "Name") return _app->newCallBack_presetName (ipRecv);
   
   return -1;
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_preset ( const pcf::IndiProperty &ipRecv )
{
   double target;
   
   if( derived().indiTargetUpdate( m_indiP_preset, target, ipRecv, true) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_preset_target = target;
   
   std::lock_guard<std::mutex> guard(derived().m_indiMutex);
   return derived().moveTo(target);
   
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_presetName( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName() != m_indiP_presetName.getName())
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   std::string newName = "";
   int newn = -1;
   
   size_t i;
   for(i=0; i< m_presetNames.size(); ++i) 
   {
      if(!ipRecv.find(m_presetNames[i])) continue;
      
      if(ipRecv[m_presetNames[i]].getSwitchState() == pcf::IndiElement::On)
      {
         if(newName != "")
         {
            derivedT::template log<text_log>("More than one " + m_presetNotation + " selected", logPrio::LOG_ERROR);
            return -1;
         }
         
         newName = m_presetNames[i];
         newn = i;
      }
   }
   
   if(newName == "" || newn < 0)
   {
      return 0; //This is just an reset of current probably
   }
   
   std::lock_guard<std::mutex> guard(derived().m_indiMutex);

   m_preset_target = m_presetPositions[newn]; //(double) newn + 1.0;
   derived().updateIfChanged(m_indiP_preset, "target",  m_preset_target, INDI_BUSY);
   
   return derived().moveTo(m_preset_target);
   
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_home( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName() != m_indiP_home.getName())
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("request")) return 0;
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      indi::updateSwitchIfChanged(m_indiP_home, "request", pcf::IndiElement::On, derived().m_indiDriver, INDI_BUSY);
      
      std::lock_guard<std::mutex> guard(derived().m_indiMutex);
      return derived().startHoming();
   }
   return 0;  
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_stop( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName() != m_indiP_stop.getName())
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   if(!ipRecv.find("request")) return 0;
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      indi::updateSwitchIfChanged(m_indiP_stop, "request", pcf::IndiElement::On, derived().m_indiDriver, INDI_BUSY);
    
      //-->do not lock mutex!
      return derived().stop();
   }
   return 0;  
}

template<class derivedT>
int stdMotionStage<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   int n = derived().presetNumber();
   
   size_t nn = n;
 
   //Check for changes and update the filterNames switch vectorm_presetNotation + ".
   bool changed = false;
   
   static int last_moving = m_moving;
   
   if(last_moving != m_moving)
   {
      changed = true;
      last_moving = m_moving;
   }
   
   for(size_t i =0; i < m_presetNames.size(); ++i)
   {
      if( i == nn )
      {
         if(m_indiP_presetName[m_presetNames[i]] != pcf::IndiElement::On) 
         {
            changed = true;
            m_indiP_presetName[m_presetNames[i]] = pcf::IndiElement::On;
         }
      }
      else
      {
         if(m_indiP_presetName[m_presetNames[i]] != pcf::IndiElement::Off) 
         {
            changed = true;
            m_indiP_presetName[m_presetNames[i]] = pcf::IndiElement::Off;
         }
      }
   }
   if(changed)
   {
      if(m_moving)
      {
         m_indiP_presetName.setState(INDI_BUSY);
      }
      else
      {
         m_indiP_presetName.setState(INDI_IDLE);
      }
            
      derived().m_indiDriver->sendSetProperty(m_indiP_presetName);
   }
   
  
   
   if(m_moving)
   {
      indi::updateIfChanged(m_indiP_preset, "current", m_preset, derived().m_indiDriver,INDI_BUSY);
      indi::updateIfChanged(m_indiP_preset, "target", m_preset_target, derived().m_indiDriver,INDI_BUSY);
   }
   else
   {
      indi::updateIfChanged(m_indiP_preset, "current", m_preset, derived().m_indiDriver,INDI_IDLE);    
      indi::updateIfChanged(m_indiP_preset, "target", m_preset_target, derived().m_indiDriver,INDI_IDLE);
   }
   
   
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //stdMotionStage_hpp
