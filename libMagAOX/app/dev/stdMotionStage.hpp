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
    
    float presetNumber();
    
    int moveTo(float); //INDI mutex will be locked on this call.
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

   int m_homePreset {-1}; ///< If >=0, this preset position is moved to after homing
   
   std::vector<std::string> m_presetNames; ///< The names of each position on the stage.
   
   std::vector<float> m_presetPositions; ///< The positions, in arbitrary units, of each preset.  If 0, then the integer position number (starting from 1) is used to calculate.
   
   ///@}
   
   std::string m_presetNotation {"preset"}; ///< Notation used to refer to a preset, should be singular, as in "preset" or "filter".
   
   bool m_fractionalPresets {true}; ///< Flag to set in constructor determining if fractional presets are allowed.  Used for INDI/GUIs.
   
   bool m_defaultPositions {true}; ///< Flag controlling whether the default preset positions (the vector index) are set in loadConfig.
   
   int8_t m_moving {0}; ///< Whether or not the stage is moving.  -2 means powered off, -1 means not homed, 0 means not moving, 1 means moving, 2 means homing.  
   int8_t m_movingState {0}; ///< Used to track the type of command.  If > 1 this is a command to move to a preset.  If 0 then it is a move to an arbitrary position.
   
   float m_preset {0}; ///< The current numerical preset position [1.0 is index 0 in the preset name vector]
   float m_preset_target {0}; ///< The target numerical preset position [1.0 is index 0 in the preset name vector]
   
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
   int setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       stdMotionStage<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   int loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

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
   int newCallBack_m_indiP_preset( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW preset name request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_m_indiP_presetName( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW home request switch toggle
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_m_indiP_home( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW stop request switch toggle
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_m_indiP_stop( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Update the INDI properties for this device controller
   /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int updateINDI();

    ///@}

   /** \name Telemeter Interface 
     * @{
     */
   
   int recordStage( bool force = false );
   
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
int stdMotionStage<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   static_cast<void>(config);
   
   config.add("stage.powerOnHome", "", "stage.powerOnHome", argType::Required, "stage", "powerOnHome", false, "bool", "If true, home at startup/power-on.  Default=false.");
   
   config.add("stage.homePreset", "", "stage.homePreset", argType::Required, "stage", "homePreset", false, "int", "If >=0, this preset number is moved to after homing.");
   
   config.add(m_presetNotation + "s.names", "", m_presetNotation + "s.names",  argType::Required, m_presetNotation+"s", "names", false, "vector<string>", "The names of the " + m_presetNotation+ "s.");
   config.add(m_presetNotation + "s.positions", "", m_presetNotation + "s.positions",  argType::Required, m_presetNotation+"s", "positions", false, "vector<float>", "The positions of the " + m_presetNotation + "s.  If omitted or 0 then order is used.");
   
   return 0;
}

template<class derivedT>
int stdMotionStage<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_powerOnHome, "stage.powerOnHome");
   config(m_homePreset, "stage.homePreset");
   
   config(m_presetNames, m_presetNotation + "s.names");
   
   if(m_defaultPositions)
   {
      m_presetPositions.resize(m_presetNames.size(), 0);
      for(size_t n=0;n<m_presetPositions.size();++n) m_presetPositions[n] = n+1;
   }
   
   config(m_presetPositions, m_presetNotation + "s.positions");
   
   if(m_defaultPositions)
   {
      for(size_t n=0;n<m_presetPositions.size();++n) if(m_presetPositions[n] == 0) m_presetPositions[n] = n+1;
   }
   
   return 0;
}
   


template<class derivedT>
int stdMotionStage<derivedT>::appStartup()
{
   double step = 0.0;
   std::string format = "%.4f";
   if(!m_fractionalPresets)
   {
      step = 1.0;
      format = "%d";
   }
 
   derived().createStandardIndiNumber( m_indiP_preset, m_presetNotation, 1.0, (double) m_presetNames.size(), step, format);
   m_indiP_preset["current"].set(0);
   m_indiP_preset["target"].set(0);
   if( derived().registerIndiPropertyNew( m_indiP_preset, st_newCallBack_stdMotionStage) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   if(derived().createStandardIndiSelectionSw( m_indiP_presetName, m_presetNotation + "Name", m_presetNames) < 0)
   {
      derivedT::template log<software_critical>({__FILE__, __LINE__});
      return -1;
   }
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
   m_moving = -2;
   m_preset = 0; 
   m_preset_target = 0;
   
   if( !derived().m_indiDriver ) return 0;
   
   return 0;
}

template<class derivedT>
int stdMotionStage<derivedT>::whilePowerOff()
{
   if( !derived().m_indiDriver ) return 0;
   
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
   
   if(name == "stop") return _app->newCallBack_m_indiP_stop(ipRecv); //Check this first to make sure it 
   if(name == "home") return _app->newCallBack_m_indiP_home(ipRecv);
   if(name == _app->m_presetNotation) return _app->newCallBack_m_indiP_preset(ipRecv);
   if(name == _app->m_presetNotation + "Name") return _app->newCallBack_m_indiP_presetName (ipRecv);
   
   return -1;
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_m_indiP_preset ( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_preset, ipRecv); 

    float target;
   
    if( derived().indiTargetUpdate( m_indiP_preset, target, ipRecv, true) < 0)
    {
        derivedT::template log<software_error>({__FILE__,__LINE__});
        return -1;
    }
   
    m_preset_target = target;
   
    std::lock_guard<std::mutex> guard(derived().m_indiMutex);
    m_movingState = 0; //this is not a preset move
    return derived().moveTo(target);
   
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_m_indiP_presetName( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_presetName, ipRecv);
   
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

   m_preset_target = m_presetPositions[newn]; 
   derived().updateIfChanged(m_indiP_preset, "target",  m_preset_target, INDI_BUSY);
   
   m_movingState = 1; //This is a preset move
   return derived().moveTo(m_preset_target);
   
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_m_indiP_home( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_home, ipRecv);
   
   if(!ipRecv.find("request")) return 0;
   
   if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
   {
      indi::updateSwitchIfChanged(m_indiP_home, "request", pcf::IndiElement::On, derived().m_indiDriver, INDI_BUSY);
      
      std::lock_guard<std::mutex> guard(derived().m_indiMutex);
      m_movingState = 0;
      return derived().startHoming();
   }
   return 0;  
}

template<class derivedT>
int stdMotionStage<derivedT>::newCallBack_m_indiP_stop( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS_DERIVED(m_indiP_stop, ipRecv);

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
      m_movingState = 0;
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
 
   //Check for changes and update the filterNames
   bool changed = false;
   
   static int8_t last_moving = -1; //Initialize so we always update first time through.
   
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
      if(m_moving > 0)
      {
         m_indiP_presetName.setState(INDI_BUSY);
      }
      else
      {
         m_indiP_presetName.setState(INDI_IDLE);
      }
            
      m_indiP_presetName.setTimeStamp(pcf::TimeStamp());
      derived().m_indiDriver->sendSetProperty(m_indiP_presetName);
   }
      
  
   
   if(m_moving && m_movingState < 1)
   {
      indi::updateIfChanged(m_indiP_preset, "current", m_preset, derived().m_indiDriver,INDI_BUSY);
      indi::updateIfChanged(m_indiP_preset, "target", m_preset_target, derived().m_indiDriver,INDI_BUSY);
   }
   else
   {
      indi::updateIfChanged(m_indiP_preset, "current", m_preset, derived().m_indiDriver,INDI_IDLE);    
      indi::updateIfChanged(m_indiP_preset, "target", m_preset_target, derived().m_indiDriver,INDI_IDLE);
   }
   
   return recordStage();
}

template<class derivedT>
int stdMotionStage<derivedT>::recordStage(bool force)
{
   static int8_t last_moving = m_moving + 100; //guarantee first run
   static float last_preset;
   static std::string last_presetName;
   
   size_t n = derived().presetNumber();

   std::string presetName;
   if(n < m_presetNames.size()) presetName = m_presetNames[n];
   
   if( m_moving != last_moving || m_preset != last_preset || presetName != last_presetName || force)
   {
      derived().template telem<telem_stage>({m_moving, m_preset, presetName});
      last_moving = m_moving;
      last_preset = m_preset;
      last_presetName = presetName;

   }
   
   
   return 0;
}
   

} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //stdMotionStage_hpp
