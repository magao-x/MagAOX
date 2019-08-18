/** \file stdFilterWheel.hpp
  * \brief Standard filter wheel interface
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup app_files
  */

#ifndef stdFilterWheel_hpp
#define stdFilterWheel_hpp


namespace MagAOX
{
namespace app
{
namespace dev 
{


/// MagAO-X standard filter wheel interface
/** Implements the standard interface to a MagAO-X filter wheel
  * 
  * The required interface to be implemented in derivedT is
  * \code
   
    int stop(); //Note that the INDI mutex will not be locked on this call 
    
    int startHoming(); //INDI mutex will be locked on this call.
    
    int moveTo(double); //INDI mutex will be locked on this call.
    \endcode 
  *
  * In addition the derived class is responsible for setting m_moving and m_filter. m_filter_target should also be set if the wheel
  * is moved via a low-level position command.
  * 
  * The derived class `derivedT` must be a MagAOXApp\<true\>, and should declare this class a friend like so: 
   \code
    friend class dev::stdFilterWheel<derivedT>;
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
class stdFilterWheel
{
protected:

   /** \name Configurable Parameters
    * @{
    */
   
   bool m_powerOnHome {false}; ///< If true, then the motor is homed at startup (by this software or actual power on)

   std::vector<std::string> m_filterNames; ///< The names of each position in the wheel.
   
   std::vector<double> m_filterPositions; ///< The positions, in filter units, of each filter.  If 0, then the integer position number is used to calculate.
   
   ///@}
   
   
   int m_moving {0}; ///< Whether or not the wheel is moving.
   
   double m_filter {0}; ///< The current numerical filter position [1.0 is index 0 in the filter name vector]
   double m_filter_target {0}; ///< The target numerical filter position [1.0 is index 0 in the filter name vector]
   
public:

   ///Destructor
   ~stdFilterWheel() noexcept;
   
   /// Setup the configuration system
   /**
     * This should be called in `derivedT::setupConfig` as
     * \code
       stdFilterWheel<derivedT>::setupConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void setupConfig(mx::app::appConfigurator & config /**< [out] the derived classes configurator*/);

   /// load the configuration system results
   /**
     * This should be called in `derivedT::loadConfig` as
     * \code
       stdFilterWheel<derivedT>::loadConfig(config);
       \endcode
     * with appropriate error checking.
     */
   void loadConfig(mx::app::appConfigurator & config /**< [in] the derived classes configurator*/);

   /// Startup function
   /** 
     * This should be called in `derivedT::appStartup` as
     * \code
       stdFilterWheel<derivedT>::appStartup();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int appStartup();

   /// Application logic 
   /** Checks the stdFilterWheel thread
     * 
     * This should be called from the derived's appLogic() as in
     * \code
       stdFilterWheel<derivedT>::appLogic();
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
       stdFilterWheel<derivedT>::onPowerOff();
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
       stdFilterWheel<derivedT>::whilePowerOff();
       \endcode
     * with appropriate error checking.
     * 
     * \returns 0 on success
     * \returns -1 on error, which is logged.
     */
   int whilePowerOff();
   
   /// Application the shutdown 
   /** Shuts down the stdFilterWheel thread
     * 
     * \code
       stdFilterWheel<derivedT>::appShutdown();
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
   
   ///The position of the wheel in filters
   pcf::IndiProperty m_indiP_filter;

   ///The name of the nearest filter for this position
   pcf::IndiProperty m_indiP_filterName;

   ///Command the wheel to home.  Any change in this property causes a home.
   pcf::IndiProperty m_indiP_home;
   
   ///Command the wheel to halt.  Any change in this property causes an immediate halt.
   pcf::IndiProperty m_indiP_stop;

public:

   /// The static callback function to be registered for stdFilterWheel properties
   /** Dispatches to the relevant handler
     * 
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_stdFilterWheel( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                             const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                           );
   
   /// Callback to process a NEW filter position request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_filter( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
   /// Callback to process a NEW filter name request
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_filterName( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);
   
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
stdFilterWheel<derivedT>::~stdFilterWheel() noexcept
{
   return;
}



template<class derivedT>
void stdFilterWheel<derivedT>::setupConfig(mx::app::appConfigurator & config)
{
   static_cast<void>(config);
   
   config.add("motor.powerOnHome", "", "motor.powerOnHome", argType::Required, "motor", "powerOnHome", false, "bool", "If true, home at startup/power-on.  Default=false.");
   
   config.add("filters.names", "", "filters.names",  argType::Required, "filters", "names", false, "vector<string>", "The names of the filters.");
   config.add("filters.positions", "", "filters.positions",  argType::Required, "filters", "positions", false, "vector<double>", "The positions of the filters.  If omitted or 0 then order is used.");
   
}

template<class derivedT>
void stdFilterWheel<derivedT>::loadConfig(mx::app::appConfigurator & config)
{
   config(m_powerOnHome, "motor.powerOnHome");
   
   config(m_filterNames, "filters.names");
   m_filterPositions.resize(m_filterNames.size(), 0);
   for(size_t n=0;n<m_filterPositions.size();++n) m_filterPositions[n] = n+1;
   config(m_filterPositions, "filters.positions");
   for(size_t n=0;n<m_filterPositions.size();++n) if(m_filterPositions[n] == 0) m_filterPositions[n] = n+1;
}
   


template<class derivedT>
int stdFilterWheel<derivedT>::appStartup()
{
 
   derived().createStandardIndiNumber( m_indiP_filter, "filter", 1.0, (double) m_filterNames.size(), 0.0, "%0.3d");
   if( derived().registerIndiPropertyNew( m_indiP_filter, st_newCallBack_stdFilterWheel) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   derived().createStandardIndiSelectionSw( m_indiP_filterName,"filterName", m_filterNames);
   if( derived().registerIndiPropertyNew( m_indiP_filterName, st_newCallBack_stdFilterWheel) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   derived().createStandardIndiRequestSw( m_indiP_home, "home");
   if( derived().registerIndiPropertyNew( m_indiP_home, st_newCallBack_stdFilterWheel) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }

   derived().createStandardIndiRequestSw( m_indiP_stop, "stop");
   if( derived().registerIndiPropertyNew( m_indiP_stop, st_newCallBack_stdFilterWheel) < 0)
   {
      #ifndef STDFILTERWHEEL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   return 0;
}

template<class derivedT>
int stdFilterWheel<derivedT>::appLogic()
{
   return 0;

}

template<class derivedT>
int stdFilterWheel<derivedT>::onPowerOff()
{
   if( !derived().m_indiDriver ) return 0;
   return 0;
}

template<class derivedT>
int stdFilterWheel<derivedT>::whilePowerOff()
{
   return 0;
}

template<class derivedT>
int stdFilterWheel<derivedT>::appShutdown()
{
   return 0;
}


template<class derivedT>
int stdFilterWheel<derivedT>::st_newCallBack_stdFilterWheel( void * app,
                                                   const pcf::IndiProperty &ipRecv
                                                 )
{
   std::string name = ipRecv.getName();
   derivedT * _app = static_cast<derivedT *>(app);
   
   if(name == "stop") return _app->newCallBack_stop(ipRecv); //Check this first to make sure it 
   if(name == "home") return _app->newCallBack_home(ipRecv);
   if(name == "filter") return _app->newCallBack_filter(ipRecv);
   if(name == "filterName") return _app->newCallBack_filterName(ipRecv);
   
   return -1;
}

template<class derivedT>
int stdFilterWheel<derivedT>::newCallBack_filter( const pcf::IndiProperty &ipRecv )
{
   double target;
   
   if( derived().indiTargetUpdate( m_indiP_filter, target, ipRecv, true) < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_filter_target = target;
   
   std::lock_guard<std::mutex> guard(derived().m_indiMutex);
   return derived().moveTo(target);
   
}

template<class derivedT>
int stdFilterWheel<derivedT>::newCallBack_filterName( const pcf::IndiProperty &ipRecv )
{
   if(ipRecv.getName() != m_indiP_filterName.getName())
   {
      derivedT::template log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }
   
   //First we calculate current filter name
   int n = floor(m_filter + 0.5) - 1;
   if(n < 0)
   {
      while(n < 0) n += m_filterNames.size();
   }
   if( n > (long) m_filterNames.size()-1 )
   {
      while( n > (long) m_filterNames.size()-1 ) n -= m_filterNames.size();
   }
      
   std::string newName = "";
   int newn = -1;
   
   size_t i;
   for(i=0; i< m_filterNames.size(); ++i) 
   {
      if(!ipRecv.find(m_filterNames[i])) continue;
      
      if(ipRecv[m_filterNames[i]].getSwitchState() == pcf::IndiElement::On)
      {
         if(newName != "")
         {
            derivedT::template log<text_log>("More than one filter selected", logPrio::LOG_ERROR);
            return -1;
         }
         
         newName = m_filterNames[i];
         newn = i;
      }
   }
   
   if(newName == "" || newn < 0)
   {
      return 0; //This is just an reset of current probably
   }
   
   m_filter_target = (double) newn + 1.0;
   indi::updateIfChanged(m_indiP_filter, "target",  m_filter_target , derived().m_indiDriver,INDI_BUSY);
   
   std::lock_guard<std::mutex> guard(derived().m_indiMutex);
   return derived().moveTo(m_filter_target);
   
}

template<class derivedT>
int stdFilterWheel<derivedT>::newCallBack_home( const pcf::IndiProperty &ipRecv )
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
int stdFilterWheel<derivedT>::newCallBack_stop( const pcf::IndiProperty &ipRecv )
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
int stdFilterWheel<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;
   
   int n = floor(m_filter + 0.5) - 1;
   if(n < 0)
   {
      while(n < 0) n += m_filterNames.size();
   }
   if( n > (long) m_filterNames.size()-1 )
   {
      while( n > (long) m_filterNames.size()-1 ) n -= m_filterNames.size();
   }
   
   if( n < 0)
   {
      derivedT::template log<software_error>({__FILE__,__LINE__, "error calculating filter index, n < 0"});
      return -1;
   }
   
   size_t nn = n;
 
   //Check for changes and update the filterNames switch vector
   bool changed = false;
   
   static int last_moving = m_moving;
   
   if(last_moving != m_moving)
   {
      changed = true;
      last_moving = m_moving;
   }
   
   for(size_t i =0; i < m_filterNames.size(); ++i)
   {
      if( i == nn )
      {
         if(m_indiP_filterName[m_filterNames[i]] != pcf::IndiElement::On) 
         {
            changed = true;
            m_indiP_filterName[m_filterNames[i]] = pcf::IndiElement::On;
         }
      }
      else
      {
         if(m_indiP_filterName[m_filterNames[i]] != pcf::IndiElement::Off) 
         {
            changed = true;
            m_indiP_filterName[m_filterNames[i]] = pcf::IndiElement::Off;
         }
      }
   }
   if(changed)
   {
      if(m_moving)
      {
         m_indiP_filterName.setState(INDI_BUSY);
      }
      else
      {
         m_indiP_filterName.setState(INDI_IDLE);
      }
            
      derived().m_indiDriver->sendSetProperty(m_indiP_filterName);
   }
   
  
   
   if(m_moving)
   {
      indi::updateIfChanged(m_indiP_filter, "current", m_filter, derived().m_indiDriver,INDI_BUSY);
      indi::updateIfChanged(m_indiP_filter, "target", m_filter_target, derived().m_indiDriver,INDI_BUSY);
   }
   else
   {
      indi::updateIfChanged(m_indiP_filter, "current", m_filter, derived().m_indiDriver,INDI_IDLE);    
      indi::updateIfChanged(m_indiP_filter, "target", m_filter_target, derived().m_indiDriver,INDI_IDLE);
   }
   
   
   return 0;
}


} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //stdFilterWheel_hpp
