/** \file outletController.hpp
 * \author Jared R. Males
 * \brief Declares and defines a power control device framework in the MagAOXApp context
 *
 * \ingroup
 *
 */

#ifndef app_outletController_hpp
#define app_outletController_hpp

#include <mx/app/application.hpp>
#include <mx/sys/timeUtils.hpp>


#include "../../INDI/libcommon/IndiProperty.hpp"
#include "../../libMagAOX/libMagAOX.hpp"
#include "../indiUtils.hpp"


#define OUTLET_STATE_UNKNOWN (-1)
#define OUTLET_STATE_OFF (0)
#define OUTLET_STATE_INTERMEDIATE (1)
#define OUTLET_STATE_ON (2)


#define OUTLET_E_NOOUTLETS (-10)
#define OUTLET_E_NOCHANNELS (-15)
#define OUTLET_E_NOVALIDCH (-20)

namespace MagAOX
{
namespace app
{
namespace dev
{

/// A generic outlet controller
/** Controls a set of outlets on a device, such as A/C power outlets or digital outputs.
  * The outlets are organized into channels, which could be made up of multiple outlets.
  *
  * derivedT must be a MagAOXApp, and additionally it must implement the functions
  * \code
    int turnOutletOn( int outletNum );

    int turnOutletOff( int outletNum );

    int updateOutletState( int outletNum );
  * \endcode 
  * and optionally
  * \code
    int updateOutletStates();
    \endcode
  *
  * Other requirements:
  * - call `setNumberOfOutlets` in the derived class constructor
  * 
  * 
  * \tparam derivedT specifies a MagAOXApp parent base class which is accessed with a `static_cast` (downcast)
  *         to perform various methods.
  *
  * \ingroup appdev
  *
  * \todo finalize 0-counting vs 1-counting rules --> do we subtract one from config-ed outlet indices if m_firstOne is true?
  * 
  */
template<class derivedT>
struct outletController
{
   bool m_firstOne {false}; ///< Flag is true if the first outlet is numbered 1, otherwise assumes starting at 0.

   std::vector<int> m_outletStates; ///< The current states of each outlet.  These MUST be updated by derived classes in the overridden \ref updatedOutletState.

   pcf::IndiProperty m_indiP_outletStates; ///< Indi Property to show individual outlet states.

   /// Structure containing the specification of one channel.
   /** A channel may include more than one outlet, may specify the order in which
     * outlets are turned on and/or off, and may specify a delay between turning outlets on
     * and/or off.
     */
   struct channelSpec
   {
      std::vector<size_t> m_outlets; ///< The outlets in this channel

      std::vector<size_t> m_onOrder; ///< [optional] The order in which outlets are turned on.  This contains the indices of m_outlets, not the outlet numbers of the device.
      std::vector<size_t> m_offOrder; ///< [optional] The order in which outlets are turned off.  This contains the indices of m_outlets, not the outlet numbers of the device.

      std::vector<unsigned> m_onDelays; ///< [optional] The delays between outlets in a multi-oultet channel.  The first entry is always ignored.  The second entry is the dealy between the first and second outlet, etc.
      std::vector<unsigned> m_offDelays; ///< [optional] The delays between outlets in a multi-oultet channel.  The first entry is always ignored.  The second entry is the dealy between the first and second outlet, etc.

      pcf::IndiProperty m_indiP_prop;
   };

   /// The map of channel specifications, which can be accessed by their names.
   std::unordered_map<std::string, channelSpec> m_channels;

   /// An INDI property which publishes the outlets associated with each channel.  Useful for GUIs, etc.
   pcf::IndiProperty m_indiP_chOutlets;
   
   /// An INDI property which publishes the total on delay for each channel.  Useful for GUIs, etc.
   pcf::IndiProperty m_indiP_chOnDelays;
   
   /// An INDI property which publishes the total off delay for each channel.  Useful for GUIs, etc.
   pcf::IndiProperty m_indiP_chOffDelays;
   
   
   ///Setup an application configurator for an outletController
   /** This is currently a no-op
     *
     * \returns 0 on success
     * \returns -1 on failure
     */
   int setupConfig( mx::app::appConfigurator & config /**< [in] an application configuration to setup */);

   /// Load the [channel] sections from an application configurator
   /** Any "unused" section from the config parser is analyzed to determine if it is a channel specification.
     * If it contains the `outlet` or `outlets` keyword, then it is considered a channel. `outlet` and `outlets`
     * are equivalent, and specify the one or more device outlets included in this channel (i.e. this may be a vector
     * value entry).
     *
     * This function then looks for `onOrder` and `offOrder` keywords, which specify the order outlets are turned
     * on or off by their indices in the vector specified by the `outlet`/`outlets` keyword (i.e not the outlet numbers).
     *
     * Next it looks for `onDelays` and `offDelays`, which specify the delays between outlet operations in milliseconds.
     * The first entry is always ignored, then the second entry specifies the delay between the first and second outlet
     * operation, etc.
     *
     * An example config file section is:
     \verbatim
     [sue]           #this channel will be named sue
     outlets=4,5     #this channel uses outlets 4 and 5
     onOrder=1,0     #outlet 5 will be turned on first
     offOrder=0,1    #Outlet 4 will be turned off first
     onDelays=0,150  #a 150 msec delay between outlet turn on
     offDelays=0,345 #a 345 msec delay between outlet turn off
     \endverbatim
     *
     * \returns 0 on success
     * \returns -1 on failure
     */
   int loadConfig( mx::app::appConfigurator & config /**< [in] an application configuration from which to load values */);

   /// Sets the number of outlets.  This should be called by the derived class constructor.
   /**
     * \returns 0 on success
     * \returns -1 on failure
     */
   int setNumberOfOutlets( int numOuts /**< [in] the number of outlets to allocate */);

   /// Get the currently stored outlet state, without updating from device.
   int outletState( int outletNum );

   /// Get the state of the outlet from the device.
   /** This will be implemented in derived classes to update the outlet state.
     * \todo this is declared pure virtual, but there is an implementation.
     * \returns 0 on success.
     * \returns -1 on error.
     */
   virtual int updateOutletState( int outletNum /**< [in] the outlet number to update */) = 0;

   /// Get the states of all outlets from the device.
   /** The default implementation for-loops through each outlet, calling \ref updateOutletState.
     * Can be re-implemented in derived classes to update the outlet states.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   virtual int updateOutletStates();

   /// Turn an outlet on.
   /** This will be implemented in derived classes to turn an outlet on.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   virtual int turnOutletOn( int outletNum /**< [in] the outlet number to turn on */) = 0;

   /// Turn an outlet off.
   /** This will be implemented in derived classes to turn an outlet off.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   virtual int turnOutletOff( int outletNum /**< [in] the outlet number to turn off */) = 0;

   /// Get the number of channels
   /**
     * \returns the number of entries in m_channels.
     */
   size_t numChannels();

   /// Get the vector of outlet indices for a channel.
   /** Mainly used for testing.
     *
     * \returns the m_outlets member of the channelSpec specified by its name.
     */
   std::vector<size_t> channelOutlets( const std::string & channel /**< [in] the name of the channel */);

   /// Get the vector of outlet on orders for a channel.
   /** Mainly used for testing.
     *
     * \returns the m_onOrder member of the channelSpec specified by its name.
     */
   std::vector<size_t> channelOnOrder( const std::string & channel /**< [in] the name of the channel */);

   /// Get the vector of outlet off orders for a channel.
   /** Mainly used for testing.
     *
     * \returns the m_offOrder member of the channelSpec specified by its name.
     */
   std::vector<size_t> channelOffOrder( const std::string & channel /**< [in] the name of the channel */);

   /// Get the vector of outlet on delays for a channel.
   /** Mainly used for testing.
     *
     * \returns the m_onDelays member of the channelSpec specified by its name.
     */
   std::vector<unsigned> channelOnDelays( const std::string & channel /**< [in] the name of the channel */);

   /// Get the vector of outlet off delays for a channel.
   /** Mainly used for testing.
     *
     * \returns the m_offDelays member of the channelSpec specified by its name.
     */
   std::vector<unsigned> channelOffDelays( const std::string & channel /**< [in] the name of the channel */);

   /// Get the state of a channel.
   /**
     * \returns OUTLET_STATE_UNKNOWN if the state is not known
     * \returns OUTLET_STATE_OFF if the channel is off (all outlets off)
     * \returns OUTLET_STATE_INTERMEDIATE if outlets are intermediate or not in the same state
     * \returns OUTLET_STATE_ON if channel is on (all outlets on)
     */
   int channelState( const std::string & channel /**< [in] the name of the channel */);

   /// Turn a channel on.
   /** This implements the outlet order and delay logic.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int turnChannelOn( const std::string & channel /**< [in] the name of the channel to turn on*/);

   /// Turn a channel off.
   /** This implements the outlet order and delay logic.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int turnChannelOff( const std::string & channel /**< [in] the name of the channel to turn on*/);


   /** \name INDI Setup
     *@{
     */

   /// The static callback function to be registered for the channel properties.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   static int st_newCallBack_channels( void * app, ///< [in] a pointer to this, will be static_cast-ed to derivedT.
                                       const pcf::IndiProperty &ipRecv ///< [in] the INDI property sent with the the new property request.
                                     );

   /// The callback called by the static version, to actually process the new request.
   /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int newCallBack_channels( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the the new property request.*/);

   /// Setup the INDI properties for this device controller
   /** This should be called in the `appStartup` function of the derived MagAOXApp.
     * \todo change this to be appStartup like other devs.
     * \returns 0 on success.
     * \returns -1 on error.
     */
   int setupINDI();

   /// Update the INDI properties for this device controller
   /** You should call this after updating the outlet states.
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
int outletController<derivedT>::setupConfig( mx::app::appConfigurator & config )
{
   static_cast<void>(config);

   return 0;
}

template<class derivedT>
int outletController<derivedT>::loadConfig( mx::app::appConfigurator & config )
{
   if( m_outletStates.size() == 0) return OUTLET_E_NOOUTLETS;

   //Get the "unused" sections.
   std::vector<std::string> sections;

   config.unusedSections(sections);

   if( sections.size() == 0 ) return OUTLET_E_NOCHANNELS;

   //Now see if any are channels, which means they have an outlet= or outlets= entry
   std::vector<std::string> chSections;

   for(size_t i=0;i<sections.size(); ++i)
   {
      if( config.isSetUnused( mx::app::iniFile::makeKey(sections[i], "outlet" ))
              || config.isSetUnused( mx::app::iniFile::makeKey(sections[i], "outlets" )) )
      {
         chSections.push_back(sections[i]);
      }
   }

   if( chSections.size() == 0 ) return OUTLET_E_NOVALIDCH;

   //Now configure the chanels.
   for(size_t n = 0; n < chSections.size(); ++n)
   {
      m_channels.emplace( chSections[n] , channelSpec());

      //---- Set outlets ----
      std::vector<size_t> outlets;
      if( config.isSetUnused( mx::app::iniFile::makeKey(chSections[n], "outlet" )))
      {
         config.configUnused( outlets, mx::app::iniFile::makeKey(chSections[n], "outlet" ) );
      }
      else
      {
         config.configUnused( outlets, mx::app::iniFile::makeKey(chSections[n], "outlets" ) );
      }

      //Subtract one if the device numbers from 1.
      for(size_t k=0;k<outlets.size(); ++k)
      {
         ///\todo test this error
         if( (int) outlets[k] - m_firstOne < 0 || (int) outlets[k] - m_firstOne > (int) m_outletStates.size())
         {
            #ifndef OUTLET_CTRL_TEST_NOLOG
            return derivedT::template log<text_log,-1>("Outlet " + std::to_string(outlets[k]) + " in Channel " + chSections[n] + " is invalid", logPrio::LOG_ERROR);
            #else
            return -1;
            #endif
         }

         outlets[k] -= m_firstOne;
      }

      m_channels[chSections[n]].m_outlets = outlets;
      ///\todo error checking on outlets

      //---- Set optional configs ----
      if( config.isSetUnused( mx::app::iniFile::makeKey(chSections[n], "onOrder" )))
      {
         std::vector<size_t> onOrder;
         config.configUnused( onOrder, mx::app::iniFile::makeKey(chSections[n], "onOrder" ) );
         
         ///\todo test this error
         if(onOrder.size() != m_channels[chSections[n]].m_outlets.size())
         {
            #ifndef OUTLET_CTRL_TEST_NOLOG
            return derivedT::template log<text_log,-1>("onOrder be same size as outlets.  In Channel " + chSections[n], logPrio::LOG_ERROR);
            #else
            return -1;
            #endif
         }
            
         m_channels[chSections[n]].m_onOrder = onOrder;
      }

      if( config.isSetUnused( mx::app::iniFile::makeKey(chSections[n], "offOrder" )))
      {
         std::vector<size_t> offOrder;
         config.configUnused( offOrder, mx::app::iniFile::makeKey(chSections[n], "offOrder" ) );
         m_channels[chSections[n]].m_offOrder = offOrder;
         ///\todo error checking on offOrder, should complain if not same length
      }

      if( config.isSetUnused( mx::app::iniFile::makeKey(chSections[n], "onDelays" )))
      {
         std::vector<unsigned> onDelays;
         config.configUnused( onDelays, mx::app::iniFile::makeKey(chSections[n], "onDelays" ) );
         m_channels[chSections[n]].m_onDelays = onDelays;
         ///\todo error checking on onDelays, should complain if not same length
      }

      if( config.isSetUnused( mx::app::iniFile::makeKey(chSections[n], "offDelays" )))
      {
         std::vector<unsigned> offDelays;
         config.configUnused( offDelays, mx::app::iniFile::makeKey(chSections[n], "offDelays" ) );
         m_channels[chSections[n]].m_offDelays = offDelays;
         ///\todo error checking on offDelays, should complain if not same length
      }
   }

   return 0;
}

template<class derivedT>
int outletController<derivedT>::setNumberOfOutlets( int numOuts )
{
   m_outletStates.resize(numOuts, -1);
   return 0;
}

template<class derivedT>
int outletController<derivedT>::outletState( int outletNum )
{
   return m_outletStates[outletNum];
}

template<class derivedT>
int outletController<derivedT>::updateOutletStates()
{
   for(size_t n=0; n<m_outletStates.size(); ++n)
   {
      int rv = updateOutletState(n);
      if(rv < 0) return rv;
   }

   return 0;
}

template<class derivedT>
size_t outletController<derivedT>::numChannels()
{
   return m_channels.size();
}

template<class derivedT>
std::vector<size_t> outletController<derivedT>::channelOutlets( const std::string & channel )
{
   return m_channels[channel].m_outlets;
}

template<class derivedT>
std::vector<size_t> outletController<derivedT>::channelOnOrder( const std::string & channel )
{
   return m_channels[channel].m_onOrder;
}

template<class derivedT>
std::vector<size_t> outletController<derivedT>::channelOffOrder( const std::string & channel )
{
   return m_channels[channel].m_offOrder;
}

template<class derivedT>
std::vector<unsigned> outletController<derivedT>::channelOnDelays( const std::string & channel )
{
   return m_channels[channel].m_onDelays;
}

template<class derivedT>
std::vector<unsigned> outletController<derivedT>::channelOffDelays( const std::string & channel )
{
   return m_channels[channel].m_offDelays;
}

template<class derivedT>
int outletController<derivedT>::channelState( const std::string & channel )
{
   int st = outletState(m_channels[channel].m_outlets[0]);

   for( size_t n = 1; n < m_channels[channel].m_outlets.size(); ++n )
   {
      if( st != outletState(m_channels[channel].m_outlets[n]) ) st = 1;
   }

   return st;
}

template<class derivedT>
int outletController<derivedT>::turnChannelOn( const std::string & channel )
{

   #ifndef OUTLET_CTRL_TEST_NOLOG
   derivedT::template log<software_debug>({__FILE__, __LINE__, "turning on channel " + channel});
   #endif

   #ifndef OUTLET_CTRL_TEST_NOINDI
   //m_channels[channel].m_indiP_prop["target"].setValue("On");
   indi::updateIfChanged(m_channels[channel].m_indiP_prop, "target", std::string("On"), derived().m_indiDriver, INDI_BUSY );
   #endif

   //If order is specified, get first outlet number
   size_t n = 0;
   if( m_channels[channel].m_onOrder.size() == m_channels[channel].m_outlets.size() ) n = m_channels[channel].m_onOrder[0];

   //turn on first outlet.
   if( turnOutletOn(m_channels[channel].m_outlets[n]) < 0 )
   {

      #ifndef OUTLET_CTRL_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__, __LINE__, "error turning on outlet " + std::to_string(n)});
      #else
         std::cerr << "Failed to turn on outlet " << n << "\n";
      #endif

      return -1;
   }

   #ifndef OUTLET_CTRL_TEST_NOLOG
   derivedT::template log<outlet_state>({ (uint8_t) (n  + m_firstOne), 2});
   #endif

   //Now do the rest
   for(size_t i = 1; i< m_channels[channel].m_outlets.size(); ++i)
   {
      //If order is specified, get next outlet number
      n=i;
      if( m_channels[channel].m_onOrder.size() == m_channels[channel].m_outlets.size() ) n = m_channels[channel].m_onOrder[i];

      //Delay if specified
      if( m_channels[channel].m_onDelays.size() == m_channels[channel].m_outlets.size() )
      {
         mx::sys::milliSleep(m_channels[channel].m_onDelays[i]);
      }

      //turn on next outlet

      if( turnOutletOn(m_channels[channel].m_outlets[n]) < 0 )
      {

         #ifndef OUTLET_CTRL_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__, __LINE__, "error turning on outlet " + std::to_string(n)});
         #else
         std::cerr << "Failed to turn on outlet " << n << "\n";
         #endif

         return -1;
      }

      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<outlet_state>({ (uint8_t) (n  + m_firstOne ), 2});
      #endif
   }

   #ifndef OUTLET_CTRL_TEST_NOLOG
   derivedT::template log<outlet_channel_state>({ channel, 2});
   #endif

   return 0;
}

template<class derivedT>
int outletController<derivedT>::turnChannelOff( const std::string & channel )
{

   #ifndef OUTLET_CTRL_TEST_NOLOG
   derivedT::template log<software_debug>({__FILE__, __LINE__, "turning off channel " + channel});
   #endif

   #ifndef OUTLET_CTRL_TEST_NOINDI
   //m_channels[channel].m_indiP_prop["target"].setValue("Off");
   indi::updateIfChanged(m_channels[channel].m_indiP_prop, "target", std::string("Off"), derived().m_indiDriver, INDI_BUSY );
   #endif

   //If order is specified, get first outlet number
   size_t n = 0;
   if( m_channels[channel].m_offOrder.size() == m_channels[channel].m_outlets.size() ) n = m_channels[channel].m_offOrder[0];

   //turn off first outlet.
   if( turnOutletOff(m_channels[channel].m_outlets[n]) < 0 )
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__, __LINE__, "error turning off outlet " + std::to_string(n)});
      #else
         std::cerr << "Failed to turn off outlet " << n << "\n";
      #endif

      return -1;
   }

   #ifndef OUTLET_CTRL_TEST_NOLOG
   derivedT::template log<outlet_state>({ (uint8_t) (n  + m_firstOne), 0});
   #endif

   //Now do the rest
   for(size_t i = 1; i< m_channels[channel].m_outlets.size(); ++i)
   {
      //If order is specified, get next outlet number
      n=i;
      if( m_channels[channel].m_offOrder.size() == m_channels[channel].m_outlets.size() ) n = m_channels[channel].m_offOrder[i];

      //Delay if specified
      if( m_channels[channel].m_offDelays.size() == m_channels[channel].m_outlets.size() )
      {
         mx::sys::milliSleep(m_channels[channel].m_offDelays[i]);
      }

      //turn off next outlet
      if( turnOutletOff(m_channels[channel].m_outlets[n]) < 0 )
      {
         #ifndef OUTLET_CTRL_TEST_NOLOG
            derivedT::template log<software_error>({__FILE__, __LINE__, "error turning off outlet " + std::to_string(n)});
         #else
            std::cerr << "Failed to turn off outlet " << n << "\n";
         #endif

         return -1;
      }

      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<outlet_state>({ (uint8_t) (n + m_firstOne), 0});
      #endif
   }

   #ifndef OUTLET_CTRL_TEST_NOLOG
   derivedT::template log<outlet_channel_state>({ channel, 0});
   #endif

   return 0;
}

template<class derivedT>
int outletController<derivedT>::st_newCallBack_channels( void * app,
                                                         const pcf::IndiProperty &ipRecv
                                                       )
{
   return static_cast<derivedT *>(app)->newCallBack_channels(ipRecv);
}

template<class derivedT>
int outletController<derivedT>::newCallBack_channels( const pcf::IndiProperty &ipRecv )
{
   //Check if we're in state READY before doing anything
   if(derived().state() != stateCodes::READY)
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<text_log>("can't change outlet state when not READY", logPrio::LOG_ERROR);
      #endif
   
      return -1;
   }
   
   //Interogate ipRecv to figure out which channel it is.
   //And then call turn on or turn off based on requested state.
   std::string name = ipRecv.getName();

   std::string state, target;

   if(ipRecv.find("state"))
   {
      state = ipRecv["state"].get<std::string>();
   }

   if(ipRecv.find("target"))
   {
      target = ipRecv["target"].get<std::string>();
   }

   if( target == "" ) target = state;
   target = mx::ioutils::toUpper(target);

   if( target == "ON" )
   {
      return turnChannelOn(name);
   }

   if(target == "OFF")
   {
      return turnChannelOff(name);
   }

   return 0;
}

template<class derivedT>
int outletController<derivedT>::setupINDI()
{
   //Register the static INDI properties
   m_indiP_chOutlets = pcf::IndiProperty(pcf::IndiProperty::Text);
   m_indiP_chOutlets.setDevice(derived().configName());
   m_indiP_chOutlets.setName("channelOutlets");
   m_indiP_chOutlets.setPerm(pcf::IndiProperty::ReadOnly);
   m_indiP_chOutlets.setState(pcf::IndiProperty::Idle);
    
   if(derived().registerIndiPropertyReadOnly(m_indiP_chOutlets) < 0)
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});   
      #endif
      return -1;
   }  
   
   m_indiP_chOnDelays = pcf::IndiProperty (pcf::IndiProperty::Number);
   m_indiP_chOnDelays.setDevice(derived().configName());
   m_indiP_chOnDelays.setName("channelOnDelays");
   m_indiP_chOnDelays.setPerm(pcf::IndiProperty::ReadOnly);
   m_indiP_chOnDelays.setState(pcf::IndiProperty::Idle);
   
   if(derived().registerIndiPropertyReadOnly(m_indiP_chOnDelays) < 0)
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
      
   m_indiP_chOffDelays = pcf::IndiProperty (pcf::IndiProperty::Number);
   m_indiP_chOffDelays.setDevice(derived().configName());
   m_indiP_chOffDelays.setName("channelOffDelays");
   m_indiP_chOffDelays.setPerm(pcf::IndiProperty::ReadOnly);
   m_indiP_chOffDelays.setState(pcf::IndiProperty::Idle);
   
   if(derived().registerIndiPropertyReadOnly(m_indiP_chOffDelays) < 0)
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
   
   //Create channel properties and register callback.
   for(auto it = m_channels.begin(); it != m_channels.end(); ++it)
   {
      it->second.m_indiP_prop = pcf::IndiProperty (pcf::IndiProperty::Text);
      it->second.m_indiP_prop.setDevice(derived().configName());
      it->second.m_indiP_prop.setName(it->first);
      it->second.m_indiP_prop.setPerm(pcf::IndiProperty::ReadWrite);
      it->second.m_indiP_prop.setState( pcf::IndiProperty::Idle );

      //add elements 'state' and 'target'
      it->second.m_indiP_prop.add (pcf::IndiElement("state"));
      it->second.m_indiP_prop.add (pcf::IndiElement("target"));

      if( derived().registerIndiPropertyNew( it->second.m_indiP_prop, st_newCallBack_channels) < 0)
      {
         #ifndef OUTLET_CTRL_TEST_NOLOG
         derivedT::template log<software_error>({__FILE__,__LINE__});
         #endif
         return -1;
      }
      
      //Load values into the static INDI properties
      m_indiP_chOutlets.add(pcf::IndiElement(it->first));
      std::string os = std::to_string(it->second.m_outlets[0]);
      for(size_t i=1;i< it->second.m_outlets.size();++i) os += "," + std::to_string(it->second.m_outlets[i]);
      m_indiP_chOutlets[it->first].set(os);
      
      m_indiP_chOnDelays.add(pcf::IndiElement(it->first));
      double sum=0;
      for(size_t i=0;i< it->second.m_onDelays.size();++i) sum += it->second.m_onDelays[i];
      m_indiP_chOnDelays[it->first].set(sum);
      
      m_indiP_chOffDelays.add(pcf::IndiElement(it->first));
      sum=0;
      for(size_t i=0;i< it->second.m_offDelays.size();++i) sum += it->second.m_offDelays[i];
      m_indiP_chOffDelays[it->first].set(sum);
      
   }
   
   //Register the outletStates INDI property, and add an element for each outlet.
   m_indiP_outletStates = pcf::IndiProperty (pcf::IndiProperty::Text);
   m_indiP_outletStates.setDevice(derived().configName());
   m_indiP_outletStates.setName("outlet");
   m_indiP_outletStates.setPerm(pcf::IndiProperty::ReadWrite);
   m_indiP_outletStates.setState( pcf::IndiProperty::Idle );

   if( derived().registerIndiPropertyReadOnly(m_indiP_outletStates) < 0)
   {
      #ifndef OUTLET_CTRL_TEST_NOLOG
      derivedT::template log<software_error>({__FILE__,__LINE__});
      #endif
      return -1;
   }
/*      
   auto result =  derived().m_indiNewCallBacks.insert( { "outlet", {&m_indiP_outletStates, nullptr}});

   if(!result.second)
   {
      return -1;
   }*/

   for(size_t i=0; i< m_outletStates.size(); ++i)
   {
      m_indiP_outletStates.add (pcf::IndiElement(std::to_string(i+m_firstOne)));
   }

   return 0;
}

std::string stateIntToString(int st);

template<class derivedT>
int outletController<derivedT>::updateINDI()
{
   if( !derived().m_indiDriver ) return 0;

   //Publish outlet states (only bother if they've changed)
   for(size_t i=0; i< m_outletStates.size(); ++i)
   {
      indi::updateIfChanged(m_indiP_outletStates, std::to_string(i+m_firstOne), stateIntToString(m_outletStates[i]), derived().m_indiDriver);
   }

   //Publish channel states (only bother if they've changed)
   for(auto it = m_channels.begin(); it != m_channels.end(); ++it)
   {
      std::string state = stateIntToString( channelState( it->first ));

      indi::updateIfChanged( it->second.m_indiP_prop, "state", state, derived().m_indiDriver );
   }

   
   
   return 0;
}

} //namespace dev
} //namespace app
} //namespace MagAOX

#endif //app_outletController_hpp
