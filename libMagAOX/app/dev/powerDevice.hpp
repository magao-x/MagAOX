/** \file powerDevice.hpp
 * \author Jared R. Males
 * \brief Declares and defines a power control device framework in the MagAOXApp context
 * 
 * \ingroup 
 *
 */

#ifndef app_powerDevice_hpp
#define app_powerDevice_hpp

#include <mx/mxlib.hpp>
#include <mx/app/application.hpp>


using namespace mx::app;

#define PWR_STATE_UNKNOWN (-1)
#define PWR_STATE_OFF (0)
#define PWR_STATE_INTERMEDIATE (1)
#define PWR_STATE_IN (2)


#define PWR_E_NOOUTLETS (-10)
#define PWR_E_NOCHANNELS (-15)
#define PWR_E_NOVALIDCH (-20)

namespace MagAOX
{
namespace app
{
namespace dev 
{


   
/// A generic power controller
/**   
  * \ingroup magaoxapp
  */ 
struct powerDevice
{   
   std::vector<int> m_outletStates;
 
   struct channelSpec
   {
      std::vector<size_t> m_outlets;
   
      std::vector<size_t> m_onOrder;
      std::vector<size_t> m_offOrder;
   
      std::vector<unsigned> m_onDelays;
      std::vector<unsigned> m_offDelays;
   
      int m_state{PWR_STATE_UNKNOWN};
   };

   int m_numberChannels;
 
   std::unordered_map<std::string, channelSpec> m_channels;

   
   ///Setup an application configurator for the [power] section
   /**
     *
     */ 
   int setupConfig( appConfigurator & config /**< [in] an application configuration to setup */);

   ///Load the [power] and [channelN] sections from an application configurator
   /**
     *
     */
   int loadConfig( appConfigurator & config /**< [in] an application configuration from which to load values */);
   
   
   int setNumberOfOutlets( int numOuts );
   
   /// Get the currently stored outlet state, without updating from device.
   int outletState( int outletNum );
   
   /// Get the state of the outlet from the device.
   virtual int updateOutletState( int outletNum ) = 0;
   
   /// Get the states of all outlets from the device.
   virtual int updateOutletStates();
   
   virtual int turnOutletOn( int outletNum ) = 0;
   
   virtual int turnOutletOff( int outletNum ) = 0;
   
   size_t numChannels();
   
   std::vector<size_t> channelOutlets( const std::string & channel );
   std::vector<size_t> channelOnOrder( const std::string & channel );
   std::vector<size_t> channelOffOrder( const std::string & channel );
   std::vector<unsigned> channelOnDelays( const std::string & channel );
   std::vector<unsigned> channelOffDelays( const std::string & channel );
   
   int channelState( const std::string & channel );
   
   int turnChannelOn( const std::string & channel );
   
   int turnChannelOff( const std::string & channel );
   
   
};

inline 
int powerDevice::setupConfig( appConfigurator & config )
{
   static_cast<void>(config);
   
   //config.add("power.numOutlets", "", "", mx::argType::Required, "power", "nuOutlets", true, "int", "number of outlets controlled by this device");
   return 0;
}

inline 
int powerDevice::loadConfig( appConfigurator & config )
{
   if( m_outletStates.size() == 0) return PWR_E_NOOUTLETS;
   
   
   //Get the "unused" sections.
   std::vector<std::string> sections;
   
   config.unusedSections(sections);
   
   if( sections.size() == 0 ) return PWR_E_NOCHANNELS;
      
   //Now see if any are channels, which means they have an outlet= or outlets= entry
   std::vector<std::string> chSections;
   
   for(size_t i=0;i<sections.size(); ++i)
   {
      if( config.isSetUnused( iniFile::makeKey(sections[i], "outlet" ))
              || config.isSetUnused( iniFile::makeKey(sections[i], "outlets" )) ) 
      {
         chSections.push_back(sections[i]);
      }
   }
   
   if( chSections.size() == 0 ) return PWR_E_NOVALIDCH;
   
   //Now configure the chanels.
   for(size_t n = 0; n < chSections.size(); ++n)
   {
      m_channels.emplace( chSections[n] , channelSpec());
      
      //---- Set outlets ----
      std::vector<size_t> outlets;
      if( config.isSetUnused( iniFile::makeKey(chSections[n], "outlet" )))
      {
         config.configUnused( outlets, iniFile::makeKey(chSections[n], "outlet" ) );
      }
      else
      {
         config.configUnused( outlets, iniFile::makeKey(chSections[n], "outlets" ) );
      }
      
      m_channels[chSections[n]].m_outlets = outlets;
      ///\todo error checking on outlets

      //---- Set optional configs ----
      if( config.isSetUnused( iniFile::makeKey(chSections[n], "onOrder" )))
      {
         std::vector<size_t> onOrder;
         config.configUnused( onOrder, iniFile::makeKey(chSections[n], "onOrder" ) );
         m_channels[chSections[n]].m_onOrder = onOrder;
         ///\todo error checking on onOrder
      }
      
      if( config.isSetUnused( iniFile::makeKey(chSections[n], "offOrder" )))
      {
         std::vector<size_t> offOrder;
         config.configUnused( offOrder, iniFile::makeKey(chSections[n], "offOrder" ) );
         m_channels[chSections[n]].m_offOrder = offOrder;
         ///\todo error checking on offOrder
      }
      
      if( config.isSetUnused( iniFile::makeKey(chSections[n], "onDelays" )))
      {
         std::vector<unsigned> onDelays;
         config.configUnused( onDelays, iniFile::makeKey(chSections[n], "onDelays" ) );
         m_channels[chSections[n]].m_onDelays = onDelays;
         ///\todo error checking on onDelays
      }
      
      if( config.isSetUnused( iniFile::makeKey(chSections[n], "offDelays" )))
      {
         std::vector<unsigned> offDelays;
         config.configUnused( offDelays, iniFile::makeKey(chSections[n], "offDelays" ) );
         m_channels[chSections[n]].m_offDelays = offDelays;
         ///\todo error checking on offDelays
      }
   }
   
   return 0;
}

inline 
int powerDevice::setNumberOfOutlets( int numOuts )
{
   m_outletStates.resize(numOuts, -1);
   return 0;
}
   
inline 
int powerDevice::outletState( int outletNum )
{
   return m_outletStates[outletNum];
}

inline
int powerDevice::updateOutletStates()
{
   for(size_t n=0; n<m_outletStates.size(); ++n)
   {
      int rv = updateOutletState(n);
      if(rv < 0) return rv;
   }
   
   return 0;
}

inline
size_t powerDevice::numChannels()
{
   return m_channels.size();
}

inline 
std::vector<size_t> powerDevice::channelOutlets( const std::string & channel )
{
   return m_channels[channel].m_outlets;
}

inline 
std::vector<size_t> powerDevice::channelOnOrder( const std::string & channel )
{
   return m_channels[channel].m_onOrder;
}

inline 
std::vector<size_t> powerDevice::channelOffOrder( const std::string & channel )
{
   return m_channels[channel].m_offOrder;
}

inline 
std::vector<unsigned> powerDevice::channelOnDelays( const std::string & channel )
{
   return m_channels[channel].m_onDelays;
}

inline 
std::vector<unsigned> powerDevice::channelOffDelays( const std::string & channel )
{
   return m_channels[channel].m_offDelays;
}

inline 
int powerDevice::channelState( const std::string & channel ) 
{
   int st = outletState(m_channels[channel].m_outlets[0]);
   
   for( size_t n = 1; n < m_channels[channel].m_outlets.size(); ++n )
   {
      if( st != outletState(m_channels[channel].m_outlets[n]) ) st = 1;
   }
   
   return st;
}

inline 
int powerDevice::turnChannelOn( const std::string & channel )
{
   //If order is specified, get first outlet number
   size_t n = 0;
   if( m_channels[channel].m_onOrder.size() > 0 ) n = m_channels[channel].m_onOrder[0];

   //turn on first outlet.
   turnOutletOn(m_channels[channel].m_outlets[n]);
   
   for(size_t i = 1; i< m_channels[channel].m_outlets.size(); ++i)
   {
      //Do the delay here...
      
      n=i;
      if( m_channels[channel].m_onOrder.size() > 0 ) n = m_channels[channel].m_onOrder[i];
      
      turnOutletOn(m_channels[channel].m_outlets[n]);
   }
   
   return 0;
}

inline 
int powerDevice::turnChannelOff( const std::string & channel )
{
   //If order is specified, get first outlet number
   size_t n = 0;
   if( m_channels[channel].m_offOrder.size() > 0 ) n = m_channels[channel].m_offOrder[0];

   //turn on first outlet.
   turnOutletOff(m_channels[channel].m_outlets[n]);
   
   for(size_t i = 1; i< m_channels[channel].m_outlets.size(); ++i)
   {
      //Do the delay here...
      
      n=i;
      if( m_channels[channel].m_offOrder.size() > 0 ) n = m_channels[channel].m_offOrder[i];
      
      turnOutletOff(m_channels[channel].m_outlets[n]);
   }
   
   return 0;
}

} //namespace dev 
} //namespace app 
} //namespace MagAOX

#endif //app_powerDevice_hpp 

