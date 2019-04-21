/** \file xtChannels.hpp
  * \brief Utility class for managing Acromag xt12XX digital I/O channels
  *
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup xt1211Ctrl_files
  * 
  * History:
  *  -- Created 2019-04-21 by JRM
  */

#ifndef xtChannels_hpp
#define xtChannels_hpp


/// Utility class for managing Acromag xt12XX digital I/O channels
/** Reads and writes input "registers" (uint16_t) for modbus protocol.
  *
  * \tparam _numChannels is the number of channels controlled by this device.
  */ 
template<int _numChannels>
struct xtChannels
{
   static constexpr int numChannels = _numChannels; ///< The number of channels controlled by the device
   
   static constexpr int numRegisters = _numChannels/4; ///< The number of registers needed for the number of channels.
   
protected:
   bool m_channels[numChannels]; ///< The state of the channels
   
   bool m_inputOnly[numChannels]; ///< Control which channels can not be set to output.
   
public:
   
   /// c'tor
   /** Sets m_channels and m_inputOnly to false.
     */
   xtChannels();
   
   /// Set a channel to be input only.
   /** A channel which is input only will not be sent as an output to the device
     * regardless of its state in m_channels.
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int setInputOnly( size_t chNo /**< [in] the channel to set to input only */ );
   
   /// Set a channel to true
   /**
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int setChannel( size_t chNo /**< [in] the channel to set */ );
   
   /// Set a channel to false
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int clearChannel( size_t chNo /**< [in] the channel to clear */ );
   
   /// Clear all channels
   /**
     * \returns 0 on success
     * \returns -1 on error
     */
   int clearAll();
   
   /// Set registers based on current channel states.
   /** Respects input only settings.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int setRegisters( uint16_t registers[numRegisters]  /**< [out] the array of registers to set */ );
   
   /// Read channel states from the registers
   /** This changes m_channels as they are read.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
   int readRegisters( uint16_t registers[numRegisters]  /**< [in] the array of registers to read */ );
   
   /// Gets the current state of a channel
   /**
     * \returns 0 if channel is not set 
     * \returns 1 if channel is set
     * \returns -1 on error
     */
   int channel( size_t chNo  /**< [in] the channel to retrieve */ );
   
};

/// Instantiaion of xtChannels for the 16-channel xt1121 
typedef xtChannels<16> xt1121Channels;

template<int numChannels>
xtChannels<numChannels>::xtChannels()
{
   for(size_t i =0; i< numChannels; ++i)
   {
      m_channels[i] = false;
      m_inputOnly[i] = false;
   }
}

template<int numChannels>
int xtChannels<numChannels>::setChannel( size_t chNo )
{
   if(chNo > numChannels - 1) return -1;
   
   m_channels[chNo] = true;
   
   return 0;
}

template<int numChannels>
int xtChannels<numChannels>::setInputOnly( size_t chNo )
{
   if(chNo > numChannels - 1) return -1;
   
   m_inputOnly[chNo] = true;
   
   return 0;
}

template<int numChannels>
int xtChannels<numChannels>::clearChannel( size_t chNo )
{
   if(chNo > numChannels - 1) return -1;
   
   m_channels[chNo] = false;
   
   return 0;
}

template<int numChannels>
int xtChannels<numChannels>::clearAll()
{
   for(size_t i=0;i<numChannels; ++i) m_channels[i] = false;
   
   return 0;
}

template<int numChannels>
int xtChannels<numChannels>::setRegisters( uint16_t registers[numRegisters] )
{
   for(size_t i =0; i < numRegisters; ++i)
   {
      registers[i] = 0;
      
      for(size_t j=0; j < 4; ++j)
      {
         if(!m_inputOnly[i*4 + j])
         {
            //Set the appropriate bit for this channel if it's true.
            if(m_channels[i*4+j])
            {
               registers[i] += (1 << j);
            }
         }
      }
   }
   
   
   return 0;
}

template<int numChannels>
int xtChannels<numChannels>::readRegisters( uint16_t registers[numRegisters] )
{
   for(size_t i =0; i < numRegisters; ++i)
   {      
      m_channels[i*4] = ((registers[i] & 1) > 0);
      m_channels[i*4+1] = ((registers[i] & 2) > 0) ;
      m_channels[i*4+2] = ((registers[i] & 4) > 0);
      m_channels[i*4+3] = ((registers[i] & 8) > 0);
   }
   
   
   return 0;
}

template<int numChannels>
int xtChannels<numChannels>::channel( size_t chNo )
{
   if(chNo > numChannels - 1) return -1;
   
   return m_channels[chNo];
   
}

#endif //xtChannels_hpp

