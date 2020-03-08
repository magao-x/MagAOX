/** \file logDefs.hpp
  * \brief Type definitions for the flatlogs format.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogs_files
  * 
  * History:
  * - 2018-08-17 created by JRM
  */
#ifndef flatlogs_logDefs_hpp
#define flatlogs_logDefs_hpp

#include <cstdint>

namespace flatlogs 
{

/// The type of the log priority code.
/** \ingroup logDefs
  */
typedef int8_t logPrioT;
   
/// The type used for seconds.  
/** Rationale: unsigned 32 bits gives us enough to last 168 yrs from the UNIX epoch, which is more than enough. 
  * 24 bits is just 1/2 year, so we would have to use a partial byte to be more optimum.
  * 
  * \ingroup logDefs
  */
typedef uint32_t secT;  
   
/// The type used for nanoseconds.  
/** Rationale: unsigned 32 bits gives >4x10^9 nanoseconds, so enough for 1x10^9 nanoseconds.
  */
typedef uint32_t nanosecT; 
   
/// The type of an event code (16-bit unsigned int).
/** Rationale: gives us 65,536 individual events.
  * \ingroup logDefs
  */
typedef uint16_t eventCodeT;

/// The type used for the short message length
/** Rationale: most flatlog entries are short, so we use minimum space for this.
  * 
  * \ingroup logDefs
  */
typedef uint8_t  msgLen0T; 

/// The type used for intermediate message length
/** Rationale: using 1+2 =3 bytes for a 256 byte message is 1.2%. 1+4 = 2.0%, and 1+8 = 3.5%.   
  * 
  * \ingroup logDefs
  */
typedef uint16_t msgLen1T; 

/// The type used for long message length
/** Rationale: once messages are 65536 or longer, the length field is negligible.  This 
  * admits messages of huge sizes.
  * 
  * \ingroup logDefs
  */
typedef uint64_t msgLen2T; 

/// The type used to refer to the message length, regardless of length.
/** This is not necessarily what is written to the buffer.  Should always be msgLen2T so it is big enough
  * to handle any possible length.  
  * \ingroup logDefs
  */
typedef msgLen2T msgLenT; 

}//namespace flatlogs

#endif //flatlogs_logDefs_hpp

