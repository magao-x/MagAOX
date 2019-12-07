/** \file logInterp.hpp
  * \brief Interpolation Of Log Entries.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  * History:
  * - 2019-12-07 created by JRM
  */

#ifndef logger_logInterp_hpp
#define logger_logInterp_hpp



#include <flatlogs/flatlogs.hpp>

namespace MagAOX
{
namespace logger
{

template<typename floatT>
int interpLog( floatT & val,
               timespec & tm,
               const floatT & val0,
               flatlogs::bufferPtrT & buffer0,
               const floatT & val1,
               flatlogs::bufferPtrT & buffer1
             )
{
   timespecX tm0 = logHeader::timespec(buffer0);
   timespecX tm1 = logHeader::timespec(buffer1);
      
   double dtm = tm.tv_sec + ((double) tm.tv_nsec)/1e9;
   double dtm0 = tm0.time_s + ((double) tm0.time_ns)/1e9;
   double dtm1 = tm1.time_s + ((double) tm1.time_ns)/1e9;
              
   val = val0 + (val1 - val0)/(dtm1-dtm0)*(dtm-dtm0);
    
   return 0;
}

}

}

#endif //logger_logInterp_hpp
