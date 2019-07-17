/** \file zaberUtils.hpp
  * \brief utilties for working with zaber stages
  *
  * \ingroup zaberLowLevel_files
  */

#ifndef zaberUtils_hpp
#define zaberUtils_hpp

#include <iostream>


#define ZUTILS_E_NOAT       (-100)
#define ZUTILS_E_NOSP       (-101)
#define ZUTILS_E_BADADD     (-102)
#define ZUTILS_E_SERIALSIZE (-103)

namespace MagAOX
{
namespace app
{

/// Parse the system.serial query
/**
  * \returns 0 on success
  * \returns \<0 on error with error code primarily meant for unit testing
  * 
  * \ingroup zaberLowLevel 
  */  
int parseSystemSerial( std::vector<int> & address,
                       std::vector<std::string> & serial,
                       const std::string & response
                     )
{
   size_t at = response.find('@', 0);

   if(at == std::string::npos)
   {
      return ZUTILS_E_NOAT;
   }
   
   while(at != std::string::npos)
   {
      size_t sp = response.find(' ', at);
      
      if(sp == std::string::npos)
      {
         return ZUTILS_E_NOSP;
      }
      
      if(sp-at  != 3 ) //Address should be 2 characters
      {
         return ZUTILS_E_BADADD;
      }
      
      int add = std::stoi( response.substr(at+1, sp-at-1));
      
      address.push_back(add);
      
      at = response.find('@', at+1);
      
      sp = response.rfind(' ', at);
      size_t ed = response.find_first_of("\n@", sp);
      if(ed == std::string::npos) ed = response.size();
      
      if(ed-sp-1 != 5)
      {
         return ZUTILS_E_SERIALSIZE;
      }
      
      std::string ser = response.substr(sp+1, ed - sp-1);
      
      serial.push_back(ser);
   }
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //zaberUtils_hpp
