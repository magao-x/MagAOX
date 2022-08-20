/** \file rhusbMonParsers.hpp
  * \brief Parsers for the MagAO-X RH USB monitor
  *
  * \ingroup rhusbMon_files
  */

#ifndef rhusbMonParsers_hpp
#define rhusbMonParsers_hpp

#include <string>

namespace MagAOX
{
namespace app
{
namespace RH
{
   
/// Parse the RH probe C temp command
/**
  * \returns -1 if the end of transmission string is not found
  * \returns -2 if there is no value in the string
  * \returns -3 if the parsed string does not begin with a digit 
  * \returns 0 on success
  */ 
int parseC( float & temp,           ///< [out] the reported temperature
            const std::string & str ///< [in] the string returned by the device
          )
{
   size_t st = str.find(" C\r\n>");
   if(st == std::string::npos)
   {
      temp = -999;
      return -1;
   }
   
   if(st == 0)
   {
      temp = -999;
      return -2;
   }

   if(!isdigit(str[0]) && str[0] != '-')
   {
      temp = -999;
      return -3;
   }

   temp = std::stof( str.substr(0, st) );
   
   return 0;
}

/// Parse the RH probe H humidity command
/**
  * \returns -1 if the end of transmission string is not found
  * \returns -2 if there is not value in the string 
  * \returns -3 if the parsed string does not begin with a digit
  * \returns 0 on success
  */
int parseH( float & humid,          ///< [out] the reported temperature
            const std::string & str ///< [in] the string returned by the device
          )
{
   size_t st = str.find(" %RH\r\n>");
   if(st == std::string::npos)
   {
      humid = -999;
      return -1;
   }
   
   if(st == 0)
   {
      humid = -999;
      return -2;
   }

   if(!isdigit(str[0]))
   {
      humid = -999;
      return -3;
   }

   humid = std::stof( str.substr(0, st) );
   
   return 0;
}

} //namespace RH
} //namespace app
} //namespace MagAOX

#endif //rhusbMonParsers_hpp
