

#ifndef siglentSDG_parsers_hpp
#define siglentSDG_parsers_hpp


#include <mx/ioutils/stringUtils.hpp>

namespace MagAOX
{
namespace app
{

/// Parse the SDG response to the OUTP query
   /**
     * Example: C1:OUTP OFF,LOAD,HZ,PLRT,NOR
     *
     * \returns 0 on success
     * \returns \<0 on error, with value indicating location of error.
     */
int parseOUTP( int & channel, ///< [out] the channel indicated by this response.
               int & output, ///< [out] the output status of the channel, ON or OFF
               const std::string & strRead ///< [in] string containing the device response
             )
{
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");

   channel = -1;
   output = -1;
   if(v.size() < 2) return -1;
   
   if(v[1] != "OUTP") return -2;

   if(v[0][0] != 'C') return -3;
   if(v[0].size() < 2) return -4;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v.size() < 3) return -5;
   
   if(v[2] == "OFF") output = 0;
   else if(v[2] == "ON") output = 1;
   else return -6;
   return 0;
}

#define SDG_PARSEERR_WVTP (-6)

/// Parse the SDG response to the BSWV query
/**
  * Example: C1:BSWV WVTP,SINE,FRQ,10HZ,PERI,0.1S,AMP,2V,AMPVRMS,0.707Vrms,OFST,0V,HLEV,1V,LLEV,-1V,PHSE,0
  *
  * \todo document tests
  * \todo update tests for new wdth parameter in PULSE
  * 
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseBSWV( int & channel, ///< [out] the channel indicated by this response.
               std::string & wvtp,
               double & freq,
               double & peri,
               double & amp,
               double & ampvrms,
               double & ofst,
               double & hlev,
               double & llev,
               double & phse,
               double & wdth,
               const std::string & strRead ///< [in] string containing the device response
             )
{
   channel = 0;
   freq = 0;
   peri = 0;
   amp = 0;
   ampvrms = 0;
   ofst = 0;
   hlev = 0;
   llev = 0;
   phse = 0;
   wdth = 0;
   
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");
   //std::cout << strRead << "\n";

   if(v.size() < 4) return -1; //We need to get to at least the WVTP parameter.

   if(v[1] != "BSWV") return -2;

   if(v[0][0] != 'C') return -3;
   if(v[0].size() < 2) return -4;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "WVTP") return -5;
   wvtp = v[3];

   if(wvtp != "SINE" && wvtp != "PULSE" && wvtp != "DC") return SDG_PARSEERR_WVTP; //We don't actually know how to handle anything else.
   
   if(wvtp == "DC")
   {
      if(v.size() < 6) return -7;
      
      if(v[4] != "OFST") return -8;
      
      ofst = mx::ioutils::convertFromString<double>(v[5]);
      
      return 0;
   }
   
   if(v.size() < 20) return -9;
   
   if(v[4] != "FRQ") return -10;
   freq = mx::ioutils::convertFromString<double>(v[5]);

   if(v[6] != "PERI") return -11;
   peri = mx::ioutils::convertFromString<double>(v[7]);

   if(v[8] != "AMP") return -12;
   amp = mx::ioutils::convertFromString<double>(v[9]);

   if(v[10] != "AMPVRMS") return -13;
   ampvrms = mx::ioutils::convertFromString<double>(v[11]);

   if(v[12] != "OFST") return -14;
   ofst = mx::ioutils::convertFromString<double>(v[13]);

   if(v[14] != "HLEV") return -15;
   hlev = mx::ioutils::convertFromString<double>(v[15]);

   if(v[16] != "LLEV") return -16;
   llev = mx::ioutils::convertFromString<double>(v[17]);

   if(wvtp == "SINE")
   {
      if(v[18] != "PHSE") return -17;
      phse = mx::ioutils::convertFromString<double>(v[19]);
   }

   if(wvtp == "PULSE")
   {
      if(v[20] != "WIDTH") return -18;
      wdth = mx::ioutils::convertFromString<double>(v[21]);
   }

   return 0;
}

/// Parse the SDG response to the MDWV query
/** Currently we are only looking for STATE,ON or STATE,OFF.  If ON,
  * we ignore the rest of the values.
  * 
  * Example: C1:MDWV STATE,OFF
  *
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseMDWV( int & channel, ///< [out] the channel indicated by this response.
               std::string & state, ///< [out] the MDWV state of the channel, ON or OFF
               const std::string & strRead ///< [in] string containing the device response
             )
{
   channel = 0;

   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");


   if(v.size() < 4) return -1;

   if(v[1] != "MDWV") return -2;

   if(v[0][0] != 'C') return -3;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "STATE") return -4;
   state = v[3];
   
   return 0;
}

/// Parse the SDG response to the SWWV query
/** Currently we are only looking for STATE,ON or STATE,OFF.  If ON,
  * we ignore the rest of the values.
  * 
  * Example: C1:SWWV STATE,OFF
  *
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseSWWV( int & channel, ///< [out] the channel indicated by this response.
               std::string & state, ///< [out] the SWWV state of the channel, ON or OFF
               const std::string & strRead ///< [in] string containing the device response
             )
{
   channel = 0;
   
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");


   if(v.size() < 4) return -1;

   if(v[1] != "SWWV") return -2;

   if(v[0][0] != 'C') return -3;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "STATE") return -4;
   state = v[3];
   
   return 0;
}

/// Parse the SDG response to the BTWV query
/** Currently we are only looking for STATE,ON or STATE,OFF.  If ON,
  * we ignore the rest of the values.
  * 
  * Example: C1:BTWV STATE,OFF
  *
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseBTWV( int & channel, ///< [out] the channel indicated by this response.
               std::string & state, ///< [out] the BTWV state of the channel, ON or OFF
               const std::string & strRead ///< [in] string containing the device response
             )
{
   channel = 0;
   
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");


   if(v.size() < 4) return -1;

   if(v[1] != "BTWV") return -2;

   if(v[0][0] != 'C') return -3;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "STATE") return -4;
   state = v[3];
   
   return 0;
}

/// Parse the SDG response to the ARWV query
/** Currently we are only looking for INDEX,0 
  * 
  * Example: C1:ARWV INDEX,0,NAME,
  *
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseARWV( int & channel, ///< [out] the channel indicated by this response.
               int & index, ///< [out] the ARWV index of the channel.  Should be 0.
               const std::string & strRead ///< [in] string containing the device response
             )
{
   channel = 0;
   index = -1;
   
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");


   if(v.size() < 4) return -1;

   if(v[1] != "ARWV") return -2;

   if(v[0][0] != 'C') return -3;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "INDEX") return -4;
   index = mx::ioutils::convertFromString<int>(v[3]);
   
   return 0;
}

/// Parse the SDG response to the SYNC query
/** 
  * 
  * Example: C1:SYNC ON
  *
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseSYNC( int & channel, ///< [out] the channel indicated by this response.
               bool & sync, ///< [out] the ARWV index of the channel.  Should be 0.
               const std::string & strRead ///< [in] string containing the device response
             )
{
   channel = 0;
   
   sync = false;
   
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");


   if(v.size() < 3) return -1;

   if(v[1] != "SYNC") return -2;

   if(v[0][0] != 'C') return -3;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] == "ON")
   {
      sync = true;
      return 0;   
   }
   else if(v[2] == "OFF")
   {
      sync = false;
      return 0;
   }
   else return -4;
   
   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //siglentSDG_hpp
