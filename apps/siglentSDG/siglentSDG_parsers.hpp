

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
int parseOUTP( int & channel,
               int & output,
               const std::string & strRead
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

/// Parse the SDG response to the BSWV query
/**
  * Example: C1:BSWV WVTP,SINE,FRQ,10HZ,PERI,0.1S,AMP,2V,AMPVRMS,0.707Vrms,OFST,0V,HLEV,1V,LLEV,-1V,PHSE,0
  *
  * \returns 0 on success
  * \returns \<0 on error, with value indicating location of error.
  */
int parseBSWV( int & channel,
               std::string & wvtp,
               double & freq,
               double & peri,
               double & amp,
               double & ampvrms,
               double & ofst,
               double & hlev,
               double & llev,
               double & phse,
               const std::string & strRead
             )
{
   std::vector<std::string> v;

   mx::ioutils::parseStringVector(v, strRead, ":, \n");

   if(v.size() < 4) return -1; //We need to get to at least the WVTP parameter.

   if(v[1] != "BSWV") return -2;

   if(v[0][0] != 'C') return -3;
   if(v[0].size() < 2) return -4;
   channel = mx::ioutils::convertFromString<int>(v[0].substr(1, v[0].size()-1));

   if(v[2] != "WVTP") return -5;
   wvtp = v[3];

   if(wvtp != "SINE") return -6; //We don't actually know how to handle anything else.
   
   if(v[4] != "FRQ") return -7;
   freq = mx::ioutils::convertFromString<double>(v[5]);

   if(v[6] != "PERI") return -8;
   peri = mx::ioutils::convertFromString<double>(v[7]);

   if(v[8] != "AMP") return -9;
   amp = mx::ioutils::convertFromString<double>(v[9]);

   if(v[10] != "AMPVRMS") return -10;
   ampvrms = mx::ioutils::convertFromString<double>(v[11]);

   if(v[12] != "OFST") return -11;
   ofst = mx::ioutils::convertFromString<double>(v[13]);

   if(v[14] != "HLEV") return -12;
   hlev = mx::ioutils::convertFromString<double>(v[15]);

   if(v[16] != "LLEV") return -13;
   llev = mx::ioutils::convertFromString<double>(v[17]);

   if(v[18] != "PHSE") return -14;
   phse = mx::ioutils::convertFromString<double>(v[19]);

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
int parseMDWV( int & channel,
               std::string & state,
               const std::string & strRead
             )
{
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
int parseSWWV( int & channel,
               std::string & state,
               const std::string & strRead
             )
{
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
int parseBTWV( int & channel,
               std::string & state,
               const std::string & strRead
             )
{
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
int parseARWV( int & channel,
               int & index,
               const std::string & strRead
             )
{
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
} //namespace app
} //namespace MagAOX

#endif //siglentSDG_hpp
