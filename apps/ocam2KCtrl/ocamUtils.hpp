/** \file ocamUtils.hpp
  * \brief Utilities for the OCAM camera
  *
  * \ingroup ocam2kCtrl_files
  */

#ifndef ocamUtils_hpp
#define ocamUtils_hpp


#include <mx/ioutils/stringUtils.hpp>

namespace MagAOX
{
namespace app
{

struct ocamTemps
{
   float CCD {0};
   float CPU {0};
   float POWER {0};
   float BIAS {0};
   float WATER {0};
   float LEFT {0};
   float RIGHT {0};
   float SET {0};
   float COOLING_POWER {0};
};


int parseTemps( ocamTemps & temps,
                const std::string & tstr
              )
{
   std::vector<std::string> v;
   mx::ioutils::parseStringVector(v, tstr, "[]");

   if( v.size() < 18) return -1;
   
   temps.CCD = mx::ioutils::convertFromString<float>( v[1] );
   temps.CPU = mx::ioutils::convertFromString<float>( v[3] );
   temps.POWER = mx::ioutils::convertFromString<float>( v[5] );
   temps.BIAS = mx::ioutils::convertFromString<float>( v[7] );
   temps.WATER = mx::ioutils::convertFromString<float>( v[9] );
   temps.LEFT = mx::ioutils::convertFromString<float>( v[11] );
   temps.RIGHT = mx::ioutils::convertFromString<float>( v[13] );
   temps.SET = mx::ioutils::convertFromString<float>( v[15] )/10.0;
   temps.COOLING_POWER = mx::ioutils::convertFromString<float>( v[17] );
   
   return 0;
}

int parseFPS( float & fps,
              const std::string & fstr
            )
{
   std::vector<std::string> v;
   mx::ioutils::parseStringVector(v, fstr, "[]");

   if( v.size() < 3) return -1;

   fps = mx::ioutils::convertFromString<float>( v[1] );

   return 0;
}

} //namespace app
} //namespace MagAOX

#endif //ocamUtils_hpp
