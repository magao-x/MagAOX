/** \file logMeta.cpp
  * \brief Declares and defines the logMeta class and related classes.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  */


#include "logMeta.hpp"

#include "generated/logTypes.hpp"


namespace MagAOX
{
namespace logger
{
   
logMetaDetail logMemberAccessor( flatlogs::eventCodeT ec,
                          const std::string & memberName
                        )
{
   switch(ec)
   {
      case telem_stdcam::eventCode:
         return telem_stdcam::getAccessor(memberName);
      case telem_telcat::eventCode:
         return telem_telcat::getAccessor(memberName);
      case telem_teldata::eventCode:
         return telem_teldata::getAccessor(memberName);
      case telem_telpos::eventCode:
         return telem_telpos::getAccessor(memberName);
      case telem_stage::eventCode:
         return telem_stage::getAccessor(memberName);
      case telem_zaber::eventCode:
         return telem_zaber::getAccessor(memberName);   
      case telem_dmspeck::eventCode:
         return telem_dmspeck::getAccessor(memberName);
      case telem_observer::eventCode:
         return telem_observer::getAccessor(memberName);
      case telem_fxngen::eventCode:
         return telem_fxngen::getAccessor(memberName);
      case telem_loopgain::eventCode:
         return telem_loopgain::getAccessor(memberName);
      default:
         std::cerr << "Missing logMemberAccessor case entry for " << ec << ":" << memberName << "\n";
         return logMetaDetail();
   }
}


logMeta::logMeta( const logMetaSpec & lms )
{
   setLog(lms);   
}
       
std::string logMeta::keyword()
{
   return m_spec.keyword;
}

std::string logMeta::comment()
{
   return m_spec.comment;
}

int logMeta::setLog( const logMetaSpec & lms )
{
   m_spec = lms;
   m_detail = logMemberAccessor(m_spec.eventCode, m_spec.member);
   
   if(m_spec.keyword == "") m_spec.keyword = m_detail.keyword;
   if(m_spec.format == "") m_spec.format = m_detail.format;
   if(m_spec.format == "")
   {
      switch(m_detail.valType)
      {
         case valTypes::String:
            m_spec.format = "%s";
            break;
         case valTypes::Bool:
            m_spec.format = "%d";
            break;
         case valTypes::Char:
            m_spec.format = "%d";
            break;
         case valTypes::UChar:
            m_spec.format = "%u";
            break;
         case valTypes::Short:
            m_spec.format = "%d";
            break;
         case valTypes::UShort:
            m_spec.format = "%u";
            break;
         case valTypes::Int:
            m_spec.format = "%d";
            break;
         case valTypes::UInt:
            m_spec.format = "%u";
            break;
         case valTypes::Long:
            m_spec.format = "%ld";
            break;
         case valTypes::ULong:
            m_spec.format = "%lu";
            break;
         case valTypes::Float:
            m_spec.format = "%g";
            break;
         case valTypes::Double:
            m_spec.format = "%g";
            break;
         case valTypes::Vector_Bool:
            m_spec.format = "%d";
            break;
         case valTypes::Vector_Float:
            m_spec.format = "%g";
            break;
         default:
            std::cerr << "Unrecognised value type for " + m_spec.device + " " + m_spec.keyword + ".  Using format %d/\n";
            m_spec.format = "%d";

      }

   }


   if(m_spec.comment == "") m_spec.comment = m_detail.comment;

   return 0;
}


std::string logMeta::value( logMap & lm,
                            const flatlogs::timespecX & stime,
                            const flatlogs::timespecX & atime
                          )
{
   if(m_detail.accessor == nullptr) return "";
         
   if(m_detail.valType == valTypes::String)
   {
      std::string vs = valueString( lm, stime, atime); 
      if(vs == m_invalidValue)
      {
         std::cerr << __FILE__ << " " << __LINE__ << " valueString returned invalid value\n";
      } 
      return vs;
   }
   else
   {
      std::string vn = valueNumber( lm, stime, atime);
      if(vn == m_invalidValue)
      {
         std::cerr << __FILE__ << " " << __LINE__ << " valueNumber returned invalid value\n";
      } 
      return vn;
   }
}

std::string logMeta::valueNumber( logMap & lm,
                                  const flatlogs::timespecX & stime,
                                  const flatlogs::timespecX & atime
                                )
{
   char str[64];

   if(m_detail.metaType == metaTypes::State)
   {
      switch(m_detail.valType)
      {
         case valTypes::Bool:
         {
            bool val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<bool(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Char:
         {
            char val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<char(*)(void*)>(m_detail.accessor), &m_hint) != 0) 
            {
               std::cerr << "getLogStateVal returned error: " << __FILE__ << " " << __LINE__ << "\n";
               return m_invalidValue;
            }
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UChar:
         {
            unsigned char val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned char(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Short:
         {
            short val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<short(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UShort:
         {
            unsigned short val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned short(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Int:
         {
            int val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<int(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UInt:
         {
            unsigned int val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned int(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Long:
         {
            long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULong:
         {
            unsigned long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::LongLong:
         {
            long long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<long long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULongLong:
         {
            unsigned long long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned long long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Float:
         {
            float val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<float(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Double:
         {
            double val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<double(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Vector_Bool:
         {
            std::vector<bool> val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<std::vector<bool>(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;

            if(val.size() == 0) return "";

            std::string res;

            for(size_t n = 0; n < val.size()-1; ++n)
            {
               snprintf(str, sizeof(str), m_spec.format.c_str(), (int) val[n]);
               res += str;
               res += ',';
            }

            snprintf(str, sizeof(str), m_spec.format.c_str(), (int) val.back());
            res += str;
            
            return res;
         }
         case valTypes::Vector_Float:
         {
            std::vector<float> val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<std::vector<float>(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;

            if(val.size() == 0) return "";

            std::string res;

            for(size_t n = 0; n < val.size()-1; ++n)
            {
               snprintf(str, sizeof(str), m_spec.format.c_str(), val[n]);
               res += str;
               res += ',';
            }

            snprintf(str, sizeof(str), m_spec.format.c_str(), val.back());
            res += str;
            
            return res;
         }
         default:
            return m_invalidValue;
      }
   }
   else if(m_detail.metaType == metaTypes::Continuous)
   {
      switch(m_detail.valType)
      {
         case valTypes::Bool:
         {
            bool val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<bool(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Char:
         {
            char val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<char(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UChar:
         {
            unsigned char val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned char(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Short:
         {
            short val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<short(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UShort:
         {
            unsigned short val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned short(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Int:
         {
            int val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<int(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UInt:
         {
            unsigned int val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned int(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Long:
         {
            long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULong:
         {
            unsigned long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::LongLong:
         {
            long long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<long long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULongLong:
         {
            unsigned long long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<unsigned long long(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Float:
         {
            float val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<float(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Double:
         {
            double val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<double(*)(void*)>(m_detail.accessor), &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         default:
            return m_invalidValue;
      }
   }

   return m_invalidValue;
   
}

std::string logMeta::valueString( logMap & lm,
                                  const flatlogs::timespecX & stime,
                                  const flatlogs::timespecX & atime
                                )
{
   std::string val;
   if(m_detail.metaType == metaTypes::State)
   {
      if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,reinterpret_cast<std::string(*)(void*)>(m_detail.accessor), &m_hint) != 0)
      {
         std::cerr << "getLogStateVal returned error " << __FILE__ << " " << __LINE__ << "\n";

         #ifdef HARD_EXIT 
         std::cerr << __FILE__ << " " << __LINE__ << "\n";
         
         exit(-1);
         #endif
         val = m_invalidValue;
      }
   }
   else
   {
      std::cerr << "String type specified as something other than state\n";
   }
   return val;
}

mx::fits::fitsHeaderCard logMeta::card( logMap &lm,
                                          const flatlogs::timespecX & stime,
                                          const flatlogs::timespecX & atime 
                                        )
{
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif
   
   std::string vstr = value(lm, stime, atime);
   
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   std::string keyw;
   if(m_detail.hierarch)
   {
      //Add spaces to make sure hierarch is invoked
      keyw = m_spec.device + " " + m_spec.keyword;
      if(keyw.size() < 9)
      {
         keyw += std::string(9-keyw.size(), ' ');
      }
   }
   else
   {
      keyw = m_spec.keyword;
   }

   if(vstr == m_invalidValue)
   {
      std::cerr << "got invalid value: " << __FILE__ << " " << __LINE__ << "\n";
      // always a string sentinel value, so return here to skip the valType conditional
      return mx::fits::fitsHeaderCard(keyw, vstr, m_spec.comment);
   }

   if(m_detail.valType == valTypes::String)
   {
      return mx::fits::fitsHeaderCard(keyw, vstr, m_spec.comment);
   }
   else 
   {
      return mx::fits::fitsHeaderCard(keyw, vstr.c_str(), m_detail.valType, m_spec.comment);
   }
}

} // logger
} // MagAOX



