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
      #ifdef HARD_EXIT 
      if(vs == m_invalidValue)
      {
         std::cerr << __FILE__ << " " << __LINE__ << " valueString returned invalid value\n";
         exit(-1);
      } 
      #endif
      return vs;
   }
   else
   {
      std::string vn = valueNumber( lm, stime, atime);
      #ifdef HARD_EXIT 
      if(vn == m_invalidValue)
      {
         std::cerr << __FILE__ << " " << __LINE__ << " valueNumber returned invalid value\n";
      } 
      #endif
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
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(bool(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Char:
         {
            char val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(char(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UChar:
         {
            unsigned char val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned char(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Short:
         {
            short val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(short(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UShort:
         {
            unsigned short val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned short(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Int:
         {
            int val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(int(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UInt:
         {
            unsigned int val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned int(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Long:
         {
            long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULong:
         {
            unsigned long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::LongLong:
         {
            long long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(long long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULongLong:
         {
            unsigned long long val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned long long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Float:
         {
            float val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(float(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Double:
         {
            double val;
            if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(double(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
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
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(bool(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Char:
         {
            char val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(char(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UChar:
         {
            unsigned char val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned char(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Short:
         {
            short val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(short(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UShort:
         {
            unsigned short val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned short(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Int:
         {
            int val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(int(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::UInt:
         {
            unsigned int val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned int(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Long:
         {
            long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULong:
         {
            unsigned long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::LongLong:
         {
            long long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(long long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::ULongLong:
         {
            unsigned long long val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(unsigned long long(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Float:
         {
            float val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(float(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
            snprintf(str, sizeof(str), m_spec.format.c_str(), val);
            return std::string(str);
         }
         case valTypes::Double:
         {
            double val;
            if( getLogContVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(double(*)(void*))m_detail.accessor, &m_hint) != 0) return m_invalidValue;
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
      if( getLogStateVal(val,lm, m_spec.device,m_spec.eventCode,stime,atime,(std::string(*)(void*))m_detail.accessor, &m_hint) != 0)
      {
         #ifdef HARD_EXIT 
         std::cerr << __FILE__ << " " << __LINE__ << "\n";
         //exit(-1);
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
   if(m_detail.valType == valTypes::String)
   {
      if(m_detail.hierarch == false)
      {
         return mx::fits::fitsHeaderCard( m_spec.keyword, value(lm, stime, atime), m_spec.comment);
      }
      else
      { 
         //Add spaces to make sure hierarch is invoked
         std::string keyw = m_spec.device + " " + m_spec.keyword;
         if(keyw.size() < 9) 
         {
            keyw += std::string(9-keyw.size(), ' ');
         }
         return mx::fits::fitsHeaderCard( keyw, value(lm, stime, atime), m_spec.comment);
      }
   }
   else 
   {
      if(m_detail.hierarch == false)
      {
         return mx::fits::fitsHeaderCard( m_spec.keyword, value(lm, stime, atime).c_str(),  m_detail.valType, m_spec.comment);
      }
      else
      {
         //Add spaces to make sure hierarch is invoked
         std::string keyw = m_spec.device + " " + m_spec.keyword;
         if(keyw.size() < 9) 
         {
            keyw += std::string(9-keyw.size(), ' ');
         }
         return mx::fits::fitsHeaderCard( keyw, value(lm, stime, atime).c_str(),  m_detail.valType, m_spec.comment);
      }
   }
}

}
}



