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
   
void * logMemberAccessor( flatlogs::eventCodeT ec,
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
      case telem_stage::eventCode:
         return telem_stage::getAccessor(memberName);
         
      default:
         return nullptr;
   }
}


logMeta::logMeta( const std::string & keyword,
                  const std::string & comment,
                  const std::string & appName,
                  const flatlogs::eventCodeT logCode,
                  const std::string & memberName,
                  const std::string & format,
                  const int metaType,
                  const int valType 
                ) : m_keyword {keyword}, m_comment {comment}, m_appName {appName}, m_format {format}, m_metaType {metaType}, m_valType {valType}
{
   setLog(logCode, memberName);   
}
       
std::string logMeta::keyword()
{
   return m_keyword;
}

std::string logMeta::comment()
{
   return m_comment;
}

int logMeta::setLog( flatlogs::eventCodeT ec,
                     const std::string & mn
                   )
{
   m_logCode = ec;
   m_memberName = mn;
   accessor = logMemberAccessor(ec, mn);
   
   return 0;
}


std::string logMeta::value( logMap & lm,
                            const flatlogs::timespecX & stime,
                            const flatlogs::timespecX & atime
                          )
{
   if(accessor == nullptr) return "";
         
   switch(m_valType)
   {
      case 0:
         return valueString( lm, stime, atime);
      case 1:
         return valueNumber( lm, stime, atime);
      default:
         return valueString( lm, stime, atime);
   }
}

std::string logMeta::valueNumber( logMap & lm,
                                  const flatlogs::timespecX & stime,
                                  const flatlogs::timespecX & atime
                                )
{
   double val;
         
   if(m_metaType == 0)
   {
      if( getLogStateVal(val,lm, m_appName,m_logCode,stime,atime,(double(*)(void*))accessor, &m_hint) != 0)
      {
         return m_invalidValue;
      }
   }
   else if(m_metaType == 1)
   {
      if( getLogContVal(val,lm, m_appName,m_logCode,stime,atime,(double(*)(void*))accessor, &m_hint) != 0)
      {
         return m_invalidValue;
      }
   }

   char str[64];
   snprintf(str, sizeof(str), m_format.c_str(), val);
   
   return std::string(str);
}

std::string logMeta::valueString( logMap & lm,
                                  const flatlogs::timespecX & stime,
                                  const flatlogs::timespecX & atime
                                )
{
   std::string val;
   if(m_metaType == 0)
   {
      if( getLogStateVal(val,lm, m_appName,m_logCode,stime,atime,(std::string(*)(void*))accessor, &m_hint) != 0)
      {
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
   if(m_valType == 0)
   {
      return mx::fits::fitsHeaderCard( m_keyword, value(lm, stime, atime), m_comment);
   }
   else 
   {
      //std::cerr <<  value(lm, stime, atime) << "\n";
      return mx::fits::fitsHeaderCard( m_keyword, value(lm, stime, atime).c_str(), m_comment);
   }
}

}
}



