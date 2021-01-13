/** \file logMeta.hpp
  * \brief Declares and defines the logMeta class and related classes.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_files
  * 
  * History:
  * - 2020-01-02 created by JRM
  */

#ifndef logger_logMeta_hpp
#define logger_logMega_hpp

#include "logMap.hpp"

///\todo this needs to be auto-generated.
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

template<typename valT>
int getLogStateVal( valT & val,
                    logMap & lm,
                    const std::string & appName,
                    eventCodeT ev,
                    const timespecX & stime,
                    const timespecX & atime,
                    valT (*getter)(void *),
                    char ** hint = 0
                  )
{
   char * atprior = nullptr;
   char * stprior = nullptr;
   
   char * _hint = nullptr;
   
   if(hint) _hint = *hint;
   else _hint = 0;

   lm.getPriorLog(stprior, appName, ev, stime, _hint);
   valT stprV = getter(logHeader::messageBuffer(stprior));
   
   
   valT atprV;
   lm.getNextLog(atprior, stprior, appName);
   
   while( logHeader::timespec(atprior) < atime )
   {
      atprV = getter(logHeader::messageBuffer(atprior));
      if(atprV != stprV)
      {
         val = atprV;
         if(hint) *hint = stprior;
         return 1;
      }
      stprior = atprior;
      lm.getNextLog(atprior, stprior, appName);
   }
   
   val = stprV;
   
   if(hint) *hint = stprior;
   return 0;
}

template<typename valT>
int getLogContVal( valT & val,
                    logMap & lm,
                    const std::string & appName,
                    eventCodeT ev,
                    const timespecX & stime,
                    const timespecX & atime,
                    valT (*getter)(void *),
                    char ** hint = 0
                  )
{
   char * atafter;
   char * stprior;
   
   char * _hint;
   if(hint) _hint = *hint;
   else _hint = 0;

   timespecX midexp = meanTimespecX(atime, stime);
   
   //Get log entry before midexp
   if(lm.getPriorLog(stprior, appName, ev, midexp, _hint)!=0)
   {
      return 1;
   }
   valT stprV = getter(logHeader::messageBuffer(stprior));
   
   //Get log entry after.
   if(lm.getNextLog(atafter, stprior, appName)!=0)
   {
      return 1;
   }
   valT atprV = getter(logHeader::messageBuffer(atafter));

   double st = logHeader::timespec(stprior).asDouble();
   double it = midexp.asDouble();
   double et = logHeader::timespec(atafter).asDouble();
   
   val = stprV + (atprV-stprV)/(et-st)*(it-st);
   
   if(hint) *hint = stprior;
   
   return 0;
}

//-- setLogCode [set the function pointers]
//-- setMemberName [get the index]
//--> above two should be same function.

/// Manage meta data for a log entry
/** Handles cases where log is a state, i.e. has one of a finite number of values, or is a
  * continuous variable, e.g. a temperature.
  * 
  * Contains the information to construct a FITS header card.
  */ 
struct logMeta
{
protected:
   std::string m_keyword;
   std::string m_comment;
   
   std::string m_appName;
   
   eventCodeT m_logCode; //When this is set, set functions pointers 

   std::string m_memberName; //When this is set, m_memberIndex gets set

   void * accessor {nullptr};
   
   std::string m_format;
   
   int m_metaType {0}; //0 for a state, 1 for a continuous variable to interpolate to midpoint
   int m_valType {0};
   bool m_isValid {false};
   std::string m_invalidValue {"invalid"};
   
   char * m_hint {nullptr};
   
public:
   
   logMeta( const std::string & keyword,
            const std::string & comment,
            const std::string & appName,
            const eventCodeT logCode,
            const std::string & memberName,
            const std::string & format,
            const int metaType,
            const int valType 
          );
          
   std::string keyword();
   
   std::string comment();
   
   int setLog( eventCodeT ec,
               const std::string & mn
             );
   
   std::string value( logMap & lm,
                      const timespecX & stime,
                      const timespecX & atime
                    );
   
   std::string valueNumber( logMap & lm,
                            const timespecX & stime,
                            const timespecX & atime
                          );
   
   std::string valueString( logMap & lm,
                            const timespecX & stime,
                            const timespecX & atime
                          );
   
   mx::improc::fitsHeaderCard card( logMap &lm,
                                    const timespecX & stime,
                                    const timespecX & atime 
                                  );
         
};


logMeta::logMeta( const std::string & keyword,
                  const std::string & comment,
                  const std::string & appName,
                  const eventCodeT logCode,
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

int logMeta::setLog( eventCodeT ec,
                     const std::string & mn
                   )
{
   m_logCode = ec;
   m_memberName = mn;
   accessor = logMemberAccessor(ec, mn);
   
   return 0;
}


std::string logMeta::value( logMap & lm,
                            const timespecX & stime,
                            const timespecX & atime
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
                                  const timespecX & stime,
                                  const timespecX & atime
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
                                  const timespecX & stime,
                                  const timespecX & atime
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

mx::improc::fitsHeaderCard logMeta::card( logMap &lm,
                                          const timespecX & stime,
                                          const timespecX & atime 
                                        )
{
   if(m_valType == 0)
   {
      return mx::improc::fitsHeaderCard( m_keyword, value(lm, stime, atime), m_comment);
   }
   else 
   {
      //std::cerr <<  value(lm, stime, atime) << "\n";
      return mx::improc::fitsHeaderCard( m_keyword, value(lm, stime, atime).c_str(), m_comment);
   }
}








#endif //logger_logMega_hpp
