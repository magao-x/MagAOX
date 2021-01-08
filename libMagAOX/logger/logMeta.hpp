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

#include <mx/ioutils/fits/fitsHeaderCard.hpp>

#include "logMap.hpp"

namespace MagAOX
{
namespace logger
{

///\todo this needs to be auto-generated.
void * logMemberAccessor( flatlogs::eventCodeT ec,
                          const std::string & memberName
                        );

template<typename valT>
int getLogStateVal( valT & val,
                    logMap & lm,
                    const std::string & appName,
                    flatlogs::eventCodeT ev,
                    const flatlogs::timespecX & stime,
                    const flatlogs::timespecX & atime,
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
   valT stprV = getter(flatlogs::logHeader::messageBuffer(stprior));
   
   
   valT atprV;
   lm.getNextLog(atprior, stprior, appName);
   
   while( flatlogs::logHeader::timespec(atprior) < atime )
   {
      atprV = getter(flatlogs::logHeader::messageBuffer(atprior));
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
                    flatlogs::eventCodeT ev,
                    const flatlogs::timespecX & stime,
                    const flatlogs::timespecX & atime,
                    valT (*getter)(void *),
                    char ** hint = 0
                  )
{
   char * atafter;
   char * stprior;
   
   char * _hint;
   if(hint) _hint = *hint;
   else _hint = 0;

   flatlogs::timespecX midexp = meanTimespecX(atime, stime);
   
   //Get log entry before midexp
   if(lm.getPriorLog(stprior, appName, ev, midexp, _hint)!=0)
   {
      return 1;
   }
   valT stprV = getter(flatlogs::logHeader::messageBuffer(stprior));
   
   //Get log entry after.
   if(lm.getNextLog(atafter, stprior, appName)!=0)
   {
      return 1;
   }
   valT atprV = getter(flatlogs::logHeader::messageBuffer(atafter));

   double st = flatlogs::logHeader::timespec(stprior).asDouble();
   double it = midexp.asDouble();
   double et = flatlogs::logHeader::timespec(atafter).asDouble();
   
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
   
   flatlogs::eventCodeT m_logCode; //When this is set, set functions pointers 

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
            const flatlogs::eventCodeT logCode,
            const std::string & memberName,
            const std::string & format,
            const int metaType,
            const int valType 
          );
          
   std::string keyword();
   
   std::string comment();
   
   int setLog( flatlogs::eventCodeT ec,
               const std::string & mn
             );
   
   std::string value( logMap & lm,
                      const flatlogs::timespecX & stime,
                      const flatlogs::timespecX & atime
                    );
   
   std::string valueNumber( logMap & lm,
                            const flatlogs::timespecX & stime,
                            const flatlogs::timespecX & atime
                          );
   
   std::string valueString( logMap & lm,
                            const flatlogs::timespecX & stime,
                            const flatlogs::timespecX & atime
                          );
   
   mx::fits::fitsHeaderCard card( logMap &lm,
                                  const flatlogs::timespecX & stime,
                                  const flatlogs::timespecX & atime 
                                );
         
};


}
}







#endif //logger_logMega_hpp
