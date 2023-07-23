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
#define logger_logMeta_hpp

#include <mx/ioutils/fits/fitsHeaderCard.hpp>
//#define HARD_EXIT
#include "logMap.hpp"

namespace MagAOX
{
namespace logger
{

//This is how the user specifies an item of log meta data (i.e. via a config file)
struct logMetaSpec
{
   std::string device;
   flatlogs::eventCodeT eventCode;
   std::string member;
   std::string keyword; //overrides the default
   std::string format; //overrides the default
   std::string comment; //overrides the default
   

   logMetaSpec()
   {
   }

   logMetaSpec( const std::string & dev,
                const flatlogs::eventCodeT ec,
                const std::string & memb,
                const std::string & k,
                const std::string & f,
                const std::string & c
              ) : device(dev), eventCode(ec), member(memb), keyword(k), format(f), comment(c)
   {
   }

   logMetaSpec( const std::string & dev,
                const flatlogs::eventCodeT ec,
                const std::string & memb
              ) : device(dev), eventCode(ec), member(memb)
   {
   }

};

//This is the data returned by the member accessor.
struct logMetaDetail
{
   std::string keyword;
   std::string comment;
   std::string format;
   int valType {-1};
   int metaType {-1};
   void * accessor {nullptr};
   bool hierarch {true}; // if false the device name is not included.

   logMetaDetail()
   {
   }

   logMetaDetail( const std::string & k,
                  const std::string & c,
                  const std::string & f,
                  int vt,
                  int mt,
                  void *acc
                ) : keyword(k), comment(c), format(f), valType(vt), metaType(mt), accessor(acc)
   {
   }

   logMetaDetail( const std::string & k,
                  const std::string & c,
                  const std::string & f,
                  int vt,
                  int mt,
                  void *acc,
                  bool h
                ) : keyword(k), comment(c), format(f), valType(vt), metaType(mt), accessor(acc), hierarch(h)
   {
   }

   logMetaDetail( const std::string & k,
                  const std::string & c,
                  int vt,
                  int mt,
                  void *acc,
                  bool h
                ) : keyword(k), comment(c), valType(vt), metaType(mt), accessor(acc), hierarch(h)
   {
   }

   logMetaDetail( const std::string & k,
                  int vt,
                  int mt,
                  void *acc
                ) : keyword(k), valType(vt), metaType(mt), accessor(acc)
   {
   }

   logMetaDetail( const std::string & k,
                  int vt,
                  int mt,
                  void *acc, 
                  bool h
                ) : keyword(k), valType(vt), metaType(mt), accessor(acc), hierarch(h)
   {
   }

};  

logMetaDetail logMemberAccessor( flatlogs::eventCodeT ec,
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

   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   if(lm.getPriorLog(stprior, appName, ev, stime, _hint) != 0) 
   {
      std::cerr << __FILE__ << " " << __LINE__ << " getPriorLog returned error for " << appName << ":" << ev << "\n";
      return -1;
   }
   valT stprV = getter(flatlogs::logHeader::messageBuffer(stprior));
   
   valT atprV;
   
   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   if(lm.getNextLog(atprior, stprior, appName) != 0) 
   {
      std::cerr << __FILE__ << " " << __LINE__ << " getNextLog returned error for " << appName << ":" << ev << "\n";
      return -1;
   }

   #ifdef DEBUG
   std::cerr << __FILE__ << " " << __LINE__ << "\n";
   #endif

   while( flatlogs::logHeader::timespec(atprior) < atime )
   {
      atprV = getter(flatlogs::logHeader::messageBuffer(atprior));
      if(atprV != stprV)
      {
         val = atprV;
         if(hint) *hint = stprior;
         return 0;
      }
      stprior = atprior;
      if(lm.getNextLog(atprior, stprior, appName) != 0)
      {
         std::cerr << __FILE__ << " " << __LINE__ << " getNextLog returned error for " << appName << ":" << ev << "\n";
         return -1;
      }
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
      std::cerr << __FILE__ << " " << __LINE__ << " getPriorLog returned error for " << appName << ":" << ev << "\n";
      return 1;
   }
   valT stprV = getter(flatlogs::logHeader::messageBuffer(stprior));
   
   //Get log entry after.
   if(lm.getNextLog(atafter, stprior, appName)!=0)
   {
      std::cerr << __FILE__ << " " << __LINE__ << " getNextLog returned error for " << appName << ":" << ev << "\n";
      #ifdef HARD_EXIT 
      exit(-1);
      #endif
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


/// Manage meta data for a log entry
/** Handles cases where log is a state, i.e. has one of a finite number of values, or is a
  * continuous variable, e.g. a temperature.
  * 
  * Contains the information to construct a FITS header card.
  */ 
struct logMeta
{
public:
   enum valTypes
   {
      String = mx::fits::fitsType<std::string>(),
      Bool = mx::fits::fitsType<bool>(),
      Char = mx::fits::fitsType<char>(),
      UChar = mx::fits::fitsType<unsigned char>(),
      Short = mx::fits::fitsType<short>(),
      UShort = mx::fits::fitsType<unsigned short>(),
      Int = mx::fits::fitsType<int>(),
      UInt = mx::fits::fitsType<unsigned int>(),
      Long = mx::fits::fitsType<long>(),
      ULong = mx::fits::fitsType<unsigned long>(),
      LongLong = mx::fits::fitsType<long long>(),
      ULongLong = mx::fits::fitsType<unsigned long long>(),
      Float = mx::fits::fitsType<float>(),
      Double = mx::fits::fitsType<double>(),
      Vector_String = 10000,
      Vector_Bool = 10002,
      Vector_Char = 10004,
      Vector_UChar = 10006,
      Vector_Short = 10008,
      Vector_UShort = 10010,
      Vector_Int = 10012,
      Vector_UInt = 10014,
      Vector_Long = 10016,
      Vector_ULong = 10018,
      Vector_LongLOng = 10020,
      Vector_ULongLong = 10022,
      Vector_Float = 10024,
      Vector_Double  = 10026
   };

   enum metaTypes
   {
      State,
      Continuous
   };

protected:
   
   logMetaSpec m_spec;
   logMetaDetail m_detail;

   bool m_isValid {false};
   std::string m_invalidValue {"invalid"};
   
   char * m_hint {nullptr};
   
public:
   
   logMeta( const logMetaSpec & lms /**< [in] the specification of this meta data entry */ );
          
   std::string keyword();
   
   std::string comment();
   
   int setLog( const logMetaSpec &);
   
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







#endif //logger_logMeta_hpp
