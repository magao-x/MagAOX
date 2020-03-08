/** \file telem_telcat.hpp
  * \brief The MagAO-X logger telem_telcat log type.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup logger_types_files
  * 
  * History:
  * - 2018-09-06 created by JRM
  */
#ifndef logger_types_telem_telcat_hpp
#define logger_types_telem_telcat_hpp

#include "generated/telem_telcat_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{


/// Log entry recording the build-time git state.
/** \ingroup logger_types
  */
struct telem_telcat : public flatbuffer_log
{
   ///The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_TELCAT;

   ///The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

//    static constexpr int getCatObj = 0;
//    static constexpr int getCatRm = 1;
//    static constexpr int getCatRa = 2;
//    static constexpr int getCatDec = 3;
//    static constexpr int getCatEp = 4;
//    static constexpr int getCatRo = 5;
   
   ///The type of the input message
   struct messageT : public fbMessage
   {
      ///Construct from components
      messageT( const std::string & catObj,     ///< [in] 
                const std::string & catRm,    ///< [in] 
                const double & catRa,    ///< [in] 
                const double & catDec,        ///< [in] 
                const double & catEp,     ///< [in] 
                const double & catRo   ///< [in] 
              )
      {
         auto _catObj = builder.CreateString(catObj);
         auto _catRm = builder.CreateString(catRm);
         auto fp = CreateTelem_telcat_fb(builder, _catObj, _catRm, catRa, catDec, catEp, catRo);
         builder.Finish(fp);
      }

   };
                 
 
   ///Get the message formatte for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_telcat_fb(msgBuffer);

      std::string msg = "[telcat] ";
     
      if(fbs->catObj() != nullptr)
      {
         msg += "obj: ";
         msg += fbs->catObj()->c_str() ;
         msg += " ";
      }
      
      msg += "ra: ";
      msg += std::to_string(fbs->catRa()) + " ";
      
      msg += "dec: ";
      msg += std::to_string(fbs->catDec()) + " ";
      
      msg += "ep: ";
      msg += std::to_string(fbs->catEp()) + " ";
      
      if(fbs->catRm() != nullptr)
      {
         msg += "rm: ";
         msg += fbs->catRm()->c_str() ;
         msg += " ";
      }
      
      msg += "ro: ";
      msg += std::to_string(fbs->catRo()) + " ";
      
      
      
      return msg;
   
   }
   
   static std::string catObj(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      if(fbs->catObj() != nullptr)
      {
         return std::string(fbs->catObj()->c_str());
      }
      else
      {
         return std::string("");
      }
   }
   
   static std::string catRm(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      if(fbs->catRm() != nullptr)
      {
         return std::string(fbs->catRm()->c_str());
      }
      else
      {
         return std::string("");
      }
   }
   
   static double catRA(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catRa();
   }
   
   static double catDec(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catDec();
   }
   
   static double catEp(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catEp();
   }
   
   static double catRo(void * msgBuffer)
   {
      auto fbs = GetTelem_telcat_fb(msgBuffer);
      return fbs->catRo();
   }
   
   /// Get pointer to the accessor for a member by name 
   /**
     * \returns the function pointer cast to void*
     * \returns -1 for an unknown member
     */ 
   static void * getAccessor( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "catObj") return (void *) &catObj;
      else if(member == "catRm") return (void *) &catRm;
      else if(member == "catRA") return (void *) &catRA;
      else if(member == "catDec") return (void *) &catDec;
      else if(member == "catEp") return (void *) &catEp;
      else if(member == "catRo") return (void *) &catRo;
      else
      {
         std::cerr << "No string member " << member << " in telem_telcat\n";
         return 0;
      }
   }
   
#if 0   
   /// Get the lookup index for a member by name 
   /**
     * \returns the index for a valid member
     * \returns -1 for an unknown member
     */ 
   static int getIndex( const std::string & member /**< [in] the name of the member */ )
   {
      if(member == "catObj") return getCatObj;
      else if(member == "catRm") return getCatRm;
      else if(member == "catRA") return getCatRa;
      else if(member == "catDec") return getCatDec;
      else if(member == "catEp") return getCatEp;
      else if(member == "catRo") return getCatRo;
      else 
      {
         std::cerr << "No member " << member << " in telem_telcat\n";
         return -1;
      }
   }
   
   /// Get a string value for a member by name
   /**
     * \returns the value for a valid member name 
     * \returns an empty string for an unknown member name
     */  
   static std::string getString( const std::string & member, ///< [in] the member name
                                 void * msgBuffer            ///< [in] the message buffer to decode
                               )
   {
      return getString(getIndex(member), msgBuffer);
   }
   
   /// Get a string value for a member by index
   /**
     * \returns the value for a valid member index 
     * \returns an empty string for an unknown member index
     */
   static std::string getString( int index,       ///< [in] the member index
                                 void * msgBuffer ///< [in] the message buffer to decode
                               )
   {
      switch(index)
      {
         case getCatObj:
            return catObj(msgBuffer);
         case getCatRm:
            return catRm(msgBuffer);
         default:
            std::cerr << "Invalid index " << index << " for type std::string\n";
            return "";
      }
   }
   
   /// Get a double value for a member by name
   /**
     * \returns the value for a valid member name 
     * \returns -1e50 for an unknown member name
     */ 
   static double  getDouble( const std::string & member, ///< [in] the member name
                             void * msgBuffer            ///< [in] the message buffer to decode
                           )
   {
      return getDouble(getIndex(member), msgBuffer);
   }
   
   /// Get a double value for a member by index
   /**
     * \returns the value for a valid member name 
     * \returns -1e50 for an unknown member name
     */
   static double  getDouble( int index,       ///< [in] the member index
                             void * msgBuffer ///< [in] the message buffer to decode
                           )
   {
      switch(index)
      {
         case getCatRa:
            return catRA(msgBuffer);
         case getCatDec:
            return catDec(msgBuffer);
         case getCatEp:
            return catEp(msgBuffer);
         case getCatRo:
            return catRo(msgBuffer);
         default:
            std::cerr << "Invalid index " << index << " for type double\n";
            return -1e50;
      }
   }
   
#endif

}; //telem_telcat

timespec telem_telcat::lastRecord = {0,0};

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_telcat_hpp

