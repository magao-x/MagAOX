/// Event.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_EVENT_HPP
#define PCF_EVENT_HPP

#include <string>
#include <iostream>
#include <sstream>
#include <syslog.h>
#include "TimeStamp.hpp"

#ifndef LOG_LOCAL0
#define LOG_LOCAL0 0
#define LOG_LOCAL1 1
#define LOG_LOCAL2 2
#define LOG_LOCAL3 3
#define LOG_LOCAL4 4
#define LOG_LOCAL5 5
#define LOG_LOCAL6 6
#define LOG_LOCAL7 7
#endif
#ifndef LOG_ERR
#define LOG_ERR 3
#define LOG_WARNING 4
#define LOG_NOTICE 5
#define LOG_INFO 6
#define LOG_DEBUG 7
#endif

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class Event
{
    // Constants.
  public:
    enum Error
    {
      enumNoError = 0,
    };

    // Used for the severity of the event.
    enum Severity
    {
      enumError = 1,
      enumWarning = 2,
      enumNotice = 3,
      enumInfo = 4,
      enumDebug = 5,
      // The default when the event is created.
      enumDefaultSeverity = enumInfo
    };

    // This enum represents the facilty type. These can be repurposed to be
    // whatever we want.
    enum Facility
    {
      enumNoFacility = 0, // Zero is a special case for the facility.
      enumLocal0 = LOG_LOCAL0,
      enumLocal1 = LOG_LOCAL1,
      enumLocal2 = LOG_LOCAL2,
      enumLocal3 = LOG_LOCAL3,
      enumLocal4 = LOG_LOCAL4,
      enumLocal5 = LOG_LOCAL5,
      enumLocal6 = LOG_LOCAL6,
      enumLocal7 = LOG_LOCAL7,
      // The current default for events.
      enumDefaultFacility = enumLocal4
    };

    // Constructor/Destructor/Copy constructor.
  public:
    Event();
    Event( const Severity &tSeverity );
    Event( const Severity &tSeverity, const Facility &tFacility );
    Event( const Event &copy );
    virtual ~Event();

    // Operators.
  public:
    const Event  &operator=( const Event &rhs );
    Event &operator<<( const Facility &tFacility );
    Event &operator<<( const Severity &tSeverity );
    Event &operator<<( const pcf::TimeStamp &ts );
    /// This function will handle "endl" among other things....
    Event &operator<<( std::ostream & ( *fcn )( std::ostream & ) );

    // Global static methods.
  public:
    /// Should we add a prefix to the event showing the severity?
    static void addSeverityPrefix( const bool &oAdd )
    {
      sm_oAddSeverityPrefix = oAdd;
    }
    /// Should we add a prefix to the event showing the milliseconds?
    static void addMillisPrefix( const bool &oAdd )
    {
      sm_oAddMillisPrefix = oAdd;
    }
    /// Should we duplicate the event to stderr (default = false)?
    static void enableStdErr( const bool &oEnable )
    {
      sm_oEnableStdErr = oEnable;
    }
    /// Set the prefix & facility & severity & open the log.
    static void init( const std::string &szPrefix,
                      const Facility &tFacility = enumDefaultFacility,
                      const Severity &tSeverity = enumDefaultSeverity );
    /// Should we enable a process-level severity label?
    static void enableError( const bool &oEnable )
    {
      sm_oEnableError = oEnable;
    }
    static void enableWarning( const bool &oEnable )
    {
      sm_oEnableWarning = oEnable;
    }
    static void enableNotice( const bool &oEnable )
    {
      sm_oEnableNotice = oEnable;
    }
    static void enableInfo( const bool &oEnable )
    {
      sm_oEnableInfo = oEnable;
    }
    static void enableDebug( const bool &oEnable )
    {
      sm_oEnableDebug = oEnable;
    }
    static void setEmailList( const std::string &szEmailList )
    {
      sm_szEmailList = szEmailList;
    }

    /// Is a certain process-level severity label enabled?
    static bool isErrorEnabled()
    {
      return sm_oEnableError;
    }
    static bool isWarningEnabled()
    {
      return sm_oEnableWarning;
    }
    static bool isNoticeEnabled()
    {
      return sm_oEnableNotice;
    }
    static bool isInfoEnabled()
    {
      return sm_oEnableInfo;
    }
    static bool isDebugEnabled()
    {
      return sm_oEnableDebug;
    }

    // Methods.
  public:
    /// Clear out any and all data in this object,
    /// and reset the severity back to enumInfo.
    virtual void clear();
    /// What facility is associated with this object?
    virtual Facility getFacility() const
    {
      return m_tFacility;
    }
    /// Get the log string assigned in this object.
    virtual std::string getLogString() const
    {
      return m_ssMsg.str();
    }
    /// Get the name for this object.
    virtual std::string getName() const
    {
      return m_szName;
    }
    /// Get the time stamp assigned in this object.
    virtual TimeStamp getTimeStamp() const
    {
      return m_tsTimeStamp;
    }
    /// What severity is associated with this object?
    virtual Severity getSeverity() const
    {
      return m_tSeverity;
    }
    /// Is the severity enabled for this message?
    bool isSeverityEnabled();
    /// Generic function that wraps the actual syslog call.
    /// This version does not actually clear the stringstream.
    virtual int log();
    /// Generic function that wraps the actual syslog call.
    /// This version WILL clear the stringstream.
    virtual int logThenClear();
    /// This sends an email notification that there is a problem.
    virtual bool sendGenericEmailAlarm( const std::string &szSubject,
                                        const std::string &szMessage );
    /// This sends an email notification that there is a problem.
    virtual bool sendTempSensorEmailAlarm( const std::string &szSensorName,
                                           const std::string &szChannelName,
                                           const float &eValue,
                                           const float &eMin,
                                           const float &eMax );
    /// This facility will be added to all events sent to the syslog.
    virtual void setFacility( const Facility &tFacility )
    {
      m_tFacility = tFacility;
    }
    /// This sets the message that will be sent to the log.
    virtual void setLogString( const std::string &szLogString )
    {
      m_ssMsg.str( szLogString );
    }
    /// This sets the name for this object.
    virtual void setName( const std::string &szName )
    {
      m_szName = szName;
    }
    /// This severity will be added to all events sent to the syslog.
    virtual void setSeverity( const Severity &tSeverity )
    {
      m_tSeverity = tSeverity;
    }
    /// What is the severity of this object?
    virtual bool isError() const
    {
      return ( m_tSeverity == enumError );
    }
    virtual bool isWarning() const
    {
      return ( m_tSeverity == enumWarning );
    }
    virtual bool isNotice() const
    {
      return ( m_tSeverity == enumNotice );
    }
    virtual bool isInfo() const
    {
      return ( m_tSeverity == enumInfo );
    }
    virtual bool isDebug() const
    {
      return ( m_tSeverity == enumDebug );
    }
    // This sets a date and time to be associated with the message.
    virtual void setTimeStamp( const TimeStamp &tsTimeStamp )
    {
      m_tsTimeStamp = tsTimeStamp;
    }

    // Templated functions.
  public:
    template<class TT> Event &operator<<( const TT &tData )
    {
      m_ssMsg << tData;
      return *this;
    }

    // Variables.
  private:
    /// The output built up over several stream calls.
    std::stringstream m_ssMsg;
    /// The name for this object.
    std::string m_szName;
    /// A date and time associated with the event.
    TimeStamp m_tsTimeStamp;
    /// The local severity of the event.
    Severity m_tSeverity;
    /// The local facility of the event.
    Facility m_tFacility;
    /// This is the comma-delimited list of email recipiants who will
    /// receive an email if an alarm is triggered.
    static std::string sm_szEmailList;
    /// Should we add a prefix to the message showing the severity?
    static bool sm_oAddSeverityPrefix;
    /// Should we add a prefix to the message showing the milliseconds?
    static bool sm_oAddMillisPrefix;
    /// Are the messages being send to stderr as well?
    static bool sm_oEnableStdErr;
    /// Which of the severities are enabled (default = all)?
    static bool sm_oEnableError;
    static bool sm_oEnableWarning;
    static bool sm_oEnableNotice;
    static bool sm_oEnableInfo;
    static bool sm_oEnableDebug;
    /// This is our prefix - do not set this directly - use init().
    /// we need a static because this variable must not "go away".
    static std::string sm_szPrefix;
    /// This is our severity - do not set this directly - use init().
    /// we need a static because this variable must not "go away".
    static Severity sm_tSeverity;
    /// This is our facility - do not set this directly - use init().
    /// we need a static because this variable must not "go away".
    static Facility sm_tFacility;

}; // Class Event
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_EVENT_HPP
