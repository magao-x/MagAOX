/// Logger.hpp
///
///  @author Paul Grenz
///
/// This object supports and interface which writes log messages to a file
/// on disk. The actual file is a static object in this
/// class, meaning that only one instance will be created
/// on a per-process basis. In this way, any number of these "Logger" objects
/// can be created, but only one file will be written to for the whole process.
///
/// The logging mechanism supports the idea of a "stream" and a severity
/// can be attached to the message. The severity can be streamed or set using
/// the "setSeverity" call. Each severity can be turned on or off, effectively
/// routing the message to nowhere. The "clear" method can be used to clear out
/// any existing text stored in the class.
///
/// Using the class is easy. Instantiate an object stream to it, then send an
/// "endl" to cause the message to be written to the file. Each write consists
/// of one line.
///
/// Example:
///
///    Logger logMsg;
///    ...
///    logMsg.clear();
///    logMsg << Logger::enumError << "An error occurred." << endl;
///    ...
///
/// You may override this behavior by calling a specific method, such as
/// "logIndi", "logAlert", "logEvent" or "logSetting". These calls replace the
/// internally maintained data and only uses the data passed in with the call.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_LOGGER_HPP
#define PCF_LOGGER_HPP

#include <string>
#include <sstream>
#include <iostream>    // To use 'cerr'.
#include "TimeStamp.hpp"
#include "LogFile.hpp"
#include "IndiProperty.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class Logger
{
  // Constants.
  public:
    // Used for the severity of the message.
    enum Severity
    {
      Error = 1,
      Warning = 2,
      Notice = 3,
      Info = 4,
      Debug = 5
    };

    // Used for the severity of the message.
    // This is for backwards compatibility with old code. In new code,
    // use the Severity enumerations, above.
    static const Severity enumError;
    static const Severity enumWarning;
    static const Severity enumNotice;
    static const Severity enumInfo;
    static const Severity enumDebug;

  // Constructor/destructor/copy constructor.
  public:
    Logger();
    Logger( const Logger::Severity &tSeverity );
    Logger( const Logger &copy );
    virtual ~Logger();

  // Operators.
  public:
    const Logger &operator=( const Logger &copy );
    Logger &operator<<( const Severity &tSeverity );
    /// A function to stream any kind of data into the internal stringstream.
    template<class TT> Logger &operator<<( const TT &tData );
    /// This function will handle "endl" among other things....
    Logger &operator<<( std::ostream & ( *fcn )( std::ostream & ) );

  // Global static methods.
  public:
    /// Should we add a prefix showing the Severity when this is logged?
    static void enableSeverityPrefix( const bool &oEnable );
    /// Should we send a duplicate to stderr when this is logged?
    static void enableStdErr( const bool &oEnable );
    /// Should we add a prefix showing the time when this is logged?
    static void enableTimeStamp( const bool &oEnable );

    /// Should we enable error messages?
    static void enableError( const bool &oEnable );
    /// Should we enable warning messages?
    static void enableWarning( const bool &oEnable );
    /// Should we enable notice messages?
    static void enableNotice( const bool &oEnable );
    /// Should we enable info messages?
    static void enableInfo( const bool &oEnable );
    /// Should we enable debug messages?
    static void enableDebug( const bool &oEnable );

    /// Set the path & base filename only & open the log.
    static void init( const std::string &szPath,
                      const std::string &szBaseFilename );

    /// Are we adding a prefix showing the Severity when this is logged?
    static bool isSeverityPrefixEnabled();
    /// Are we sending a duplicate to stderr when this is logged?
    static bool isStdErrEnabled();
    /// Are we adding a prefix showing the time when this is logged?
    static bool isTimeStampEnabled();

    /// Is the err Severity enabled?
    static bool isErrorEnabled();
    /// Is the warning Severity enabled?
    static bool isWarningEnabled();
    /// Is the notice Severity enabled?
    static bool isNoticeEnabled();
    /// Is the info Severity enabled?
    static bool isInfoEnabled();
    /// Is the debug Severity enabled?
    static bool isDebugEnabled();

  // Methods.
  public:
    /// Clear out any and all data in this object,
    /// and reset the Severity back the Logger::enumInfo.
    virtual void clear();
    /// Should we enable the clearing of messages after they have been logged?
    virtual void enableClearAfterLog( const bool &oEnable );
    /// What Severity is associated with this object?
    virtual Severity getSeverity() const;
    /// Get the message stored in this object.
    virtual std::string getText() const;
    /// Will the message be cleared after logging?
    virtual bool isClearAfterLogEnabled() const;
    /// Check the Severity a different way.
    virtual bool isDebug() const;
    virtual bool isError() const;
    virtual bool isInfo() const;
    virtual bool isNotice() const;
    virtual bool isWarning() const;
    /// Returns the current length of the string contained here.
    virtual unsigned int length() const;
    /// Generic function that wraps the actual system call.
    /// This version does not actually clear the stringstream.
    virtual void log();
    /// Logs an ALARM to the INDI log using a set format (ISO 8061). The timestamp
    /// is optional and defaults to "now".
    /// Also sends an email notification that there is a problem. The assigned
    /// (comma-separated) recipiants are used.
    void logAlarm( const Logger::Severity &tSeverity,
                   const std::string &szRecipients,
                   const std::string &szSubject,
                   const std::string &szMessage,
                   const pcf::TimeStamp &tsNow = pcf::TimeStamp::now() );
    /// Logs an ALERT to the INDI log using a set format (ISO 8061). The timestamp
    /// is optional and defaults to "now".
    /// ALERT 2013-09-23T11:41:40.959453Z Any string you want
    void logAlert( const Logger::Severity &tSeverity,
                   const std::string &szMessage,
                   const pcf::TimeStamp &tsNow = TimeStamp::now() );
    /// Logs an EVENT to the INDI log using a set format (ISO 8061). The timestamp
    /// is optional and defaults to "now".
    /// EVENT 2013-09-23T11:41:40.959453Z Name Message
    void logEvent( const Logger::Severity &tSeverity,
                   const std::string &szName,
                   const std::string &szMessage,
                   const pcf::TimeStamp &tsNow = TimeStamp::now() );
    /// Logs a SETTING to the INDI log using a set format. The timestamp
    /// is optional and defaults to "now".
    /// SETTING 2013-09-23T11:41:40.959453Z Name = Value
    template<class TT> void logSetting( const Logger::Severity &tSeverity,
                            const std::string &szName,
                            const TT &tValue,
                            const pcf::TimeStamp &tsNow = pcf::TimeStamp::now() );
    /// Logs an INDI property to stderr, which ends up in the INDI log.
    virtual void logIndi( const Logger::Severity &tSeverity,
                          const pcf::IndiProperty &propLog );
    /// Generic function that wraps the actual log call.
    /// This version WILL clear the stringstream.
    virtual void logThenClear();
    /// This prefix will be prepended to all messages to the log.
    virtual void setSeverity( const Severity &tSeverity );
    /// This sets the message stored here.
    virtual void setText( const std::string &szText );

  // Variables.
  private:
    /// The output built up over several stream calls.
    std::stringstream m_ssMsg;
    /// The Severity of the message.
    Severity m_tSeverity;
    /// Should the output be cleared after it is logged?
    bool m_oClearAfterLog;
    /// This is the underlying log file object this
    /// class uses to actually write to the file.
    static pcf::LogFile sm_lfLog;

}; // Class Logger

////////////////////////////////////////////////////////////////////////////////
/// A function to stream any kind of data into the internal stringstream.

template<class TT> Logger &Logger::operator<<( const TT &tData )
{
  m_ssMsg << tData;
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Logs a SETTING to the INDI log using a set format. The timestamp
/// is optional and defaults to "now".
/// SETTING 2013-09-23T11:41:40.959453Z Name = Value

template<class TT> void Logger::logSetting( const Logger::Severity &tSeverity,
                                            const std::string &szName,
                                            const TT &tValue,
                                            const pcf::TimeStamp &tsNow )
{
  m_ssMsg.str( "" );
  setSeverity( tSeverity );
  m_ssMsg << "SETTING " << tsNow.getFormattedIso8601Str()
          << " " << szName << " = " << tValue;
  std::cerr << m_ssMsg.str() << std::endl;
  log();
}

////////////////////////////////////////////////////////////////////////////////

} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
// Make a derived class to handle additional future features.
// todo: Eventually this will be broken out into a separate file.
class Message : public Logger
{
}; // Class Message
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_LOGGER_HPP
