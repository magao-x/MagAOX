/// LogFile.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_LOG_FILE_HPP
#define PCF_LOG_FILE_HPP

#include <string>
#include <fstream>
#include "MutexLock.hpp"
#include "TimeStamp.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class LogFile
{
  // Constructor/destructor/copy constructor.
  public:
    LogFile();
    virtual ~LogFile();

  // Methods
  public:
    /// Close the underlying file. todo: DEPRECATED.
    void close();
    // Create a properly constructed name for the file.
    //static std::string createLogfileName( const std::string &szPath,
    //                                      const std::string &szFilename,
    //                                      const pcf::TimeStamp &tsName = TimeStamp::now() );
    /// Should we add a severity prefix to each log message? This is only
    /// applied if one of the severity log methods is called.
    void enableSeverityPrefix( const bool &oEnable );
    /// Should we duplicate the messages to stderr?
    void enableStdErr( const bool &oEnable );
    /// Should we add a time stamp to each log message? The messages routed
    /// to stderr do not get a timestamp regardless of this setting.
    void enableTimeStamp( const bool &oEnable );
    /// Get the name for this object.
    std::string getName() const;
    /// Set the needed info.
    void init( const std::string &szPath,
               const std::string &szBaseFilename );
    /// Are we enabling the severity prefix in the log message?
    bool isSeverityPrefixEnabled();
    /// Are we enabling the replication of the message to stderr?
    bool isStdErrEnabled();
    /// Are we enabling the time stamp in the log message?
    bool isTimeStampEnabled();
    /// Function that writes to the log with no severity information.
    virtual void log( const std::string &szLogString );
    /// Set the needed info & open the file. Deprecated.
    void open( const std::string &szPath,
               const std::string &szBaseFilename,
               const std::string &szSymLinkName = "" );
    /// This sets the name for this object.
    void setName( const std::string &szName );

    /// Is the debug Severity enabled?
    bool isDebugEnabled();
    /// Is the err Severity enabled?
    bool isErrorEnabled();
    /// Is the info Severity enabled?
    bool isInfoEnabled();
    /// Is the notice Severity enabled?
    bool isNoticeEnabled();
    /// Is the warning Severity enabled?
    bool isWarningEnabled();

    /// Should we enable debug messages?
    void enableDebug( const bool &oEnable );
    /// Should we enable error messages?
    void enableError( const bool &oEnable );
    /// Should we enable info messages?
    void enableInfo( const bool &oEnable );
    /// Should we enable notice messages?
    void enableNotice( const bool &oEnable );
    /// Should we enable warning messages?
    void enableWarning( const bool &oEnable );

    /// Function that writes to the log with a severity of 'debug'.
    virtual void logDebug( const std::string &szLogString );
    /// Function that writes to the log with a severity of 'error'.
    virtual void logError( const std::string &szLogString );
    /// Function that writes to the log with a severity of 'info'.
    virtual void logInfo( const std::string &szLogString );
    /// Function that writes to the log with a severity of 'notice'.
    virtual void logNotice( const std::string &szLogString );
    /// Function that writes to the log with a severity of 'warning'.
    virtual void logWarning( const std::string &szLogString );

  // Helper functions.
  private:
    //void makePath( const std::string &szPath ) const;
    //void makeSymLink( const std::string &szFilename,
    //                  const std::string &szLinkname ) const;
    void writeToFile( const std::string &szLogString,
                      const std::string &szSeverityPrefix );

  // Variables.
  private:
    /// The name for this object.
    std::string m_szName;
    /// Should we add a time stamp to each log message? The messages routed
    /// to stderr do not get a timestamp regardless of this setting.
    bool m_oEnableStdErr;
    /// Is a time stamp being added to each log message?
    bool m_oEnableTimeStamp;
    /// Should we add a severity prefix to each log message? This is only
    /// applied if one of the severity log methods is called.
    bool m_oEnableSeverityPrefix;
    /// This is our path & base filename to log to.
    /// Do not set this directly - use init().
    std::string m_szPath;
    std::string m_szBaseFilename;
    /// This should prevent collisions of multiple threads writing to the file
    /// at the same time.
    mutable pcf::MutexLock m_mutFile;
    /// These are variables which control whether or not messages
    /// actually are sent to the log.
    bool m_oEnableDebug;
    bool m_oEnableError;
    bool m_oEnableInfo;
    bool m_oEnableNotice;
    bool m_oEnableWarning;
    /// This hold whether a valid filename has been set ('init' has been called).
    bool m_oIsInit;

}; // Class LogFile
} // Namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_LOG_FILE_HPP
