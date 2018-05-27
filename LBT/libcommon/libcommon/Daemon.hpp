/// Daemon.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef PCF_DAEMON_HPP
#define PCF_DAEMON_HPP

#include <string>
#include "ConfigFile.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class Daemon
{
    // Constructor/destructor
  public:
    Daemon();
    Daemon( const std::string &szProgName,
            const std::string &szVersion );
    virtual ~Daemon();

    // Methods
  public:
    /// The help message outputted to stderr.
    void displayHelp();
    std::string getConfigFileName() const
    {
      return m_szConfigFileName;
    }
    std::string getLogFileName() const
    {
      return m_szLogFileName;
    }
    std::string getProgName() const
    {
      return m_szProgName;
    }
    std::string getVersion() const
    {
      return m_szVersion;
    }
    bool initialize( int argc, char **argv, const bool &oFork = true );
    bool isQuitProcess() const
    {
      return sm_oQuitProcess;
    }
    static void processSignalHandler( int nSignal );

    // Variables
  private:
    /// This program's name.
    std::string m_szProgName;
    /// The current version of the program.
    std::string m_szVersion;
    /// The option string to use for this program.
    std::string m_szOptions;
    /// The default name of the log file.
    std::string m_szLogFileName;
    /// The default name of the config file.
    std::string m_szConfigFileName;
    /// The flag to tell this to quit.
    static bool sm_oQuitProcess;

}; // class Daemon
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // PCF_DAEMON_HPP
