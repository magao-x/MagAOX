/** \file sysMonitor.hpp
 * \brief Publish and telemeter host diagnostics, GPU measurements, IRQ placement, and RAID health.
 *
 * To view logdump files: logdump -f sysMonitor
 *
 * To view sysMonitor with cursesIndi:
 * 1. /opt/MagAOX/bin/xindiserver -n xindiserverMaths
 * 2. /opt/MagAOX/bin/sysMonitor -n sysMonitor
 * 3. /opt/MagAOX/bin/cursesINDI
 *
 * \ingroup sysMonitor_files
 *
 * History:
 * - 2018-08-10 created by CJB
 */
#ifndef sysMonitor_hpp
#define sysMonitor_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "../../libMagAOX/sys/boundedCommand.hpp"
#include "gpuMonitor.hpp"
#include "irqMonitor.hpp"
#include "raidMonitor.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <sys/wait.h>
#include <chrono>
#include <map>
#include <set>
#include <cctype>
#include <mx/math/floatUtils.hpp>

namespace MagAOX
{
namespace app
{

/** MagAO-X application to read and report system statistics
 *
 */
class sysMonitor : public MagAOXApp<>, public dev::telemeter<sysMonitor>
{

    friend class dev::telemeter<sysMonitor>;

    /// Telemeter interface used by the framework lifecycle macros.
    using telemeterT = dev::telemeter<sysMonitor>;

    /// Select the host CPU-temperature output format.
    enum class sysType
    {
        Intel,
        AMD
    };

  protected:
    /** \name System Statistics - Data
     * @{ */

    /// CPU family selecting the supported sensors output parser.
    sysType m_sysType{ sysType::Intel };

    /// CPU warning threshold in Celsius; zero requests discovery from Intel sensors output.
    int m_warningCoreTemp = 0;

    /// CPU critical threshold in Celsius; zero requests discovery from Intel sensors output.
    int m_criticalCoreTemp = 0;

    /// Drive warning threshold in Celsius; zero derives it from the first measured temperature.
    int m_warningDiskTemp = 0;

    /// Drive critical threshold in Celsius; zero derives it from the first measured temperature.
    int m_criticalDiskTemp = 0;

    /// INDI summary of minimum, maximum, and mean CPU load fractions.
    pcf::IndiProperty m_indiP_core_loads;

    /// INDI summary of minimum, maximum, and mean CPU temperatures in Celsius.
    pcf::IndiProperty m_indiP_core_temps;

    /// INDI temperatures indexed by the drive names returned by hddtemp.
    pcf::IndiProperty m_indiP_drive_temps;

    /// INDI disk and RAM utilization fractions.
    pcf::IndiProperty m_indiP_usage;

    /// CPU temperatures from the current sensors sample, in Celsius.
    std::vector<float> m_coreTemps;

    /// CPU activity fractions from the current mpstat sample.
    std::vector<float> m_coreLoads;

    /// Configured drive names retained from the diskNames setting.
    std::vector<std::string> m_diskNameList;

    /// Drive identities associated with the current temperature sample.
    std::vector<std::string> m_diskNames;

    /// Current drive temperatures in Celsius, ordered like m_diskNames.
    std::vector<float> m_diskTemps;

    /// Most recently parsed utilization fraction for the root filesystem.
    float m_rootUsage = 0;

    /// Most recently parsed utilization fraction for the /data filesystem.
    float m_dataUsage = 0;

    /// Most recently parsed utilization fraction for the /boot filesystem.
    float m_bootUsage = 0;

    /// Most recently parsed RAM utilization fraction.
    float m_ramUsage = 0;

    ///@}

    /// Publish the current CPU, drive, RAM, Chrony, and latency-control values through INDI.
    /** \returns 0 on completion. */
    int updateVals();

    /** \name System Diagnostics - Data
     * @{ */

    /// Whether the optional NVIDIA collector is enabled.
    bool m_gpuEnabled{ false };

    /// Minimum time between GPU query starts, in seconds.
    double m_gpuInterval{ 5 };

    /// Maximum userspace wait for a GPU query, in seconds.
    double m_gpuTimeout{ 2 };

    /// GPU UUIDs required even before successful discovery.
    std::vector<std::string> m_gpuExpected;

    /// Owns the bounded GPU query process until it has been reaped.
    sys::boundedCommand m_gpuCommand;

    /// Current GPU samples, including unavailable previously known devices.
    std::vector<telem_gpu::gpuT> m_gpus;

    /// Last recorded samples for value-change suppression.
    std::vector<telem_gpu::gpuT> m_lastGPUs;

    /// Overall GPU collection status, also published when discovery returns no devices.
    std::string m_gpuStatus{ "disabled" };

    /// Last recorded overall GPU status.
    std::string m_lastGPUStatus;

    /// Whether any GPU telemetry record has been made by this instance.
    bool m_gpuRecorded{ false };

    /// Logical dedicated-device affinity rules, resolved anew each poll.
    std::vector<sysMonitorDetail::ruleT> m_irqRules;

    /// Minimum time between IRQ checks, in seconds.
    double m_irqInterval{ 5 };

    /// Current placement results for every configured logical IRQ group.
    std::vector<telem_irq::groupT> m_irqs;

    /// Last recorded IRQ policy flags, in the startup-logged rule order.
    std::vector<uint8_t> m_lastIRQOK;

    /// Whether any IRQ telemetry record has been made by this instance.
    bool m_irqRecorded{ false };

    /// Monotonic start times of the most recent GPU and IRQ polls.
    std::chrono::steady_clock::time_point m_lastGPUPoll{}, m_lastIRQPoll{};

    /// Track first polls independently of the monotonic clock's epoch.
    bool m_gpuPolled{ false }, m_irqPolled{ false };

    /// Properties keyed by NVIDIA index; map nodes keep registered addresses valid.
    std::map<std::string, pcf::IndiProperty> m_gpuProperties, m_gpuInfoProperties;

    /// Overall GPU state and per-rule IRQ state exposed to operators.
    pcf::IndiProperty m_indiP_gpuStatus, m_indiP_irqStatus, m_indiP_irqWarning;

    /// Enable automatic discovery of Linux MD arrays.
    bool m_raidEnabled{ true };

    /// Minimum time between MD status reads, in seconds.
    double m_raidInterval{ 5 };

    /// Last read start and whether any RAID read has been attempted.
    std::chrono::steady_clock::time_point m_lastRAIDPoll{};

    bool m_raidPolled{ false }; ///< Whether the first RAID read has occurred.

    /// Current arrays, including unavailable previously discovered names, in stable index order.
    std::vector<sysMonitorDetail::raidArrayT> m_raids;

    /// Last recorded compact array values for change suppression.
    std::vector<telem_raid::arrayT> m_lastRAIDs;

    /// Current and last recorded collector states.
    uint8_t m_raidStatus{ telem_raid::Unchecked }, m_lastRAIDStatus{ telem_raid::Unchecked };

    std::string m_raidMessage{ "unchecked" }; ///< Human-readable aggregate collection result.

    double m_raidCheckTime{ 0 }; ///< Actual time of the latest read attempt, retained on forced records.

    bool m_raidRecorded{ false }; ///< Whether this instance has emitted RAID telemetry.

    /// Stable storage for dynamically discovered array measurements and identification.
    std::map<std::string, pcf::IndiProperty> m_raidProperties, m_raidInfoProperties;

    /// Overall collection status and operator-warning switch.
    pcf::IndiProperty m_indiP_raidStatus, m_indiP_raidWarning;

    ///@}

    /** \name System Diagnostics
     * @{ */

    /// Load and validate optional diagnostic settings and named IRQ sections.
    int loadDiagnosticsConfig( mx::app::appConfigurator &cfg /**< [in,out] configuration source */ );

    /// Initialize only diagnostic properties, without starting collectors or threads.
    int startupDiagnostics();

    /// Collect due diagnostics using the supplied monotonic scheduling time.
    int pollDiagnostics( std::chrono::steady_clock::time_point now /**< [in] monotonic poll time */ );

    /// Query the NVIDIA utility; virtual so offline tests can inject command results.
    virtual sys::boundedCommand::result queryGPUs();

    /// Check current IRQ files; virtual so offline tests can supply fixtures.
    virtual std::vector<telem_irq::groupT> queryIRQs( double sampleTime /**< [in] wall-clock check time */ );

    /// Merge a query result while preserving identity and invalidating missing devices.
    void acceptGPUResult( const sys::boundedCommand::result &result /**< [in] command outcome */,
                          double                             sampleTime /**< [in] wall-clock measurement time */ );

    /// Publish the current diagnostic state, registering new GPU identities as needed.
    int publishDiagnostics();

    /// Record first, changed, or explicitly forced GPU state.
    int recordGPUs( bool force = false /**< [in] record even when values are unchanged */ );

    /// Record first, changed, or explicitly forced IRQ OK flags and the actual check time.
    int recordIRQs( bool force = false /**< [in] record even when values are unchanged */ );

    /// Register aggregate RAID diagnostics before the first discovery read.
    int startupRAID();

    /// Read MD status; virtual so tests can inject snapshots without accessing hardware.
    virtual sysMonitorDetail::raidSnapshotT queryRAID( double checkTime /**< [in] actual read time */ );

    /// Merge discovery results while retaining unavailable names and stable telemetry indices.
    void acceptRAIDResult( const sysMonitorDetail::raidSnapshotT &result /**< [in] current read outcome */ );

    /// Report whether collection failed or any observed array needs operator attention.
    bool raidWarning() const;

    /// Publish current array measurements, identities, and aggregate status.
    int publishRAID();

    /// Record first, changed, or forced compact RAID state with its actual check time.
    int recordRAID( bool force = false /**< [in] record even when values are unchanged */ );

    ///@}

  public:
    /// Construct the app with its build identity and default diagnostic settings.
    sysMonitor();

    /// Destroy app-owned diagnostic state without propagating exceptions.
    ~sysMonitor() noexcept
    {
    }

    /// Register host thresholds, GPU/IRQ/RAID diagnostics, and telemeter configuration.
    virtual void setupConfig();

    /// Load and validate host settings, diagnostic policy, and telemetry configuration.
    virtual void loadConfig();

    /// Register diagnostic INDI properties and start telemetry and the latency-control thread.
    virtual int appStartup();

    /// Collect due diagnostics, record telemetry, and publish current values.
    virtual int appLogic();

    /// Stop command collection, join the latency-control thread, and shut down telemetry.
    virtual int appShutdown();

    /** \name System Statistics
     * @{ */

    /// Collect CPU temperatures from sensors output.
    /** Makes system call and then parses result to add temperatures to vector of values
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on successful completion otherwise
     */
    int findCPUTemperatures( std::vector<float> &temps /**< [out] measured CPU temperatures in Celsius */ );

    /// Parses string from system call to find CPU temperatures
    /** When a valid string is read in, the value from that string is stored.
     * This calls the appropriate function based on m_sysType.
     *
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
    int parseCPUTemperatures( float             &temp, /**< [out] parsed temperature in Celsius */
                              const std::string &line  /**< [in] one sensors output line */
    );

    /// Parses string from system call to find CPU temperatures on an Intel system
    /** When a valid string is read in, the value from that string is stored
     *
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
    int parseCPUTemperaturesIntel( float             &temp, /**< [out] parsed temperature in Celsius */
                                   const std::string &line  /**< [in] one sensors output line */
    );

    /// Parses string from system call to find CPU temperatures on an AMD system
    /** When a valid string is read in, the value from that string is stored
     *
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
    int parseCPUTemperaturesAMD( float             &temp, /**< [out] parsed temperature in Celsius */
                                 const std::string &line  /**< [in] one sensors output line */
    );

    /// Checks if any core temperatures are warning or critical levels
    /** Compare against configured thresholds or the limits discovered from Intel sensors output.
     *
     * \returns 1 if a temperature value is at the warning level
     * \returns 2 if a temperature value is at critical level
     * \returns 0 otherwise (all temperatures are considered normal)
     */
    int criticalCoreTemperature(
        std::vector<float> &v /**< [in] CPU temperatures to compare with configured thresholds */ );

    /// Finds all CPU core usage loads
    /** Makes system call and then parses result to add usage loads to vector of values
     *
     * \returns -1 on error with system command or output file reading
     * \returns 0 on completion
     */
    int findCPULoads( std::vector<float> &loads /**< [out] measured CPU activity fractions */ );

    /// Parses string from system call to find CPU usage loads
    /** Convert the final idle-percentage field to an activity fraction.
     *
     * \returns -1 for empty input or a failed numeric conversion.
     * \returns 1 when the line has too few fields to contain a CPU sample.
     * \returns 0 when a value is stored.
     */
    int parseCPULoads( float             &loadVal, /**< [out] CPU activity fraction parsed from the idle percentage */
                       const std::string &line     /**< [in] one mpstat output line */
    );

    /// Finds all drive temperatures
    /** Makes 'hddtemp' system call and then parses result to add temperatures to vector of values
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on successful completion otherwise
     */
    int findDiskTemperature( std::vector<std::string> &hdd_names, /**< [out] drive names reported by hddtemp */
                             std::vector<float> &hdd_temps /**< [out] temperatures in Celsius, ordered like hdd_names */
    );

    /// Parses string from system call to find drive temperatures
    /** When a valid string is read in, the drive name and value from that string is stored
     *
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
    int parseDiskTemperature( std::string       &driveName, /**< [out] parsed drive name without the /dev/ prefix */
                              float             &temp,      /**< [out] parsed drive temperature in Celsius */
                              const std::string &line       /**< [in] one hddtemp output line */
    );

    /// Checks if any drive temperatures are warning or critical levels
    /** Warning and critical temperatures are either user-defined or generated based on initial drive temperature values
     *
     * \returns 1 if a temperature value is at the warning level
     * \returns 2 if a temperature value is at critical level
     * \returns 0 otherwise (all temperatures are considered normal)
     */
    int criticalDiskTemperature(
        std::vector<float> &v /**< [in] drive temperatures to compare with configured thresholds */ );

    /// Finds usages of space for following directory paths: /; /data; /boot
    /** Usage values are fractions between zero and one. A path absent from df output retains its previous value.
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 if at least one of the return values is found
     */
    int findDiskUsage( float &rootUsage, /**< [in,out] utilization fraction for the root filesystem */
                       float &dataUsage, /**< [in,out] utilization fraction for the /data filesystem */
                       float &bootUsage  /**< [in,out] utilization fraction for the /boot filesystem */
    );

    /// Parses string from system call to find drive usage space
    /** When a valid string is read in, the value from that string is stored
     *
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
    int parseDiskUsage( std::string line,      /**< [in] one df output line */
                        float      &rootUsage, /**< [in,out] root utilization, updated when this line identifies / */
                        float &dataUsage, /**< [in,out] data utilization, updated when this line identifies /data */
                        float &bootUsage  /**< [in,out] boot utilization, updated when this line identifies /boot */
    );

    /// Finds current RAM usage
    /** This usage value is stored as a decimal value between 0 and 1 (e.g. value of 0.39 means RAM usage is 39%)
     *
     * \returns -1 on error with system command or output reading
     * \returns 0 on completion
     */
    int findRamUsage( float &ramUsage /**< [in,out] RAM utilization fraction, updated when a sample is parsed */ );

    /// Parses string from system call to find RAM usage
    /** When a valid string is read in, the value from that string is stored
     *
     * \returns -1 on invalid string being read in
     * \returns 0 on completion and storing of value
     */
    int parseRamUsage( std::string line, /**< [in] one free -m output line */
                       float &ramUsage   /**< [in,out] RAM utilization fraction, updated for a recognized memory row */
    );

    ///@}

    /** \name Chrony Status - Data
     * @{ */
  protected:
    /// Reference identifier reported by chronyc tracking.
    std::string m_chronySourceMac;

    /// Address of the current Chrony synchronization source.
    std::string m_chronySourceIP;

    /// Published YES/NO synchronization indication derived from the source.
    std::string m_chronySynch;

    /// Current system-time correction in seconds.
    double m_chronySystemTime{ 0 };

    /// Most recent estimated clock offset in seconds.
    double m_chronyLastOffset{ 0 };

    /// Root-mean-square clock offset in seconds.
    double m_chronyRMSOffset{ 0 };

    /// Estimated clock-frequency correction in parts per million.
    double m_chronyFreq{ 0 };

    /// Residual frequency error in parts per million.
    double m_chronyResidFreq{ 0 };

    /// Estimated frequency-error bound in parts per million.
    double m_chronySkew{ 0 };

    /// Total delay to the reference clock in seconds.
    double m_chronyRootDelay{ 0 };

    /// Accumulated dispersion to the reference clock in seconds.
    double m_chronyRootDispersion{ 0 };

    /// Interval between clock updates in seconds.
    double m_chronyUpdateInt{ 0 };

    /// Leap-second status reported by chronyc.
    std::string m_chronyLeap;

    /// INDI synchronization state and source identity.
    pcf::IndiProperty m_indiP_chronyStatus;

    /// INDI clock-correction and offset measurements.
    pcf::IndiProperty m_indiP_chronyStats;

    ///@}

    /** \name Chrony Status
     * @{ */
  public:
    /// Collect Chrony tracking status and record changed status/statistics.
    /** Uses the chrony tracking command
     *
     * \returns -1 on error
     * \returns 0 on success
     */
    int findChronyStatus();

    ///@}

    /** \name Set Latency - Data
     * @{ */

    /// Requested low-latency mode, applied by the existing latency-control thread.
    bool m_setLatency{ false };

    /// Scheduling priority requested for the latency-control thread.
    int m_setlatThreadPrio{ 0 };

    /// Owned worker that applies CPU governor and latency settings.
    std::thread m_setlatThread;

    /// Startup handshake preventing worker activity before thread configuration completes.
    bool m_setlatThreadInit{ true };

    /// Kernel thread ID published for the latency-control worker.
    pid_t m_setlatThreadID{ 0 };

    /// INDI scheduling and identity details for the latency-control thread.
    pcf::IndiProperty m_setlatThreadProp;

    /// INDI toggle carrying the requested low-latency state.
    pcf::IndiProperty m_indiP_setlat;

    ///@}

    /** \name Set Latency
     * @{ */

    /// Forward the framework thread entrypoint to setlatThreadExec().
    static void setlatThreadStart( sysMonitor *s /**< [in] a pointer to a sysMonitor instance (normally this) */ );

    /// Apply requested CPU governor/latency settings until app shutdown.
    void setlatThreadExec();

  public:
    /// Accept the INDI toggle requesting low-latency CPU operation.
    /** The interface macro declares the received-property callback and its static dispatch wrapper. */
    INDI_NEWCALLBACK_DECL( sysMonitor, m_indiP_setlat );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Force records whose telemetry intervals have elapsed.
    int checkRecordTimes();

    /// Force a periodic CPU load record from the current cached state.
    int recordTelem( const telem_coreloads *unused /**< [in] unused telemetry type selector */ );

    /// Force a periodic CPU temperature record from the current cached state.
    int recordTelem( const telem_coretemps *unused /**< [in] unused telemetry type selector */ );

    /// Force a periodic drive temperature record from the current cached state.
    int recordTelem( const telem_drivetemps *unused /**< [in] unused telemetry type selector */ );

    /// Force a periodic resource utilization record from the current cached state.
    int recordTelem( const telem_usage *unused /**< [in] unused telemetry type selector */ );

    /// Force a periodic Chrony synchronization record from the current cached state.
    int recordTelem( const telem_chrony_status *unused /**< [in] unused telemetry type selector */ );

    /// Force a periodic Chrony clock-statistics record from the current cached state.
    int recordTelem( const telem_chrony_stats *unused /**< [in] unused telemetry type selector */ );

    /// Force a periodic GPU record without claiming another measurement.
    int recordTelem( const telem_gpu *unused /**< [in] telemetry type selector */ );

    /// Force a periodic IRQ record without claiming another check.
    int recordTelem( const telem_irq *unused /**< [in] telemetry type selector */ );

    /// Force a periodic RAID record without claiming another check.
    int recordTelem( const telem_raid *unused /**< [in] telemetry type selector */ );

    /// Record CPU load fractions according to the existing change/force policy.
    int recordCoreLoads( bool force = false /**< [in] emit a record regardless of cached values */ );

    /// Record CPU temperatures according to the existing change/force policy.
    int recordCoreTemps( bool force = false /**< [in] emit a record regardless of cached values */ );

    /// Record identified drive temperatures according to the existing change/force policy.
    int recordDriveTemps( bool force = false /**< [in] emit a record regardless of cached values */ );

    /// Record disk and RAM utilization according to the existing change/force policy.
    int recordUsage( bool force = false /**< [in] emit a record regardless of cached values */ );

    /// Record Chrony synchronization state according to the existing change/force policy.
    int recordChronyStatus( bool force = false /**< [in] emit a record regardless of cached values */ );

    /// Record Chrony clock statistics according to the existing change/force policy.
    int recordChronyStats( bool force = false /**< [in] emit a record regardless of cached values */ );

    ///@}
};

inline sysMonitor::sysMonitor() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    //m_loopPause = 100000; //Set default to 1 milli-second due to mpstat averaging time of 1 sec.
    return;
}

void sysMonitor::setupConfig()
{
    config.add( "sysType",
                "",
                "sysType",
                argType::Required,
                "",
                "sysType",
                false,
                "string",
                "The system type, Intel (default) or AMD" );
    config.add( "diskNames",
                "",
                "diskNames",
                argType::Required,
                "",
                "diskNames",
                false,
                "vector<string>",
                "The names (/dev/sdX) of the drives to monitor" );
    config.add( "warningCoreTemp",
                "",
                "warningCoreTemp",
                argType::Required,
                "",
                "warningCoreTemp",
                false,
                "int",
                "The warning temperature for CPU cores." );
    config.add( "criticalCoreTemp",
                "",
                "criticalCoreTemp",
                argType::Required,
                "",
                "criticalCoreTemp",
                false,
                "int",
                "The critical temperature for CPU cores." );
    config.add( "warningDiskTemp",
                "",
                "warningDiskTemp",
                argType::Required,
                "",
                "warningDiskTemp",
                false,
                "int",
                "The warning temperature for the disk." );
    config.add( "criticalDiskTemp",
                "",
                "criticalDiskTemp",
                argType::Required,
                "",
                "criticalDiskTemp",
                false,
                "int",
                "The critical temperature for disk." );

    config.add( "gpu.enabled",
                "",
                "gpu.enabled",
                argType::Required,
                "gpu",
                "enabled",
                false,
                "bool",
                "Enable read-only NVIDIA GPU diagnostics (default false)." );
    config.add( "gpu.interval",
                "",
                "gpu.interval",
                argType::Required,
                "gpu",
                "interval",
                false,
                "double",
                "GPU polling interval in seconds (default 5)." );
    config.add( "gpu.timeout",
                "",
                "gpu.timeout",
                argType::Required,
                "gpu",
                "timeout",
                false,
                "double",
                "GPU command deadline in seconds (default 2)." );
    config.add( "gpu.expectedUUIDs",
                "",
                "gpu.expectedUUIDs",
                argType::Required,
                "gpu",
                "expectedUUIDs",
                false,
                "vector<string>",
                "Optional GPU UUIDs required at startup." );
    config.add( "irq.interval",
                "",
                "irq.interval",
                argType::Required,
                "irq",
                "interval",
                false,
                "double",
                "Dedicated IRQ checking interval in seconds (default 5). Configure groups in [irq.NAME] sections." );

    config.add( "raid.enabled",
                "",
                "raid.enabled",
                argType::Required,
                "raid",
                "enabled",
                false,
                "bool",
                "Enable read-only Linux MD array discovery (default true)." );
    config.add( "raid.interval",
                "",
                "raid.interval",
                argType::Required,
                "raid",
                "interval",
                false,
                "double",
                "RAID polling interval in seconds (default 5)." );

    TELEMETER_SETUP_CONFIG( config );
}

void sysMonitor::loadConfig()
{
    std::string st;
    //Configure for default, such that logs are correct after config
    if( m_sysType == sysType::Intel )
    {
        st = "Intel";
    }
    else if( m_sysType == sysType::AMD )
    {
        st = "AMD";
    }

    config( st, "sysType" );

    if( st == "Intel" )
    {
        m_sysType = sysType::Intel;
    }
    else if( st == "AMD" )
    {
        m_sysType = sysType::AMD;
    }
    else
    {
        log<software_critical>( { __FILE__, __LINE__, "Invalid system type specified." } );
        m_shutdown = 1;
    }

    config( m_diskNameList, "diskNames" );
    config( m_warningCoreTemp, "warningCoreTemp" );
    config( m_criticalCoreTemp, "criticalCoreTemp" );
    config( m_warningDiskTemp, "warningDiskTemp" );
    config( m_criticalDiskTemp, "criticalDiskTemp" );

    if( loadDiagnosticsConfig( config ) < 0 )
    {
        m_shutdown = true;
    }

    dev::telemeter<sysMonitor>::loadConfig( config );
}

int sysMonitor::appStartup()
{

    if( startupDiagnostics() < 0 )
    {
        return -1;
    }

    REG_INDI_NEWPROP_NOCB( m_indiP_core_temps, "core_temps", pcf::IndiProperty::Number );
    m_indiP_core_temps.add( pcf::IndiElement( "max" ) );
    m_indiP_core_temps.add( pcf::IndiElement( "min" ) );
    m_indiP_core_temps.add( pcf::IndiElement( "mean" ) );

    REG_INDI_NEWPROP_NOCB( m_indiP_core_loads, "core_loads", pcf::IndiProperty::Number );
    m_indiP_core_loads.add( pcf::IndiElement( "max" ) );
    m_indiP_core_loads.add( pcf::IndiElement( "min" ) );
    m_indiP_core_loads.add( pcf::IndiElement( "mean" ) );

    REG_INDI_NEWPROP_NOCB( m_indiP_drive_temps, "drive_temps", pcf::IndiProperty::Number );
    findDiskTemperature( m_diskNames, m_diskTemps );

    for( unsigned int i = 0; i < m_diskTemps.size(); i++ )
    {
        m_indiP_drive_temps.add( pcf::IndiElement( m_diskNames[i] ) );
        m_indiP_drive_temps[m_diskNames[i]].set<double>( m_diskTemps[i] );
    }

    REG_INDI_NEWPROP_NOCB( m_indiP_usage, "resource_use", pcf::IndiProperty::Number );
    m_indiP_usage.add( pcf::IndiElement( "root_usage" ) );
    m_indiP_usage.add( pcf::IndiElement( "boot_usage" ) );
    m_indiP_usage.add( pcf::IndiElement( "data_usage" ) );
    m_indiP_usage.add( pcf::IndiElement( "ram_usage" ) );

    m_indiP_usage["root_usage"].set<double>( 0.0 );
    m_indiP_usage["boot_usage"].set<double>( 0.0 );
    m_indiP_usage["data_usage"].set<double>( 0.0 );
    m_indiP_usage["ram_usage"].set<double>( 0.0 );

    REG_INDI_NEWPROP_NOCB( m_indiP_chronyStatus, "chrony_status", pcf::IndiProperty::Text );
    m_indiP_chronyStatus.add( pcf::IndiElement( "synch" ) );
    m_indiP_chronyStatus.add( pcf::IndiElement( "source" ) );

    REG_INDI_NEWPROP_NOCB( m_indiP_chronyStats, "chrony_stats", pcf::IndiProperty::Number );
    m_indiP_chronyStats.add( pcf::IndiElement( "system_time" ) );
    m_indiP_chronyStats.add( pcf::IndiElement( "last_offset" ) );
    m_indiP_chronyStats.add( pcf::IndiElement( "rms_offset" ) );

    createStandardIndiToggleSw( m_indiP_setlat, "set_latency" );
    registerIndiPropertyNew( m_indiP_setlat, INDI_NEWCALLBACK( m_indiP_setlat ) );

    TELEMETER_APP_STARTUP;

    if( threadStart( m_setlatThread,
                     m_setlatThreadInit,
                     m_setlatThreadID,
                     m_setlatThreadProp,
                     m_setlatThreadPrio,
                     "",
                     "set_latency",
                     this,
                     setlatThreadStart ) < 0 )
    {
        log<software_critical>( { __FILE__, __LINE__ } );
        return -1;
    }

    state( stateCodes::READY );
    return 0;
}

int sysMonitor::appLogic()
{

    if( pollDiagnostics( std::chrono::steady_clock::now() ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__, "Could not publish system diagnostics." } );
    }

    m_coreTemps.clear();
    int rvCPUTemp = findCPUTemperatures( m_coreTemps );

    if( rvCPUTemp >= 0 )
    {
        rvCPUTemp = criticalCoreTemperature( m_coreTemps );
    }

    if( rvCPUTemp >= 0 )
    {
        if( rvCPUTemp == 1 )
        {
            log<telem_coretemps>( m_coreTemps, logPrio::LOG_WARNING );
        }
        else if( rvCPUTemp == 2 )
        {
            log<telem_coretemps>( m_coreTemps, logPrio::LOG_ALERT );
        }

        recordCoreTemps();
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, "Could not log values for CPU core temps." } );
    }

    m_coreLoads.clear();
    int rvCPULoad = findCPULoads( m_coreLoads );

    if( rvCPULoad >= 0 )
    {
        recordCoreLoads();
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, "Could not log values for CPU core loads." } );
    }

    m_diskNames.clear();
    m_diskTemps.clear();
    int rvDiskTemp = findDiskTemperature( m_diskNames, m_diskTemps );

    if( rvDiskTemp >= 0 )
    {
        rvDiskTemp = criticalDiskTemperature( m_diskTemps );
    }

    if( rvDiskTemp >= 0 )
    {
        if( rvDiskTemp == 1 )
        {
            log<telem_drivetemps>( { m_diskNames, m_diskTemps }, logPrio::LOG_WARNING );
        }
        else if( rvDiskTemp == 2 )
        {
            log<telem_drivetemps>( { m_diskNames, m_diskTemps }, logPrio::LOG_ALERT );
        }

        recordDriveTemps();
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, "Could not log values for drive temps." } );
    }

    int rvDiskUsage = findDiskUsage( m_rootUsage, m_dataUsage, m_bootUsage );
    int rvRamUsage  = findRamUsage( m_ramUsage );

    if( rvDiskUsage >= 0 && rvRamUsage >= 0 )
    {
        recordUsage();
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, "Could not log values for usage." } );
    }

    if( findChronyStatus() == 0 )
    {
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, "Could not get chronyd status." } );
    }

    if( telemeter<sysMonitor>::appLogic() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return 0;
    }

    updateVals();

    return 0;
}

int sysMonitor::appShutdown()
{
    m_gpuCommand.shutdown();
    try
    {
        if( m_setlatThread.joinable() )
        {
            m_setlatThread.join();
        }
    }
    catch( ... )
    {
    }

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int sysMonitor::findCPUTemperatures( std::vector<float> &temps )
{
    std::vector<std::string> commandList{ "sensors" };

    std::vector<std::string> commandOutput, commandError;

    if( sys::runCommand( commandOutput, commandError, commandList ) < 0 )
    {
        if( commandOutput.size() < 1 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, commandOutput[0] } );
        }
    }

    if( commandError.size() > 0 )
    {
        for( size_t n = 0; n < commandError.size(); ++n )
        {
            log<software_error>( { __FILE__, __LINE__, "sensors stderr: " + commandError[n] } );
        }
    }

    int rv = -1;

    for( size_t n = 0; n < commandOutput.size(); ++n )
    {
        float tempVal;

        if( parseCPUTemperatures( tempVal, commandOutput[n] ) == 0 )
        {
            temps.push_back( tempVal );
            rv = 0;
        }
    }

    return rv;
}

int sysMonitor::parseCPUTemperatures( float &temp, const std::string &line )
{
    if( m_sysType == sysType::Intel )
    {
        return parseCPUTemperaturesIntel( temp, line );
    }
    else if( m_sysType == sysType::AMD )
    {
        return parseCPUTemperaturesAMD( temp, line );
    }
    else
    {
        log<software_error>( { __FILE__, __LINE__, "invalid system type" } );
        return -1;
    }
}

int sysMonitor::parseCPUTemperaturesIntel( float &temp, const std::string &line )
{
    if( line.length() <= 1 )
    {
        temp = -999;
        return -1;
    }

    std::string str = line.substr( 0, 5 );

    if( str.compare( "Core " ) == 0 )
    {
        size_t st = line.find( ':', 0 );

        if( st == std::string::npos )
        {
            log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing CPU temperatures." } );
            temp = -999;
            return -1;
        }

        ++st;

        size_t ed = line.find( 'C', st );

        if( ed == std::string::npos )
        {
            log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing CPU temperatures." } );
            temp = -999;
            return -1;
        }

        --ed;

        std::string temp_str = line.substr( st, ed - st );

        try
        {
            temp = std::stof( temp_str );
        }
        catch( const std::invalid_argument &e )
        {
            log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing CPU temperatures." } );
            temp = -999;
            return -1;
        }

        if( m_warningCoreTemp == 0 )
        {
            std::istringstream       iss( line );
            std::vector<std::string> tokens{ std::istream_iterator<std::string>{ iss },
                                             std::istream_iterator<std::string>{} };
            try
            {
                tokens.at( 5 ).pop_back();
                tokens.at( 5 ).pop_back();
                tokens.at( 5 ).pop_back();
                tokens.at( 5 ).pop_back();
                tokens.at( 5 ).erase( 0, 1 );
                m_warningCoreTemp = std::stof( tokens.at( 5 ) );
            }
            catch( const std::invalid_argument &e )
            {
                log<software_error>(
                    { __FILE__, __LINE__, "Invalid read occured when parsing warning CPU temperatures." } );
                return -1;
            }
        }

        if( m_criticalCoreTemp == 0 )
        {
            std::istringstream       iss( line );
            std::vector<std::string> tokens{ std::istream_iterator<std::string>{ iss },
                                             std::istream_iterator<std::string>{} };
            try
            {
                tokens.at( 8 ).pop_back();
                tokens.at( 8 ).pop_back();
                tokens.at( 8 ).pop_back();
                tokens.at( 8 ).pop_back();
                tokens.at( 8 ).erase( 0, 1 );
                m_criticalCoreTemp = std::stof( tokens.at( 8 ) );
            }
            catch( const std::invalid_argument &e )
            {
                log<software_error>(
                    { __FILE__, __LINE__, "Invalid read occured when parsing critical CPU temperatures." } );
                temp = -999;
                return -1;
            }
        }

        return 0;
    }
    else
    {
        temp = -999;
        return -1;
    }
}

int sysMonitor::parseCPUTemperaturesAMD( float &temp, const std::string &line )
{
    if( line.length() <= 1 )
    {
        temp = -999;
        return -1;
    }

    std::string str = line.substr( 0, 6 );

    if( str.compare( "Tctl: " ) == 0 )
    {
        size_t ed = line.find( 'C', 0 );

        if( ed == std::string::npos )
        {
            log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing CPU temperatures." } );
            temp = -999;
            return -1;
        }

        str = line.substr( 7, ( ( ed - 1 ) - 7 ) ); //ed-1 to eat degree.

        try
        {
            temp = std::stof( str );
        }
        catch( ... )
        {
            log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing CPU temperatures." } );
            temp = -999;
            return -1;
        }

        return 0;
    }
    else
    {
        temp = -999;
        return -1;
    }
}

int sysMonitor::criticalCoreTemperature( std::vector<float> &v )
{
    int coreNum = 0, rv = 0;

    for( auto it : v )
    {
        float temp = it;

        if( temp >= m_warningCoreTemp && temp < m_criticalCoreTemp )
        {
            std::cout << "Warning temperature for Core " << coreNum << std::endl;

            if( rv < 2 )
            {
                rv = 1;
            }
        }
        else if( temp >= m_criticalCoreTemp )
        {
            std::cout << "Critical temperature for Core " << coreNum << std::endl;
            rv = 2;
        }

        ++coreNum;
    }

    return rv;
}

int sysMonitor::findCPULoads( std::vector<float> &loads )
{
    std::vector<std::string> commandList{ "mpstat", "-P", "ALL", "1", "1" };
    std::vector<std::string> commandOutput, commandError;

    if( sys::runCommand( commandOutput, commandError, commandList ) < 0 )
    {
        if( commandOutput.size() < 1 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, commandOutput[0] } );
        }
    }

    if( commandError.size() > 0 )
    {
        for( size_t n = 0; n < commandError.size(); ++n )
        {
            log<software_error>( { __FILE__, __LINE__, "mpstat stderr: " + commandError[n] } );
        }
    }

    int rv = -1;
    // If output lines are less than 5 (with one CPU, guarenteed output is 5)
    if( commandOutput.size() < 5 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "not enough lines returned by mpstat" } );
    }
    //start iterating at fourth line
    for( auto line = commandOutput.begin() + 4; line != commandOutput.end(); line++ )
    {
        float loadVal;

        if( parseCPULoads( loadVal, *line ) == 0 )
        {
            loads.push_back( loadVal );
            rv = 0;
        }
    }

    return rv;
}

int sysMonitor::parseCPULoads( float &loadVal, const std::string &line )
{
    if( line.length() <= 1 )
    {
        log<software_error>( { __FILE__, __LINE__, "zero length line in parseCPULoads." } );
        return -1;
    }

    std::istringstream iss( line );

    std::vector<std::string> tokens( std::istream_iterator<std::string>{ iss }, std::istream_iterator<std::string>{} );

    if( tokens.size() < 8 )
    {
        return 1;
    }

    float cpu_load;
    try
    {
        cpu_load = 100.0 - std::stof( tokens.at( tokens.size() - 1 ) );
    }
    catch( const std::invalid_argument &e )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing CPU core usage." } );
        return -1;
    }
    catch( const std::out_of_range &e )
    {
        log<software_error>( { __FILE__, __LINE__, "Out of range exception in parseCPULoads." } );
        return -1;
    }

    cpu_load /= 100;
    loadVal = cpu_load;
    return 0;
}

int sysMonitor::findDiskTemperature( std::vector<std::string> &hdd_names, std::vector<float> &hdd_temps )
{
    std::vector<std::string> commandList{ "hddtemp" };

    std::vector<std::string> commandOutput, commandError;

    if( sys::runCommand( commandOutput, commandError, commandList ) < 0 )
    {
        if( commandOutput.size() < 1 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, commandOutput[0] } );
        }
    }

    if( commandError.size() > 0 )
    {
        for( size_t n = 0; n < commandError.size(); ++n )
        {
            if( commandError[n].find( "doesn't have a temperature sensor" ) != std::string::npos )
            {
                continue;
            }

            log<software_error>( { __FILE__, __LINE__, "hddtemp stderr: " + commandError[n] } );
        }
    }

    int rv = -1;

    for( auto line : commandOutput )
    {
        std::string driveName;
        float       tempVal;

        if( parseDiskTemperature( driveName, tempVal, line ) == 0 )
        {
            hdd_names.push_back( driveName );
            hdd_temps.push_back( tempVal );
            rv = 0;
        }
    }

    return rv;

    //return 0;
}

int sysMonitor::parseDiskTemperature( std::string &driveName, float &hdd_temp, const std::string &line )
{
    float tempValue;

    if( line.length() <= 6 )
    {
        driveName = "";
        hdd_temp  = -999;
        return -1;
    }

    size_t sp = line.find( ':', 0 );
    driveName = line.substr( 5, sp - 5 );

    std::istringstream       iss( line );
    std::vector<std::string> tokens{ std::istream_iterator<std::string>{ iss }, std::istream_iterator<std::string>{} };

    for( auto temp_s : tokens )
    {
        try
        {
            if( isdigit( temp_s.at( 0 ) ) && temp_s.substr( temp_s.length() - 1, 1 ) == "C" )
            {
                temp_s.pop_back();
                temp_s.pop_back();
                try
                {
                    tempValue = std::stof( temp_s );
                }
                catch( const std::invalid_argument &e )
                {
                    log<software_error>(
                        { __FILE__, __LINE__, "Invalid read occured when parsing drive temperatures." } );
                    hdd_temp  = -999;
                    driveName = "";
                    return -1;
                }

                hdd_temp = tempValue;

                if( m_warningDiskTemp == 0 )
                {
                    m_warningDiskTemp = tempValue + ( .1 * tempValue );
                }

                if( m_criticalDiskTemp == 0 )
                {
                    m_criticalDiskTemp = tempValue + ( .2 * tempValue );
                }

                return 0;
            }
        }
        catch( const std::out_of_range &e )
        {
            hdd_temp  = -999;
            driveName = "";
            return -1;
        }
    }

    hdd_temp  = -999;
    driveName = "";
    return -1;
}

int sysMonitor::criticalDiskTemperature( std::vector<float> &v )
{
    int rv = 0;

    for( auto it : v )
    {
        float temp = it;

        if( temp >= m_warningDiskTemp && temp < m_criticalDiskTemp )
        {
            std::cout << "Warning temperature for Disk" << std::endl;

            if( rv < 2 )
            {
                rv = 1;
            }
        }
        else if( temp >= m_criticalDiskTemp )
        {
            std::cout << "Critical temperature for Disk " << std::endl;
            rv = 2;
        }
    }

    return rv;
}

int sysMonitor::findDiskUsage( float &rootUsage, float &dataUsage, float &bootUsage )
{
    std::vector<std::string> commandList{ "df" };

    std::vector<std::string> commandOutput, commandError;

    if( sys::runCommand( commandOutput, commandError, commandList ) < 0 )
    {
        if( commandOutput.size() < 1 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, commandOutput[0] } );
        }
    }

    if( commandError.size() > 0 )
    {
        for( size_t n = 0; n < commandError.size(); ++n )
        {
            log<software_error>( { __FILE__, __LINE__, "df stderr: " + commandError[n] } );
        }
    }

    int rv = -1;

    for( auto line : commandOutput )
    {
        int rvDiskUsage = parseDiskUsage( line, rootUsage, dataUsage, bootUsage );

        if( rvDiskUsage == 0 )
        {
            rv = 0;
        }
    }

    return rv;
}

int sysMonitor::parseDiskUsage( std::string line, float &rootUsage, float &dataUsage, float &bootUsage )
{
    if( line.length() <= 1 )
    {
        return -1;
    }

    std::istringstream       iss( line );
    std::vector<std::string> tokens{ std::istream_iterator<std::string>{ iss }, std::istream_iterator<std::string>{} };

    try
    {
        if( tokens.at( 5 ).compare( "/" ) == 0 )
        {
            tokens.at( 4 ).pop_back();
            try
            {
                rootUsage = std::stof( tokens.at( 4 ) ) / 100;
                return 0;
            }
            catch( const std::invalid_argument &e )
            {
                log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing drive usage." } );
                return -1;
            }
        }
        else if( tokens.at( 5 ).compare( "/data" ) == 0 )
        {
            tokens.at( 4 ).pop_back();
            try
            {
                dataUsage = std::stof( tokens.at( 4 ) ) / 100;
                return 0;
            }
            catch( const std::invalid_argument &e )
            {
                log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing drive usage." } );
                return -1;
            }
        }
        else if( tokens.at( 5 ).compare( "/boot" ) == 0 )
        {
            tokens.at( 4 ).pop_back();
            try
            {
                bootUsage = std::stof( tokens.at( 4 ) ) / 100;
                return 0;
            }
            catch( const std::invalid_argument &e )
            {
                log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing drive usage." } );
                return -1;
            }
        }
    }
    catch( const std::out_of_range &e )
    {
        return -1;
    }

    return -1;
}

int sysMonitor::findRamUsage( float &ramUsage )
{
    std::vector<std::string> commandList{ "free", "-m" };

    std::vector<std::string> commandOutput, commandError;

    if( sys::runCommand( commandOutput, commandError, commandList ) < 0 )
    {
        if( commandOutput.size() < 1 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, commandOutput[0] } );
        }
    }

    if( commandError.size() > 0 )
    {
        for( size_t n = 0; n < commandError.size(); ++n )
        {
            log<software_error>( { __FILE__, __LINE__, "free stderr: " + commandError[n] } );
        }
    }

    for( auto line : commandOutput )
    {
        if( parseRamUsage( line, ramUsage ) == 0 )
        {
            return 0;
        }
    }

    return -1;
}

int sysMonitor::parseRamUsage( std::string line, float &ramUsage )
{
    if( line.length() <= 1 )
    {
        return -1;
    }

    std::istringstream       iss( line );
    std::vector<std::string> tokens{ std::istream_iterator<std::string>{ iss }, std::istream_iterator<std::string>{} };
    try
    {
        if( tokens.at( 0 ).compare( "Mem:" ) != 0 )
        {
            return -1;
        }

        ramUsage = std::stof( tokens.at( 2 ) ) / std::stof( tokens.at( 1 ) );

        if( ramUsage > 1 || ramUsage == 0 )
        {
            ramUsage = -1;
            return -1;
        }

        return 0;
    }
    catch( const std::invalid_argument &e )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid read occured when parsing RAM usage." } );
        return -1;
    }
    catch( const std::out_of_range &e )
    {
        return -1;
    }
}

int sysMonitor::findChronyStatus()
{
    std::vector<std::string> commandList{ "chronyc", "-c", "tracking" };

    std::vector<std::string> commandOutput, commandError;

    if( sys::runCommand( commandOutput, commandError, commandList ) < 0 )
    {
        if( commandOutput.size() < 1 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__ } );
        }
        else
        {
            return log<software_error, -1>( { __FILE__, __LINE__, commandOutput[0] } );
        }
    }

    if( commandError.size() > 0 )
    {
        for( size_t n = 0; n < commandError.size(); ++n )
        {
            log<software_error>( { __FILE__, __LINE__, "chronyc stderr: " + commandError[n] } );
        }
    }

    if( commandOutput.size() < 1 )
    {
        log<software_error>( { __FILE__, __LINE__, "no response from chronyc -c" } );
        return -1;
    }

    std::vector<std::string> results;
    mx::ioutils::parseStringVector( results, commandOutput[0], ',' );

    if( results.size() < 1 )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong number of fields from chronyc -c" } );
        return -1;
    }

    static std::string last_mac;
    static std::string last_ip;
    m_chronySourceMac = results[0];
    m_chronySourceIP  = results[1];

    if( m_chronySourceMac == "7F7F0101" || m_chronySourceIP == "127.0.0.1" )
    {
        m_chronySynch = "NO";
        log<text_log>( "chrony is not synchronized", logPrio::LOG_WARNING );
    }
    else
    {
        m_chronySynch = "YES";
    }

    if( last_mac != m_chronySourceMac || last_ip != m_chronySourceIP )
    {
        log<text_log>( "chrony is synchronizing to " + m_chronySourceMac + " / " + m_chronySourceIP );
        last_mac = m_chronySourceMac;
        last_ip  = m_chronySourceIP;
    }

    m_chronySystemTime     = std::stod( results[4] );
    m_chronyLastOffset     = std::stod( results[5] );
    m_chronyRMSOffset      = std::stod( results[6] );
    m_chronyFreq           = std::stod( results[7] );
    m_chronyResidFreq      = std::stod( results[8] );
    m_chronySkew           = std::stod( results[9] );
    m_chronyRootDelay      = std::stod( results[10] );
    m_chronyRootDispersion = std::stod( results[11] );
    m_chronyUpdateInt      = std::stod( results[12] );
    m_chronyLeap           = results[13];

    recordChronyStatus();
    recordChronyStats();

    return 0;
}

int sysMonitor::updateVals()
{
    float min, max, mean;

    if( m_coreLoads.size() > 0 )
    {
        min  = m_coreLoads[0];
        max  = m_coreLoads[0];
        mean = m_coreLoads[0];

        for( size_t n = 1; n < m_coreLoads.size(); ++n )
        {
            if( m_coreLoads[n] < min )
            {
                min = m_coreLoads[n];
            }

            if( m_coreLoads[n] > max )
            {
                max = m_coreLoads[n];
            }

            mean += m_coreLoads[n];
        }

        mean /= m_coreLoads.size();

        updateIfChanged<float>( m_indiP_core_loads, { "min", "max", "mean" }, { min, max, mean } );
    }

    if( m_coreTemps.size() > 0 )
    {
        min  = m_coreTemps[0];
        max  = m_coreTemps[0];
        mean = m_coreTemps[0];

        for( size_t n = 1; n < m_coreTemps.size(); ++n )
        {
            if( m_coreTemps[n] < min )
            {
                min = m_coreTemps[n];
            }

            if( m_coreTemps[n] > max )
            {
                max = m_coreTemps[n];
            }

            mean += m_coreTemps[n];
        }

        mean /= m_coreTemps.size();

        updateIfChanged<float>( m_indiP_core_temps, { "min", "max", "mean" }, { min, max, mean } );
    }

    updateIfChanged( m_indiP_drive_temps, m_diskNames, m_diskTemps );

    updateIfChanged<float>( m_indiP_usage,
                            { "root_usage", "boot_usage", "data_usage", "ram_usage" },
                            { m_rootUsage, m_bootUsage, m_dataUsage, m_ramUsage } );

    updateIfChanged( m_indiP_chronyStatus, "synch", m_chronySynch );
    updateIfChanged( m_indiP_chronyStatus, "source", m_chronySourceIP );

    updateIfChanged<double>( m_indiP_chronyStats,
                             { "system_time", "last_offset", "rms_offset" },
                             { m_chronySystemTime, m_chronyLastOffset, m_chronyRMSOffset } );

    if( m_setLatency )
    {
        updateSwitchIfChanged( m_indiP_setlat, "toggle", pcf::IndiElement::On, INDI_OK );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_setlat, "toggle", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

inline void sysMonitor::setlatThreadStart( sysMonitor *s )
{
    s->setlatThreadExec();
}

inline void sysMonitor::setlatThreadExec()
{
    m_setlatThreadID = syscall( SYS_gettid );

    //Wait fpr the thread starter to finish initializing this thread.
    while( m_setlatThreadInit == true && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    int fd = 0;

    while( m_shutdown == 0 )
    {
        if( m_setLatency )
        {
            if( fd <= 0 )
            {
                elevatedPrivileges ep( this );

                for( size_t cpu = 0; cpu < m_coreLoads.size(); ++cpu ) ///\todo this needs error checks
                {
                    std::string cpuFile = "/sys/devices/system/cpu/cpu";
                    cpuFile += std::to_string( cpu );
                    cpuFile += "/cpufreq/scaling_governor";
                    int     wfd    = open( cpuFile.c_str(), O_WRONLY );
                    ssize_t perfsz = sizeof( "performance" );
                    ssize_t wrtsz  = write( wfd, "performance", perfsz );

                    if( wrtsz != perfsz )
                    {
                        log<software_error>(
                            { __FILE__,
                              __LINE__,
                              "error setting performance governor for CPU " + std::to_string( cpu ) } );
                    }

                    close( wfd );
                }

                log<text_log>( "set governor to performance", logPrio::LOG_NOTICE );

                fd = open( "/dev/cpu_dma_latency", O_WRONLY );

                if( fd <= 0 )
                {
                    log<software_error>( { __FILE__, __LINE__, "error opening cpu_dma_latency" } );
                }
                else
                {
                    int l = 0;

                    if( write( fd, &l, sizeof( l ) ) != sizeof( l ) )
                    {
                        log<software_error>( { __FILE__, __LINE__, "error writing to cpu_dma_latency" } );
                    }
                    else
                    {
                        log<text_log>( "set latency to 0", logPrio::LOG_NOTICE );
                    }
                }
            }
        }
        else
        {
            if( fd != 0 )
            {
                close( fd );
                fd = 0;
                log<text_log>( "restored CPU latency to default", logPrio::LOG_NOTICE );

                elevatedPrivileges ep( this );

                for( size_t cpu = 0; cpu < m_coreLoads.size(); ++cpu ) ///\todo this needs error checks
                {
                    std::string cpuFile = "/sys/devices/system/cpu/cpu";
                    cpuFile += std::to_string( cpu );
                    cpuFile += "/cpufreq/scaling_governor";
                    int     wfd   = open( cpuFile.c_str(), O_WRONLY );
                    ssize_t pwrsz = sizeof( "powersave" );
                    ssize_t wrtsz = write( wfd, "powersave", pwrsz );

                    if( wrtsz != pwrsz )
                    {
                        log<software_error>( { __FILE__,
                                               __LINE__,
                                               "error setting powersave governor for CPU " + std::to_string( cpu ) } );
                    }

                    close( wfd );
                }

                log<text_log>( "set governor to powersave", logPrio::LOG_NOTICE );
            }
        }

        sleep( 1 );
    }

    if( fd )
    {
        close( fd );
    }
}

INDI_NEWCALLBACK_DEFN( sysMonitor, m_indiP_setlat )(
    const pcf::IndiProperty &ipRecv /**< [in] received low-latency toggle property */ )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_setlat, ipRecv );

    if( ipRecv.getName() != m_indiP_setlat.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off )
    {
        m_setLatency = false;
    }

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
    {
        m_setLatency = true;
    }

    return 0;
}

inline int sysMonitor::checkRecordTimes()
{
    return telemeter<sysMonitor>::checkRecordTimes( telem_coreloads(),
                                                    telem_coretemps(),
                                                    telem_drivetemps(),
                                                    telem_usage(),
                                                    telem_chrony_status(),
                                                    telem_chrony_stats(),
                                                    telem_gpu(),
                                                    telem_irq(),
                                                    telem_raid() );
}

int sysMonitor::recordTelem( const telem_coreloads * )
{
    return recordCoreLoads( true );
}

int sysMonitor::recordTelem( const telem_coretemps * )
{
    return recordCoreTemps( true );
}

int sysMonitor::recordTelem( const telem_drivetemps * )
{
    return recordDriveTemps( true );
}

int sysMonitor::recordTelem( const telem_usage * )
{
    return recordUsage( true );
}

int sysMonitor::recordTelem( const telem_chrony_status * )
{
    return recordChronyStatus( true );
}

int sysMonitor::recordTelem( const telem_chrony_stats * )
{
    return recordChronyStats( true );
}

int sysMonitor::recordCoreLoads( bool force )
{
    static std::vector<float> old_coreLoads;

    if( old_coreLoads.size() != m_coreLoads.size() )
    {
        old_coreLoads.resize( m_coreLoads.size(), -1e30 );
    }

    bool write = false;

    for( size_t n = 0; n < m_coreLoads.size(); ++n )
    {
        if( m_coreLoads[n] != old_coreLoads[n] )
        {
            write = true;
        }
    }

    if( force || write )
    {
        telem<telem_coreloads>( m_coreLoads );

        for( size_t n = 0; n < m_coreLoads.size(); ++n )
        {
            old_coreLoads[n] = m_coreLoads[n];
        }
    }

    return 0;
}

int sysMonitor::recordCoreTemps( bool force )
{
    static std::vector<float> old_coreTemps;

    if( old_coreTemps.size() != m_coreTemps.size() )
    {
        old_coreTemps.resize( m_coreTemps.size(), -1e30 );
    }

    bool write = false;

    for( size_t n = 0; n < m_coreTemps.size(); ++n )
    {
        if( m_coreTemps[n] != old_coreTemps[n] )
        {
            write = true;
        }
    }

    if( force || write )
    {
        telem<telem_coretemps>( m_coreTemps );

        for( size_t n = 0; n < m_coreTemps.size(); ++n )
        {
            old_coreTemps[n] = m_coreTemps[n];
        }
    }

    return 0;
}

int sysMonitor::recordDriveTemps( bool force )
{
    static std::vector<std::string> old_diskNames;
    static std::vector<float>       old_diskTemps;

    if( old_diskTemps.size() != m_diskTemps.size() || old_diskNames.size() != m_diskNames.size() )
    {
        old_diskNames.resize( m_diskNames.size() );
        old_diskTemps.resize( m_diskTemps.size(), -1e30 );
    }

    bool write = false;

    for( size_t n = 0; n < m_diskTemps.size(); ++n )
    {
        if( m_diskTemps[n] != old_diskTemps[n] || m_diskNames[n] != old_diskNames[n] )
        {
            write = true;
        }
    }

    if( force || write )
    {
        telem<telem_drivetemps>( { m_diskNames, m_diskTemps } );

        for( size_t n = 0; n < m_diskTemps.size(); ++n )
        {
            old_diskNames[n] = m_diskNames[n];
            old_diskTemps[n] = m_diskTemps[n];
        }
    }

    return 0;
}

int sysMonitor::recordUsage( bool force )
{
    static float old_ramUsage  = 0;
    static float old_bootUsage = 0;
    static float old_rootUsage = 0;
    static float old_dataUsage = 0;

    if( old_ramUsage != m_ramUsage || old_bootUsage != m_bootUsage || old_rootUsage != m_rootUsage ||
        old_dataUsage != m_dataUsage || force )
    {
        telem<telem_usage>( { m_ramUsage, m_bootUsage, m_rootUsage, m_dataUsage } );

        old_ramUsage  = m_ramUsage;
        old_bootUsage = m_bootUsage;
        old_rootUsage = m_rootUsage;
        old_dataUsage = m_dataUsage;
    }

    return 0;
}

int sysMonitor::recordChronyStatus( bool force )
{
    static std::string old_chronySourceMac;
    static std::string old_chronySourceIP;
    static std::string old_chronySynch;
    static std::string old_chronyLeap;

    if( old_chronySourceMac != m_chronySourceMac || old_chronySourceIP != m_chronySourceIP ||
        old_chronySynch != m_chronySynch || old_chronyLeap != m_chronyLeap || force )
    {
        telem<telem_chrony_status>( { m_chronySourceMac, m_chronySourceIP, m_chronySynch, m_chronyLeap } );

        old_chronySourceMac = m_chronySourceMac;
        old_chronySourceIP  = m_chronySourceIP;
        old_chronySynch     = m_chronySynch;
        old_chronyLeap      = m_chronyLeap;
    }

    return 0;
}

int sysMonitor::recordChronyStats( bool force )
{
    static double old_chronySystemTime     = 1e50; //to force an update the first time no matter what
    static double old_chronyLastOffset     = 0;
    static double old_chronyRMSOffset      = 0;
    static double old_chronyFreq           = 0;
    static double old_chronyResidFreq      = 0;
    static double old_chronySkew           = 0;
    static double old_chronyRootDelay      = 0;
    static double old_chronyRootDispersion = 0;
    static double old_chronyUpdateInt      = 0;

    if( old_chronySystemTime == m_chronySystemTime || old_chronyLastOffset == m_chronyLastOffset ||
        old_chronyRMSOffset == m_chronyRMSOffset || old_chronyFreq == m_chronyFreq ||
        old_chronyResidFreq == m_chronyResidFreq || old_chronySkew == m_chronySkew ||
        old_chronyRootDelay == m_chronyRootDelay || old_chronyRootDispersion == m_chronyRootDispersion ||
        old_chronyUpdateInt == m_chronyUpdateInt || force )
    {

        telem<telem_chrony_stats>( { m_chronySystemTime,
                                     m_chronyLastOffset,
                                     m_chronyRMSOffset,
                                     m_chronyFreq,
                                     m_chronyResidFreq,
                                     m_chronySkew,
                                     m_chronyRootDelay,
                                     m_chronyRootDispersion,
                                     m_chronyUpdateInt } );

        old_chronySystemTime     = m_chronySystemTime;
        old_chronyLastOffset     = m_chronyLastOffset;
        old_chronyRMSOffset      = m_chronyRMSOffset;
        old_chronyFreq           = m_chronyFreq;
        old_chronyResidFreq      = m_chronyResidFreq;
        old_chronySkew           = m_chronySkew;
        old_chronyRootDelay      = m_chronyRootDelay;
        old_chronyRootDispersion = m_chronyRootDispersion;
        old_chronyUpdateInt      = m_chronyUpdateInt;
    }

    return 0;
}

inline int sysMonitor::loadDiagnosticsConfig( mx::app::appConfigurator &cfg )
{
    auto parseBoolean = []( bool &value, const std::string &text )
    {
        if( text == "true" || text == "1" )
        {
            value = true;
            return true;
        }

        if( text == "false" || text == "0" )
        {
            value = false;
            return true;
        }

        return false;
    };

    std::string enabled = m_gpuEnabled ? "true" : "false";
    cfg( enabled, "gpu.enabled" );

    if( !parseBoolean( m_gpuEnabled, enabled ) )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "gpu.enabled must be true, false, 1, or 0." } );
    }

    cfg( m_gpuInterval, "gpu.interval" );
    cfg( m_gpuTimeout, "gpu.timeout" );
    cfg( m_gpuExpected, "gpu.expectedUUIDs" );
    cfg( m_irqInterval, "irq.interval" );
    cfg( m_raidInterval, "raid.interval" );
    enabled = m_raidEnabled ? "true" : "false";
    cfg( enabled, "raid.enabled" );

    if( !parseBoolean( m_raidEnabled, enabled ) )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "raid.enabled must be true, false, 1, or 0." } );
    }

    if( !mx::math::isFinite( m_gpuInterval ) || m_gpuInterval <= 0 || !mx::math::isFinite( m_gpuTimeout ) ||
        m_gpuTimeout <= 0 || !mx::math::isFinite( m_irqInterval ) || m_irqInterval <= 0 ||
        !mx::math::isFinite( m_raidInterval ) || m_raidInterval <= 0 )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "Diagnostic intervals and timeout must be finite and positive." } );
    }

    std::set<std::string> expected;

    for( auto &uuid : m_gpuExpected )
    {
        const auto key = sysMonitorDetail::gpuKey( uuid );

        if( key.empty() || !expected.insert( key ).second )
        {
            return log<software_critical, -1>(
                { __FILE__, __LINE__, "Invalid or duplicate expected GPU UUID: " + uuid } );
        }

        std::transform( uuid.begin() + 4,
                        uuid.end(),
                        uuid.begin() + 4,
                        []( unsigned char c )
                        {
                            return std::tolower( c );
                        } );
    }

    m_irqRules.clear();
    std::vector<std::string> sections;
    cfg.unusedSections( sections );

    for( const auto &section : sections )
    {
        if( section.compare( 0, 4, "irq." ) != 0 )
        {
            continue;
        }

        sysMonitorDetail::ruleT rule;
        rule.m_name = section.substr( 4 );

        if( rule.m_name == "summary" )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, "IRQ rule name summary is reserved." } );
        }

        cfg.configUnused( rule.m_match, mx::app::iniFile::makeKey( section, "match" ) );
        cfg.configUnused( rule.m_cpuset, mx::app::iniFile::makeKey( section, "cpuset" ) );
        cfg.configUnused( rule.m_cpus, mx::app::iniFile::makeKey( section, "cpus" ) );
        std::string required = "true", requireEffective = "true";
        cfg.configUnused( required, mx::app::iniFile::makeKey( section, "required" ) );
        cfg.configUnused( requireEffective, mx::app::iniFile::makeKey( section, "requireEffective" ) );

        if( !parseBoolean( rule.m_required, required ) || !parseBoolean( rule.m_requireEffective, requireEffective ) )
        {
            return log<software_critical, -1>(
                { __FILE__, __LINE__, "IRQ policy booleans must be true, false, 1, or 0: " + rule.m_name } );
        }

        m_irqRules.push_back( rule );
    }

    std::string error;

    if( !sysMonitorDetail::validateIRQRules( m_irqRules, error ) )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, error } );
    }

    return 0;
}

inline int sysMonitor::startupDiagnostics()
{
    if( startupRAID() < 0 )
    {
        return -1;
    }

    m_gpuStatus = m_gpuEnabled ? "unchecked" : "disabled";
    REG_INDI_NEWPROP_NOCB( m_indiP_gpuStatus, "gpu_status", pcf::IndiProperty::Text );
    m_indiP_gpuStatus.add( pcf::IndiElement( "status", m_gpuStatus ) );
    REG_INDI_NEWPROP_NOCB( m_indiP_irqStatus, "irq_status", pcf::IndiProperty::Text );
    m_indiP_irqStatus.add( pcf::IndiElement( "summary", m_irqRules.empty() ? "disabled" : "unchecked" ) );

    if( registerIndiPropertyNew( m_indiP_irqWarning,
                                 "irq_warning",
                                 pcf::IndiProperty::Switch,
                                 pcf::IndiProperty::ReadOnly,
                                 INDI_IDLE,
                                 pcf::IndiProperty::AnyOfMany,
                                 nullptr ) < 0 )
    {
        return -1;
    }

    bool warning = false;
    m_irqs.clear();
    std::sort( m_irqRules.begin(),
               m_irqRules.end(),
               []( const auto &a, const auto &b )
               {
                   return a.m_name < b.m_name;
               } );

    for( size_t n = 0; n < m_irqRules.size(); ++n )
    {
        const auto &rule = m_irqRules[n];
        log<text_log>( "IRQ policy " + std::to_string( n ) + ": " + rule.m_name + " match=" + rule.m_match +
                           ( rule.m_cpuset.empty() ? " cpus=" + rule.m_cpus : " cpuset=" + rule.m_cpuset ) +
                           " required=" + std::to_string( rule.m_required ) +
                           " requireEffective=" + std::to_string( rule.m_requireEffective ),
                       logPrio::LOG_NOTICE );

        telem_irq::groupT group;
        group.m_name    = rule.m_name;
        group.m_status  = "unchecked";
        group.m_warning = rule.m_required;
        warning         = warning || group.m_warning;
        m_irqs.push_back( group );
        m_indiP_irqStatus.add( pcf::IndiElement( rule.m_name, group.m_status ) );
    }

    m_indiP_irqWarning.add( pcf::IndiElement( "active", warning ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    m_indiP_irqWarning.setState( warning ? INDI_ALERT : INDI_IDLE );

    for( const auto &uuid : m_gpuExpected )
    {
        telem_gpu::gpuT gpu;
        gpu.m_uuid   = uuid;
        gpu.m_status = m_gpuStatus;
        m_gpus.push_back( gpu );
    }

    return publishDiagnostics();
}

inline sys::boundedCommand::result sysMonitor::queryGPUs()
{
    return m_gpuCommand.run(
        { "/usr/bin/nvidia-smi",
          "--query-gpu=index,uuid,pci.bus_id,temperature.gpu,utilization.gpu,utilization.memory,memory."
          "used,memory.total,clocks.current.graphics,clocks.current.sm,clocks.current.memory",
          "--format=csv,noheader,nounits" },
        m_gpuTimeout );
}

inline std::vector<telem_irq::groupT> sysMonitor::queryIRQs( double sampleTime )
{
    return sysMonitorDetail::checkIRQs( m_irqRules, "/proc", cpusetPath(), sampleTime );
}

inline void sysMonitor::acceptGPUResult( const sys::boundedCommand::result &result, double sampleTime )
{
    std::vector<telem_gpu::gpuT> samples;
    const auto                   previousStatus = m_gpuStatus;

    if( result.m_status != sys::boundedCommand::status::success )
    {
        using commandStatus = sys::boundedCommand::status;
        switch( result.m_status )
        {
        case commandStatus::spawnError:
            m_gpuStatus = "could not start /usr/bin/nvidia-smi";
            break;
        case commandStatus::exitError:
            m_gpuStatus = "GPU query exit " + std::to_string( result.m_exitCode );
            break;
        case commandStatus::signalError:
            m_gpuStatus = "GPU query signal " + std::to_string( result.m_exitCode );
            break;
        case commandStatus::timeout:
            m_gpuStatus = "GPU query timed out";
            break;
        case commandStatus::outputLimit:
            m_gpuStatus = "GPU query output exceeded limit";
            break;
        case commandStatus::ioError:
            m_gpuStatus = "GPU query capture failed";
            break;
        case commandStatus::busy:
            m_gpuStatus = "previous GPU query is still pending cleanup";
            break;
        case commandStatus::success:
            break;
        }
    }
    else if( sysMonitorDetail::parseGPUs( samples, result.m_output, sampleTime ) < 0 )
    {
        m_gpuStatus = "invalid GPU response";
    }
    else
    {
        m_gpuStatus = samples.empty() ? "no GPUs found" : "ok";
    }

    for( const auto &old : m_gpus )
    {
        auto found = std::find_if( samples.begin(),
                                   samples.end(),
                                   [&]( const auto &gpu )
                                   {
                                       return gpu.m_uuid == old.m_uuid;
                                   } );

        if( found == samples.end() )
        {
            auto missing     = old;
            missing.m_valid  = 0;
            missing.m_status = m_gpuStatus == "ok" ? "missing" : m_gpuStatus;

            if( std::any_of( samples.begin(),
                             samples.end(),
                             [&]( const auto &gpu )
                             {
                                 return gpu.m_index == old.m_index;
                             } ) )
            {
                missing.m_index = telem_gpu::UnknownIndex;
            }

            samples.push_back( missing );
        }
    }

    std::sort( samples.begin(),
               samples.end(),
               []( const auto &a, const auto &b )
               {
                   return a.m_uuid < b.m_uuid;
               } );

    if( m_gpuStatus == "ok" && std::any_of( samples.begin(),
                                            samples.end(),
                                            []( const auto &gpu )
                                            {
                                                return gpu.m_status != "ok";
                                            } ) )
    {
        m_gpuStatus = "partial";
    }

    if( previousStatus != m_gpuStatus )
    {
        log<text_log>( "GPU monitoring: " + m_gpuStatus,
                       m_gpuStatus == "ok" ? logPrio::LOG_NOTICE : logPrio::LOG_WARNING );
    }

    for( const auto &gpu : samples )
    {
        const auto old = std::find_if( m_gpus.begin(),
                                       m_gpus.end(),
                                       [&]( const auto &value )
                                       {
                                           return value.m_uuid == gpu.m_uuid;
                                       } );

        if( gpu.m_index != telem_gpu::UnknownIndex &&
            ( old == m_gpus.end() || old->m_index != gpu.m_index || old->m_pci != gpu.m_pci ) )
        {
            log<text_log>( "GPU mapping: index=" + std::to_string( gpu.m_index ) + " uuid=" + gpu.m_uuid +
                               " pci=" + gpu.m_pci,
                           logPrio::LOG_NOTICE );
        }

        if( old == m_gpus.end() || old->m_status != gpu.m_status )
        {
            log<text_log>( "GPU " + gpu.m_uuid + ": " + gpu.m_status,
                           gpu.m_status == "ok" ? logPrio::LOG_NOTICE : logPrio::LOG_WARNING );
        }
    }

    m_gpus = std::move( samples );
}

inline int sysMonitor::pollDiagnostics( std::chrono::steady_clock::time_point now )
{
    const auto entered = std::chrono::steady_clock::now();

    if( m_gpuEnabled &&
        ( !m_gpuPolled || std::chrono::duration<double>( now - m_lastGPUPoll ).count() >= m_gpuInterval ) )
    {
        m_gpuPolled         = true;
        m_lastGPUPoll       = now;
        auto         result = queryGPUs();
        const double sampleTime =
            std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count();
        acceptGPUResult( result, sampleTime );
        recordGPUs();
    }

    const auto irqNow = now + ( std::chrono::steady_clock::now() - entered );

    if( !m_irqRules.empty() &&
        ( !m_irqPolled || std::chrono::duration<double>( irqNow - m_lastIRQPoll ).count() >= m_irqInterval ) )
    {
        m_irqPolled   = true;
        m_lastIRQPoll = irqNow;
        const double sampleTime =
            std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count();
        auto groups = queryIRQs( sampleTime );

        for( const auto &group : groups )
        {
            const auto old = std::find_if( m_irqs.begin(),
                                           m_irqs.end(),
                                           [&]( const auto &value )
                                           {
                                               return value.m_name == group.m_name;
                                           } );

            if( old == m_irqs.end() || old->m_status == "unchecked" || old->m_expected != group.m_expected )
            {
                log<text_log>( "IRQ expectation " + group.m_name +
                                   ": expected=" + ( group.m_expected.empty() ? "unavailable" : group.m_expected ),
                               logPrio::LOG_NOTICE );
            }

            if( old == m_irqs.end() || old->m_status != group.m_status || old->m_warning != group.m_warning ||
                ( group.m_warning && !( group == *old ) ) )
            {
                std::string message = "IRQ " + group.m_name + ": " + group.m_status;

                if( group.m_warning )
                {
                    message += "; expected=" + group.m_expected;

                    for( const auto &irq : group.m_irqs )
                    {
                        message += "; IRQ " + std::to_string( irq.m_number ) + " requested=" + irq.m_requested +
                                   " effective=" + irq.m_effective;
                    }
                }

                log<text_log>( message, group.m_warning ? logPrio::LOG_WARNING : logPrio::LOG_NOTICE );
            }
        }

        m_irqs = std::move( groups );
        recordIRQs();
    }

    const auto raidNow = now + ( std::chrono::steady_clock::now() - entered );

    if( m_raidEnabled &&
        ( !m_raidPolled || std::chrono::duration<double>( raidNow - m_lastRAIDPoll ).count() >= m_raidInterval ) )
    {
        m_raidPolled   = true;
        m_lastRAIDPoll = raidNow;
        const double checkTime =
            std::chrono::duration<double>( std::chrono::system_clock::now().time_since_epoch() ).count();
        acceptRAIDResult( queryRAID( checkTime ) );
        recordRAID();
    }

    return publishDiagnostics();
}

inline int sysMonitor::publishDiagnostics()
{
    for( auto &[key, property] : m_gpuProperties )
    {
        const bool assigned = std::any_of( m_gpus.begin(),
                                           m_gpus.end(),
                                           [&]( const auto &gpu )
                                           {
                                               return gpu.m_index != telem_gpu::UnknownIndex &&
                                                      key == "gpu_" + std::to_string( gpu.m_index );
                                           } );

        if( !assigned )
        {
            updateIfChanged( property, "valid", 0.0, INDI_ALERT );
            updateIfChanged( m_gpuInfoProperties.at( key ), "status", std::string( "unavailable" ), INDI_ALERT );
        }
    }

    for( const auto &gpu : m_gpus )
    {
        if( gpu.m_index == telem_gpu::UnknownIndex )
        {
            continue;
        }

        const auto key = "gpu_" + std::to_string( gpu.m_index );

        if( m_gpuProperties.count( key ) == 0 )
        {
            auto &number = m_gpuProperties[key];
            number       = pcf::IndiProperty( pcf::IndiProperty::Number );
            number.setDevice( m_configName );
            number.setName( key );
            number.setPerm( pcf::IndiProperty::ReadOnly );
            number.setState( INDI_IDLE );

            for( const auto &name : { "temperature",
                                      "load",
                                      "memory_load",
                                      "memory_used",
                                      "memory_total",
                                      "clock_graphics",
                                      "clock_sm",
                                      "clock_memory",
                                      "valid",
                                      "sample_time" } )
            {
                number.add( pcf::IndiElement( name, 0.0 ) );
            }

            auto &info = m_gpuInfoProperties[key];
            info       = pcf::IndiProperty( pcf::IndiProperty::Text );
            info.setDevice( m_configName );
            info.setName( key + "_info" );
            info.setPerm( pcf::IndiProperty::ReadOnly );
            info.setState( INDI_IDLE );
            info.add( pcf::IndiElement( "uuid", gpu.m_uuid ) );
            info.add( pcf::IndiElement( "pci", gpu.m_pci ) );
            info.add( pcf::IndiElement( "status", gpu.m_status ) );

            if( registerIndiPropertyNew( number, nullptr ) < 0 || registerIndiPropertyNew( info, nullptr ) < 0 )
            {
                return -1;
            }

            if( m_indiDriver )
            {
                m_indiDriver->sendDefProperty( number );
                m_indiDriver->sendDefProperty( info );
            }
        }

        const auto state = !m_gpuEnabled ? INDI_IDLE : ( gpu.m_status == "ok" ? INDI_OK : INDI_ALERT );
        updateIfChanged<double>( m_gpuProperties.at( key ),
                                 { "temperature",
                                   "load",
                                   "memory_load",
                                   "memory_used",
                                   "memory_total",
                                   "clock_graphics",
                                   "clock_sm",
                                   "clock_memory",
                                   "valid",
                                   "sample_time" },
                                 { gpu.m_temperature,
                                   gpu.m_load,
                                   gpu.m_memoryLoad,
                                   gpu.m_memoryUsed,
                                   gpu.m_memoryTotal,
                                   gpu.m_clockGraphics,
                                   gpu.m_clockSM,
                                   gpu.m_clockMemory,
                                   double( gpu.m_valid ),
                                   gpu.m_sampleTime },
                                 state );
        updateIfChanged<std::string>( m_gpuInfoProperties.at( key ),
                                      { "uuid", "pci", "status" },
                                      { gpu.m_uuid, gpu.m_pci, gpu.m_status },
                                      state );
    }

    updateIfChanged( m_indiP_gpuStatus,
                     "status",
                     m_gpuStatus,
                     !m_gpuEnabled ? INDI_IDLE : ( m_gpuStatus == "ok" ? INDI_OK : INDI_ALERT ) );
    const bool warning = std::any_of( m_irqs.begin(),
                                      m_irqs.end(),
                                      []( const auto &group )
                                      {
                                          return group.m_warning;
                                      } );
    const auto state =
        m_irqRules.empty() ? INDI_IDLE : ( warning ? INDI_ALERT : ( m_irqPolled ? INDI_OK : INDI_IDLE ) );
    std::vector<std::string> names{ "summary" },
        statuses{ m_irqRules.empty() ? "disabled" : ( !m_irqPolled ? "unchecked" : ( warning ? "warning" : "ok" ) ) };

    for( const auto &group : m_irqs )
    {
        names.push_back( group.m_name );
        statuses.push_back( group.m_status );
    }

    updateIfChanged( m_indiP_irqStatus, names, statuses, state );

    updateSwitchIfChanged(
        m_indiP_irqWarning, "active", warning ? pcf::IndiElement::On : pcf::IndiElement::Off, state );
    return publishRAID();
}

inline int sysMonitor::recordGPUs( bool force )
{
    if( m_gpuEnabled && ( !m_gpuRecorded || force || m_gpuStatus != m_lastGPUStatus || m_gpus != m_lastGPUs ) )
    {
        std::vector<telem_gpu::gpuT> indexed;

        for( const auto &gpu : m_gpus )
        {
            if( gpu.m_index != telem_gpu::UnknownIndex )
            {
                indexed.push_back( gpu );
            }
        }

        telem<telem_gpu>( { m_gpuStatus, indexed } );
        m_lastGPUStatus = m_gpuStatus;
        m_lastGPUs      = m_gpus;
        m_gpuRecorded   = true;
    }

    return 0;
}

inline int sysMonitor::recordIRQs( bool force )
{
    std::vector<uint8_t> ok;

    for( const auto &rule : m_irqRules )
    {
        const auto group = std::find_if( m_irqs.begin(),
                                         m_irqs.end(),
                                         [&]( const auto &value )
                                         {
                                             return value.m_name == rule.m_name;
                                         } );
        ok.push_back( group != m_irqs.end() && group->m_status != "unchecked" && !group->m_warning );
    }

    if( !m_irqRules.empty() && ( !m_irqRecorded || force || ok != m_lastIRQOK ) )
    {
        telem<telem_irq>( { ok, m_irqs.empty() ? 0 : m_irqs.front().m_checkTime } );
        m_lastIRQOK   = std::move( ok );
        m_irqRecorded = true;
    }

    return 0;
}

inline int sysMonitor::recordTelem( const telem_gpu * )
{
    return recordGPUs( true );
}

inline int sysMonitor::recordTelem( const telem_irq * )
{
    return recordIRQs( true );
}

inline int sysMonitor::startupRAID()
{
    m_raidStatus  = m_raidEnabled ? telem_raid::Unchecked : telem_raid::Disabled;
    m_raidMessage = m_raidEnabled ? "unchecked" : "disabled";
    REG_INDI_NEWPROP_NOCB( m_indiP_raidStatus, "raid_status", pcf::IndiProperty::Text );
    m_indiP_raidStatus.add( pcf::IndiElement( "summary", m_raidMessage ) );

    if( registerIndiPropertyNew( m_indiP_raidWarning,
                                 "raid_warning",
                                 pcf::IndiProperty::Switch,
                                 pcf::IndiProperty::ReadOnly,
                                 INDI_IDLE,
                                 pcf::IndiProperty::AnyOfMany,
                                 nullptr ) < 0 )
    {
        return -1;
    }

    m_indiP_raidWarning.add( pcf::IndiElement( "active", pcf::IndiElement::Off ) );
    return 0;
}

inline sysMonitorDetail::raidSnapshotT sysMonitor::queryRAID( double checkTime )
{
    return sysMonitorDetail::checkRAID( "/proc/mdstat", checkTime );
}

inline bool sysMonitor::raidWarning() const
{
    if( !m_raidEnabled )
    {
        return false;
    }

    return m_raidStatus == telem_raid::ReadError || m_raidStatus == telem_raid::ParseError ||
           std::any_of( m_raids.begin(),
                        m_raids.end(),
                        []( const auto &array )
                        {
                            return !( array.m_sample.m_flags & telem_raid::OK );
                        } );
}

inline void sysMonitor::acceptRAIDResult( const sysMonitorDetail::raidSnapshotT &result )
{
    std::vector<sysMonitorDetail::raidArrayT> arrays;

    if( result.m_result == telem_raid::Ready )
    {
        arrays             = result.m_arrays;
        uint32_t nextIndex = m_raids.size();

        for( auto &array : arrays )
        {
            const auto old         = std::find_if( m_raids.begin(),
                                           m_raids.end(),
                                           [&]( const auto &value )
                                           {
                                               return value.m_name == array.m_name;
                                           } );
            array.m_sample.m_index = old == m_raids.end() ? nextIndex++ : old->m_sample.m_index;

            if( old == m_raids.end() || old->m_level != array.m_level )
            {
                log<text_log>( "RAID mapping: index=" + std::to_string( array.m_sample.m_index ) +
                                   " name=" + array.m_name + " level=" + array.m_level,
                               logPrio::LOG_NOTICE );
            }
        }
    }

    for( const auto &old : m_raids )
    {
        if( std::none_of( arrays.begin(),
                          arrays.end(),
                          [&]( const auto &value )
                          {
                              return value.m_name == old.m_name;
                          } ) )
        {
            auto       missing = old;
            const bool absent  = result.m_result == telem_raid::Ready || result.m_result == telem_raid::NotPresent;
            missing.m_status   = absent ? "missing" : "unavailable";
            missing.m_members.clear();
            missing.m_sample = { old.m_sample.m_index, absent ? telem_raid::Missing : 0U, 0, 0 };
            arrays.push_back( std::move( missing ) );
        }
    }

    std::sort( arrays.begin(),
               arrays.end(),
               []( const auto &a, const auto &b )
               {
                   return a.m_sample.m_index < b.m_sample.m_index;
               } );

    for( const auto &array : arrays )
    {
        const auto old     = std::find_if( m_raids.begin(),
                                       m_raids.end(),
                                       [&]( const auto &value )
                                       {
                                           return value.m_name == array.m_name;
                                       } );
        const bool warning = !( array.m_sample.m_flags & telem_raid::OK );

        if( old == m_raids.end() || old->m_status != array.m_status || !( old->m_sample == array.m_sample ) ||
            ( warning && old->m_members != array.m_members ) )
        {
            std::string message = "RAID " + array.m_name + ": " + array.m_status;

            if( warning )
            {
                message += "; level=" + array.m_level + " active=" + std::to_string( array.m_sample.m_active ) + "/" +
                           std::to_string( array.m_sample.m_expected ) + " members=" + array.m_members;
            }

            log<text_log>( message, warning ? logPrio::LOG_WARNING : logPrio::LOG_NOTICE );
        }
    }

    const auto oldStatus  = m_raidStatus;
    const auto oldMessage = m_raidMessage;
    m_raids               = std::move( arrays );
    m_raidStatus          = result.m_result;
    m_raidCheckTime       = result.m_checkTime;
    m_raidMessage         = result.m_result == telem_raid::Ready
                                ? ( m_raids.empty() ? "no arrays" : ( raidWarning() ? "warning" : "ok" ) )
                                : result.m_status;

    if( oldStatus != m_raidStatus || oldMessage != m_raidMessage )
    {
        log<text_log>( "RAID monitoring: " + m_raidMessage,
                       raidWarning() ? logPrio::LOG_WARNING : logPrio::LOG_NOTICE );
    }
}

inline int sysMonitor::publishRAID()
{
    for( const auto &array : m_raids )
    {
        const auto key = "raid_" + std::to_string( array.m_sample.m_index );

        if( m_raidProperties.count( key ) == 0 )
        {
            auto &number = m_raidProperties[key];
            number       = pcf::IndiProperty( pcf::IndiProperty::Number );
            number.setDevice( m_configName );
            number.setName( key );
            number.setPerm( pcf::IndiProperty::ReadOnly );
            number.setState( INDI_IDLE );

            for( const auto &name : { "valid", "ok", "degraded", "expected", "active", "flags", "check_time" } )
            {
                number.add( pcf::IndiElement( name, 0.0 ) );
            }

            auto &info = m_raidInfoProperties[key];
            info       = pcf::IndiProperty( pcf::IndiProperty::Text );
            info.setDevice( m_configName );
            info.setName( key + "_info" );
            info.setPerm( pcf::IndiProperty::ReadOnly );
            info.setState( INDI_IDLE );
            info.add( pcf::IndiElement( "name", array.m_name ) );
            info.add( pcf::IndiElement( "level", array.m_level ) );
            info.add( pcf::IndiElement( "status", array.m_status ) );

            if( registerIndiPropertyNew( number, nullptr ) < 0 || registerIndiPropertyNew( info, nullptr ) < 0 )
            {
                return -1;
            }

            if( m_indiDriver )
            {
                m_indiDriver->sendDefProperty( number );
                m_indiDriver->sendDefProperty( info );
            }
        }

        const auto flags = array.m_sample.m_flags;
        const auto state = !m_raidEnabled ? INDI_IDLE : ( flags & telem_raid::OK ? INDI_OK : INDI_ALERT );
        updateIfChanged<double>( m_raidProperties.at( key ),
                                 { "valid", "ok", "degraded", "expected", "active", "flags", "check_time" },
                                 { double( bool( flags & telem_raid::Valid ) ),
                                   double( bool( flags & telem_raid::OK ) ),
                                   double( bool( flags & telem_raid::Degraded ) ),
                                   double( array.m_sample.m_expected ),
                                   double( array.m_sample.m_active ),
                                   double( flags ),
                                   m_raidCheckTime },
                                 state );
        updateIfChanged<std::string>( m_raidInfoProperties.at( key ),
                                      { "name", "level", "status" },
                                      { array.m_name, array.m_level, array.m_status },
                                      state );
    }

    const bool warning = raidWarning();
    const auto state   = !m_raidEnabled || m_raidStatus == telem_raid::Unchecked
                             ? INDI_IDLE
                             : ( warning ? INDI_ALERT : ( m_raids.empty() ? INDI_IDLE : INDI_OK ) );
    updateIfChanged( m_indiP_raidStatus, "summary", m_raidMessage, state );
    updateSwitchIfChanged(
        m_indiP_raidWarning, "active", warning ? pcf::IndiElement::On : pcf::IndiElement::Off, state );
    return 0;
}

inline int sysMonitor::recordRAID( bool force )
{
    std::vector<telem_raid::arrayT> arrays;

    for( const auto &array : m_raids )
    {
        arrays.push_back( array.m_sample );
    }

    if( m_raidEnabled && ( !m_raidRecorded || force || m_raidStatus != m_lastRAIDStatus || arrays != m_lastRAIDs ) )
    {
        telem<telem_raid>( { m_raidStatus, arrays, m_raidCheckTime } );
        m_lastRAIDs      = std::move( arrays );
        m_lastRAIDStatus = m_raidStatus;
        m_raidRecorded   = true;
    }

    return 0;
}

inline int sysMonitor::recordTelem( const telem_raid * )
{
    return recordRAID( true );
}

} //namespace app
} //namespace MagAOX

#endif //sysMonitor_hpp
