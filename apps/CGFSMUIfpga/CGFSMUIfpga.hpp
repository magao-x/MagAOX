/** \file CGFSMUIfpga.hpp
  * \brief The MagAO-X CGFSMUIfpga (CGraph FSM (test) User I/F) header file
  *
  * \ingroup CGFSMUIfpga_files
  */

#ifndef CGFSMUIfpga_hpp
#define CGFSMUIfpga_hpp

#include <cmath>
#include <cfenv>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup CGFSMUIfpga
  * \brief The CGFSMUIfpga proof-of-concept FPGA UI application
  * \details INDI-driven User Interface (UI) to drive the CGFSMHIfpga
  *          (FPGA Hardware Interface) MagAO-X app by writing to the
  *          DAC's floating-point members in a shmim, which values may
  *          be converted to 24-bit integers and those integers written
  *          to the FPGA device's memory map by the CGFSMHIfpga app
  *          (HI => Hardware Inteface).
  *          This app was originally intended as a proof of concept to
  *          validate the FPGA HI software and not for typical
  *          operational use; it may be used as a test driver.
  *          Eventually an app to automatically and continuously control
  *          the FSM will be written to drive the FPGA based on
  *          wavefront measurements.
  *          Acronyms in this app's name and elsewhere:
  *            CG   - Coronagraph
  *            FSM  - Fast Steerable Mirror
  *            UI   - User Interface
  *            fpga - Field-Programmable Gate Array
  *            HI   - Hardware Interface
  *
  * <a href="../handbook/operating/software/apps/CGFSMUIfpga.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup CGFSMUIfpga_files
  * \ingroup CGFSMUIfpga
  */

namespace MagAOX
{
namespace app
{
////////////////////////////////////////////////////////////////////////
/// The MagAO-X CGFSMUIfpga
/** \brief Front-end User Interface control of a Fast Steerable Mirror
  *
  * \details
  * Input three floating-point distance setpoints, in microns, via INDI;
  * write INDI values to a low-latency shmim (SHared Memory IMage).
  *
  * Downstream, the back-end hardware interface app uses the shmim as
  * input to control an FPGA:
  * - The FPGA drives three Digital-to-Analog Converters (DACs; Dac).
  * - The DACs drive three piezoelectric devices attached to a FSM.
  * - The piezoelectric devices control the FSM's Tip, Tilt, and Piston.
  *
  * \ingroup CGFSMUIfpga
  */
class CGFSMUIfpga : public MagAOXApp<true>, public dev::frameGrabber<CGFSMUIfpga>
{
   //Give the test harness access.
   friend class CGFSMUIfpga_test;

   // Use the base frameGrabber type to write new setpoints as available
   friend class dev::frameGrabber<CGFSMUIfpga>;
   typedef dev::frameGrabber<CGFSMUIfpga> frameGrabberT;

public:

   //Enumeration of the source of the micron (um) input values
   enum class SPSource
   {indi         // Manual control via INDI, for testing
   ,shm          // Automated control via the shmim interface (~250Hz)
   ,disabled     // No control, operation is disabled
   ,invalid      // Non-functional placeholder
   };

protected:

   /** \name Configurable Parameters
     *@{
     */

   // Parameters which will be configurable at runtime via config file
   // (None)

   ///@}

   /// INDI interface definitions

   // 1) The setpoints, in engineering units (micron; um); range [0-10]
   pcf::IndiProperty m_indiP_Dac_um;
   // <device>.Dac_um.A, number, floating point
   // <device>.Dac_um.B, number, floating point
   // <device>.Dac_um.C, number, floating point
   // Local members to store data
   double m_DacA_um{0.0};                      // DAC micron (um) positions
   double m_DacB_um{0.0};
   double m_DacC_um{0.0};

   // 2) INDI control of logging
   // <device>.logging_control.period_base2_logarithm, unsigned integer
   pcf::IndiProperty m_indiP_logging_control;
   // Local members to store data
   unsigned int m_log_period_base2log{0};      // Steps of ->md->cnt0 between INDI updates

   // 3) INDI log data (read-only)
   // <device>.logging.counts, number, integer
   // <device>.logging.updates, number, integer
   pcf::IndiProperty m_indiP_logs;
   // Local members to store data
   unsigned int m_counts_in_period{0};         // INDI read-only
   unsigned int m_updates_in_period{0};        // INDI read-only

   /// Class data members

   double m_period{0.0};                // 1<<m_log_period_base2log
   uint64_t m_period_mask{0};           // m_period - 1
   unsigned int m_local_cnt2{0};        // Count of calls to acquireAndCheckValid
   typedef struct timespec NANOTIME;
   float m_fps{250.0};           // frameGrabber frames per second
   NANOTIME m_nano{0,0};     // For pacing
   long m_delta_nano{0};        // For pacing

   /// Constants

   // Constant used by frameGrabber
   static constexpr bool c_frameGrabber_flippable{false};        ///< frameGrabber boolean
   // "Image" dimensions and data type
   static constexpr uint32_t m_shmimWidth {3};                   ///< The width of the "image"
   static constexpr uint32_t m_shmimHeight {1};                  ///< The height of the "image"
   static constexpr uint8_t m_shmimDataType{IMAGESTRUCT_DOUBLE}; ///< The ImageStreamIO type code.

public:
   /// Default c'tor.
   CGFSMUIfpga();

   /// D'tor, declared and defined for noexcept.
   ~CGFSMUIfpga() noexcept
   {}

   /// Setup the configuration system (called by MagAOXApp::setup())
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   /// Load the configuration system results (called by MagAOXApp::setup())
   /** Wrapper for loadConfigImpl
     */
   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for CGFSMUIfpga.
   /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.  Currently nothing in this app
   virtual int appShutdown();

   /// frameGrabber thread callbacks
   /** -     allocate - called during startup/setup to assign pointers
     * - processImage - called when semaphore is triggered
     */

   /// INDI callbacks:
   /// - DAC um data; written to frameGrabber shmim
   /// - Logging control and logged data
   INDI_NEWCALLBACK_DECL(CGFSMUIfpga, m_indiP_Dac_um);
   INDI_NEWCALLBACK_DECL(CGFSMUIfpga, m_indiP_logging_control);
   INDI_NEWCALLBACK_DECL(CGFSMUIfpga, m_indiP_logs);

   int configureAcquisition();
   int startAcquisition();
   int acquireAndCheckValid();
   int loadImageIntoStream(void * dest);
   int reconfig();
   float fps()
   {
      return m_fps;
   }

};
// class CGFSMUIfpga ...
////////////////////////////////////////////////////////////////////////

CGFSMUIfpga::CGFSMUIfpga() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

/// Setup to read configuration data from TOML file
void CGFSMUIfpga::setupConfig()
{
   // Setup frameGrabber, which will set up three TOML config parameters
   // N.B. stanza name is [framegrabber], not [frameGrabber]
   // - framegrabber.threadPrio
   // - framegrabber.cpuset
   // - framegrabber.shmimName
   // - framegrabber.circBuffLength
   // So in the TOML config file, the shmim config might be like this:
   //
   //     [framegrabber]
   //     threadPrio = N
   //     cpuset = M
   //     shmimName = fpga0
   //     circBuffLength = L
   //
   frameGrabberT::setupConfig(config);
}

// Load configuration from TOML file, using setup from setupConfig above
int CGFSMUIfpga::loadConfigImpl( mx::app::appConfigurator & _config )
{
   frameGrabberT::loadConfig(_config);
   return 0;
}

// Override of virtural loadConfig() is a wrapper for loadConfigImpl()
void CGFSMUIfpga::loadConfig()
{
   loadConfigImpl(config);
}

int CGFSMUIfpga::appStartup()
{
   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   // INDI interface to the micron (um) user input values
   REG_INDI_NEWPROP(m_indiP_Dac_um, "Dac_um", pcf::IndiProperty::Number);
   indi::addNumberElement<double>( m_indiP_Dac_um, "A",  0.0,  10.0, 1.0e-5,  "%f", "");
   indi::addNumberElement<double>( m_indiP_Dac_um, "B",  0.0,  10.0, 1.0e-5,  "%f", "");
   indi::addNumberElement<double>( m_indiP_Dac_um, "C",  0.0,  10.0, 1.0e-6,  "%f", "");
   m_indiP_Dac_um["A"].set<double>(0.0);
   m_indiP_Dac_um["B"].set<double>(0.0);
   m_indiP_Dac_um["C"].set<double>(0.0);

   REG_INDI_NEWPROP(m_indiP_logging_control, "logging_control", pcf::IndiProperty::Number);
   indi::addNumberElement<unsigned int>( m_indiP_logging_control, "period_base2log",  0,  16, 1,  "%d", "");

   REG_INDI_NEWPROP_NOCB(m_indiP_logs, "logs", pcf::IndiProperty::Number);
   indi::addNumberElement<unsigned int>( m_indiP_logs, "counts",  0,  16, 1,  "%d", "");
   indi::addNumberElement<unsigned int>( m_indiP_logs, "updates",  0,  16, 1,  "%d", "");
   state(stateCodes::READY);

   return 0;
}

int CGFSMUIfpga::appLogic()
{
   if( frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   return 0;
}

int CGFSMUIfpga::appShutdown()
{
   frameGrabberT::appShutdown();
   return 0;
}

////////////////////////////////////////////////////////////////////////
// Interfaces called from frameGrabber thread routine fgThreadExec

/// \brief Configure frameGrabber shmim dimensions and other properties
int CGFSMUIfpga::configureAcquisition()
{
   frameGrabberT::m_width = m_shmimWidth;
   frameGrabberT::m_height = m_shmimHeight;
   frameGrabberT::m_dataType = m_shmimDataType;
   frameGrabberT::m_defaultFlip = frameGrabberT::fgFlipNone;
   // N.B. frameGrabberT::m_circBuffLength is set via TOML config file
   //      configuration parameter frameGrabber.circBuffLength
   // Leave the following at default or TOML-based values
   // - m_latencyCircBuffMaxTime
   // - m_latencyCircBuffMaxLength
   return 0;
}

/// \brief One-time data initialization of source at start of acquisition
int CGFSMUIfpga::startAcquisition()
{
   // Nothing to do here
   return 0;
}

/// \brief Update class member from INDI propertyelement value
/** \returns true if the class member is updated
  * \returns false if the class member is not updated (no change)
  */
template<typename T>
inline bool update_member(T& member_ref
                         ,const pcf::IndiProperty& indiP_elems
                         ,const std::string elemname
                         )
{
   // Check if element name is in the INDI property instance
   if (indiP_elems.find(elemname))
   {
      // Get the INDI value of the element
      T indi_val =  indiP_elems[elemname].get<T>();
      // If the INDI value is different than the current class value ...
      if (indi_val != member_ref)
      {
         // ... then update the class value and return [true]
         member_ref = indi_val;
         return true;
      }
   }
   return false;
}

/// \brief Acquire the DAC um values from the class' INDI property
/// \returns 0 if any of the values are new
/// \returns 1 if all of the values have not changed
/// \details This is called by the frameGrabber, probably based on the
/// fps() value and time.  See also this->update_indi_element()
int CGFSMUIfpga::acquireAndCheckValid()
{
   if (!m_nano.tv_sec)
   {
      if (clock_gettime(CLOCK_REALTIME,&m_nano)<0)
      {
          return -1;
      };
      m_delta_nano = (long) ((1e9 / ((double)fps())) + 0.5);
   }
   m_nano.tv_nsec += m_delta_nano;
   if (m_nano.tv_nsec > 1000000000)
   {
      m_nano.tv_nsec -= 1000000000;
      m_nano.tv_sec += 1;
   }
   clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &m_nano, (NANOTIME*)0);
   errno = 0;

   ++m_counts_in_period;
   if (m_log_period_base2log>0 && !(m_local_cnt2 & m_period_mask))
   {
      m_indiP_logs["counts"] = m_counts_in_period;
      m_indiP_logs.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_logs);
      m_counts_in_period = 0;
   }
   
   bool b{false};
   b |= update_member(m_DacA_um, m_indiP_Dac_um, "A");
   b |= update_member(m_DacB_um, m_indiP_Dac_um, "B");
   b |= update_member(m_DacC_um, m_indiP_Dac_um, "C");
   if (!b) { ++m_local_cnt2; }
   return b ? 0 : 1;
}

/// \brief Move class um values into downstream imageStream (shmim)
/// \details This is called by the frameGrabber when the routine above,
/// acquireAndCheckValid(...), returns [true]
int CGFSMUIfpga::loadImageIntoStream(void * dest)
{
   ++m_updates_in_period;
   if (m_log_period_base2log>0 && !(m_local_cnt2 & m_period_mask))
   {
      m_indiP_logs["updates"] = m_updates_in_period;
      m_indiP_logs.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_logs);
      m_updates_in_period = 0;
   }
   
   // dest is a pointer to three contiguous doubles in a shmim that
   // correspond to Tip, Tilt, and Piston of the FSM, in micron (um).
   // A downstream process, CGFSMHIfpga, reads these values from the
   // shmim and writes then to the FPGA's mapped memory.
   double* pDac_um = (double*) dest;
   *(pDac_um++) = m_DacA_um;
   *(pDac_um++) = m_DacB_um;
   *pDac_um     = m_DacC_um;
   return 0;
}

int CGFSMUIfpga::reconfig()
{
   return 0;
}

/// \brief Copy received INDI property value to class' INDI property
template<typename T>
inline bool update_indi_element(const pcf::IndiProperty& ipRecv
                                        ,pcf::IndiProperty& indiP_elems
                                        ,const std::string elemname
                                        , T lolim = 1
                                        , T hilim = 0
                                        )
{
   // Proceed IFF element name is in both INDI property instances
   if (ipRecv.find(elemname) && indiP_elems.find(elemname))
   {
      // Get element values from the received and class INDI props, ...
      T new_val =  ipRecv[elemname].get<T>();
      T indi_val =  indiP_elems[elemname].get<T>();
      // ... update the class INDI element if the received differs, ...
      if (indi_val != new_val
          && (lolim > hilim || ( lolim <= new_val && new_val <= hilim))
         )
      {
         indiP_elems[elemname] = new_val;
         // ... and return [true] to indicate the change
         return true;
      }
   }
   // Return [false] for either no change, or this elemname not in
   // received INDI property, or elemname is not in class' INDI property
   return false;
}

/// \brief INDI callback for one or more INDI-input DAC um values
/// \returns 0 for success
/// \returns -1 for error i.e. ipRecv has different property name*
/// \details This is the heart of the INDI side of this application.
/// User-supplied values are stored in INDI properties i.e. the class
/// member INDI property** is used to store the micron values entered by
/// the user.  When values come in via the INDI protocol, it triggers a
/// call of this callback, at which point this routine copies those
/// values from the received IndiProperty (ipRecv) to the class's member
/// IndiPropery (m_indiP_Dac_um).
/// * than class-local INDI property [m_indiP_Dac_um], in which case, we
///   should wonder why this callback was called
/// ** in this case, the INDI property is this->m_indiP_Dac_um, with
///    element names A, B, and C, corresponding to the three DACs
INDI_NEWCALLBACK_DEFN(CGFSMUIfpga, m_indiP_Dac_um)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_Dac_um.getName())
   {
      // Received a new value for a micron (um) setpoint from INDI
      // Assign true to boolean b IFF a value has changed
      bool b{false};
      b |= update_indi_element<double>(ipRecv, m_indiP_Dac_um, "A");
      b |= update_indi_element<double>(ipRecv, m_indiP_Dac_um, "B");
      b |= update_indi_element<double>(ipRecv, m_indiP_Dac_um, "C");
      if (b)
      {
         // Update INDI state if any of the elements' values changed
         m_indiP_Dac_um.setState (pcf::IndiProperty::Ok);
         m_indiDriver->sendSetProperty (m_indiP_Dac_um);
      }
      return 0;
   }
   return -1;
}

/// \brief INDI callback for logging period, base-2 log unsigned integer
/// \returns 0 for success
/// \returns -1 for error i.e. ipRecv has different property name
/// \details This is the callback for the INDI logging property input;
/// the only element is period_base2log.  A value of 0 turns off logging
/// and a positive value from 1 to 16 sets the logging period at 2 to
/// 65536 frames; a value above 16 is ignored
INDI_NEWCALLBACK_DEFN(CGFSMUIfpga, m_indiP_logging_control)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getName() == m_indiP_logging_control.getName())
   {
      // Received a new value for the period of logging data
      // Returns true to boolean b IFF INDI value has changed, AND
      // is in the correct range [0:16]
      if (update_indi_element<unsigned int>(ipRecv, m_indiP_logging_control
                                                , "period_base2log"
                                                , 0, 16
                                                ))
      {
         // INDI property value has changed, update the class member
         if (update_member(m_log_period_base2log, m_indiP_logging_control, "period_base2log"))
         {
            // If m_log_period_base2log is non-zero, calculate the
            // period and the mask to be applied to
            // m_imageStream->md->cnt0
            if (m_log_period_base2log > 0)
            {
               m_period = m_period_mask = (1 << m_log_period_base2log);
               --m_period_mask;
            }
         }
      }
      m_indiP_Dac_um.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_Dac_um);
      return 0;
   }
   return -1;
}

} //namespace app
} //namespace MagAOX

#endif //CGFSMUIfpga_hpp
