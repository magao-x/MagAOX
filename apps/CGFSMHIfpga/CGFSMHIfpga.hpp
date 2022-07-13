/** \file CGFSMHIfpga.hpp
  * \brief The MagAO-X CGFSMHIfpga (CGraph FSM Hardware I/f) header file
  *
  * \ingroup CGFSMHIfpga_files
  */

#ifndef CGFSMHIfpga_hpp
#define CGFSMHIfpga_hpp

#include <cmath>
#include <cfenv>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"
#include "CGraphFSMHardwareInterface.cpp"

/** \defgroup CGFSMHIfpga
  * \brief The CGFSMHIfpga application to interface to the FPGA device driver
  *
  * <a href="../handbook/operating/software/apps/CGFSMHIfpga.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup CGFSMHIfpga_files
  * \ingroup CGFSMHIfpga
  */

namespace MagAOX
{
namespace app
{

/// Linear scaling class
/** Floating point input, integer output
  */
class DtoIscaling
{
protected:
    // Input limits
    double m_um_lo{0.0};
    double m_um_hi{10.0};
    // Output limits
    uint32_t m_int_lo{0};
    uint32_t m_int_hi{(1<<20) - 1};
    // Linear model parameters
    double m_slope{104857.5};
    double m_offset{0.0};
public:
    // C'tors
    DtoIscaling();
    DtoIscaling(const double& um_lo, const double& um_hi, const uint32_t& int_lo, const uint32_t& int_hi);
    // Scale a value
    uint32_t scale(double& um_input);
    // Assign limits' values:  from config file at runtime; from INDI
    void limits_from_config(mx::app::appConfigurator& _config, const std::string& pfx);
    void setLimits(const double& um_lo, const double& um_hi, const uint32_t& int_lo, const uint32_t& int_hi);
    // Read limit and model properties
    const double& um_lo() const { return m_um_lo; }
    const double& um_hi() const { return m_um_hi; }
    const uint32_t& int_lo() const { return m_int_lo; }
    const uint32_t& int_hi() const { return m_int_hi; }
    const double& slope() const { return m_slope; }
    const double& offset() const { return m_offset; }
private:
    // Convert limits to slope and offset
    void setSlopeOffset();
};
// Two C'tors:  first uses default limits; second uses input limits
DtoIscaling::DtoIscaling() { setSlopeOffset(); }
DtoIscaling::DtoIscaling(const double& um_lo, const double& um_hi, const uint32_t& int_lo, const uint32_t& int_hi)
{
    setLimits(um_lo, um_hi, int_lo, int_hi);
}

// Modify limits via direct call
void DtoIscaling::setLimits(const double& um_lo, const double& um_hi, const uint32_t& int_lo, const uint32_t& int_hi)
{
    m_um_lo = um_lo;
    m_um_hi = um_hi;
    m_int_lo = int_lo;
    m_int_hi = int_hi;
    setSlopeOffset();
}

// Modify limits via appConfigurator (TOML file)
/* pfx is typically "scaling." */
void DtoIscaling::limits_from_config(mx::app::appConfigurator& _config, const std::string& pfx)
{
    if (_config.isSet(pfx + "um_lo")) { _config(m_um_lo, pfx + "um_lo"); };
    if (_config.isSet(pfx + "um_hi")) { _config(m_um_hi, pfx + "um_hi"); };
    if (_config.isSet(pfx + "int_lo")) { _config(m_int_lo, pfx + "int_lo"); };
    if (_config.isSet(pfx + "int_hi")) { _config(m_int_hi, pfx + "int_hi"); };
    setSlopeOffset();
}

// Convert limits to linear model parameters; protect against divby0
void DtoIscaling::setSlopeOffset()
{
    m_offset = m_int_lo;
    m_slope = ((double)(m_int_hi - m_int_lo)) / (m_um_hi - m_um_lo);
    if (!finite(m_slope)) { m_slope = 0.0; }
}

// Scale an input value
uint32_t DtoIscaling::scale(double& um_input)
{
    if (um_input <= m_um_lo) { return m_int_lo; }
    if (um_input >= m_um_hi) { return m_int_hi; }
    return (uint32_t) round((m_slope * (um_input - m_um_lo)) + m_offset);
}
// End of DtoIscaling
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
/// The MagAO-X CGFSMHIfpga
/** Control Tip, Tilt, and Piston of a Fast Steerable Mirror (FSM)
  *
  * Input three floating-point distance values, in microns (um)
  * Linearly scale the inputs to integers (typically a 20-bit range)
  * Output the integer values to a FPGA Linux device driver's memory map
  *
  * The FPGA drives three Digital-to-Analog Converters (DACs; Dac) and
  * piezoelectric devices attached to a Fast Steerable Mirror (FSM) to
  * control the FSM's Tip, Tilt, and Piston.
  *
  * \ingroup CGFSMHIfpga
  */
class CGFSMHIfpga : public MagAOXApp<true>, public dev::shmimMonitor<CGFSMHIfpga>
{
   //Give the test harness access.
   friend class CGFSMHIfpga_test;

   // Use the base shmimMonitor type to detect when new setpoints are available
   // - defaults to dev::shmimMonitor<CGFSMHIfpga,shmimT>
   //   - That shmimT means
   //     - The configSection() (prefix) will be "shmimMonitor"
   //     - The indiPrefix() will be "sm"
   // - Data for shmimMonitorT is in m_shmimStream, comprising
   //   - shared memory and meta-data for that shared memory
   //   - Semaphores
   friend class dev::shmimMonitor<CGFSMHIfpga>;
   typedef dev::shmimMonitor<CGFSMHIfpga> shmimMonitorT;

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
   // -       m_linux_dev - FPGA device name e.g. /dev/fpga0
   // - m_setpoint_source - Active source of micron input values
   // -     m_Dac_scaling - Setpoint scaling from micron to 24-bit int
   std::string m_linux_dev{""};
   SPSource m_setpoint_source{SPSource::shm};
   DtoIscaling m_Dac_scaling{DtoIscaling()};

   ///@}

   /// INDI interface definitions

   // 1) Source of the um setpoint; m_setpoint_source above; will be
   // 1.1) EITHER [SPSource::indi] for testing and/or manual control,
   // 1.2)     OR [SPSource::shm] for control from Tip/Tilt controller
   // 1.2)     OR [SPSource::disabled] to not scale um setpoint to FSM
   pcf::IndiProperty m_indiP_setpoint_source;
   // <device>.setpoint_source.indi, switch
   // <device>.setpoint_source.shm, switch
   // <device>.setpoint_source.disabled, switch

   // 2) The setpoints, in engineering units (micron; um); range [0-10]
   // 2.1) These will only be of interest when setpoint source is ::shm
   pcf::IndiProperty m_indiP_Dac_um;
   // <device>.Dac_10um.A, number, floating point, 0-10um
   // <device>.Dac_10um.B, number, floating point, 0-10um
   // <device>.Dac_10um.C, number, floating point, 0-10um

   // 3) The setpoints, in instrument units, uint32_t; range [0-1M)
   // 3.1) These will not take NEW values from INDI, but will notify
   //      INDI when the value change
   // 3.2) These will only be of interest when setpoint source is ::shm
   pcf::IndiProperty m_indiP_Dac_setpoint;
   // <device>.Dac_SP.A, number, uint32_t, 0-((2**20)-1), read-only
   // <device>.Dac_SP.B, number, uint32_t, 0-((2**20)-1), read-only
   // <device>.Dac_SP.C, number, uint32_t, 0-((2**20)-1), read-only

   // 4) The scaling
   pcf::IndiProperty m_indiP_Dac_scaling;
   // <device>.Dac_scaling.um_lo, number, double
   // <device>.Dac_scaling.um_hi, number, double
   // <device>.Dac_scaling.sp_lo, number, uint32_t, 0-((2**20)-1)
   // <device>.Dac_scaling.sp_hi, number, uint32_t, 0-((2**20)-1)

   // Local members to store data
   double m_DacA_um{0.0};                      // Micron positions
   double m_DacB_um{0.0};
   double m_DacC_um{0.0};
   uint32_t m_DacA_setpoint{0};                // Integer positions
   uint32_t m_DacB_setpoint{0};
   uint32_t m_DacC_setpoint{0};
   int m_FPGA_fd{-1};                          // /dev/ FPGA file descr
   CGraphFSMHardwareInterface* m_p_interface;  // FPGA memory map ptr

   //TODO:  are some of these obsolete or not needed or redundant?
   //IMAGE m_shmimStream;
   //std::string m_shmimChannel {""};                            ///< The name of the shmim file
   //bool m_shmimOpened {false};
   //bool m_shmimRestart {false};
   //bool m_shmimOffloading {false};

   static constexpr uint32_t m_shmimWidth {3};                   ///< The width of the "image"
   static constexpr uint32_t m_shmimHeight {1};                  ///< The height of the "image"
   static constexpr uint8_t m_shmimDataType{IMAGESTRUCT_DOUBLE}; ///< The ImageStreamIO type code.
   static constexpr size_t m_shmimTypeSize {8};                  ///< The size of the type, in bytes.

public:
   /// Default c'tor.
   CGFSMHIfpga();

   /// D'tor, declared and defined for noexcept.
   ~CGFSMHIfpga() noexcept
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

   /// Implementation of the FSM for CGFSMHIfpga.
   /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.  Currently nothing in this app
   virtual int appShutdown();

   /// shmimMonitor thread callbacks
   /** -     allocate - called during startup/setup to assign pointers
     * - processImage - called when semaphore is triggered
     */
   int allocate( const dev::shmimT & placeholder     /**< [in] tag to differentiate shmimMonitor parents.*/);

   int processImage( void * curr_src,                ///< [in] pointer to start of current frame.
                     const dev::shmimT & placeholder ///< [in] tag to differentiate shmimMonitor parents.
                   );

   /// - Writing data FPGA device
   int updateDacs(bool update_INDI /* = false */);

   /// INDI input callbacks:  source of the DAC um data; DAC um data;
   ///                        DAC scaling data
   INDI_NEWCALLBACK_DECL(CGFSMHIfpga, m_indiP_setpoint_source);
   INDI_NEWCALLBACK_DECL(CGFSMHIfpga, m_indiP_Dac_um);
   INDI_NEWCALLBACK_DECL(CGFSMHIfpga, m_indiP_Dac_scaling);

   /// INDI output callbacks:  DAC um data; converted to uint32_t
   INDI_SETCALLBACK_DECL(CGFSMHIfpga, Dac_setpoint);

};
// class CGFSMHIfpga ...
////////////////////////////////////////////////////////////////////////

CGFSMHIfpga::CGFSMHIfpga() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   return;
}

/// Setup to read configuration data from TOML file
void CGFSMHIfpga::setupConfig()
{
   // Setup shmimMonitor, which will set up three TOML config parameters
   // - <cs>.threadPrio
   // - <cs>.cpuset
   // - <cs>.shmimName
   // where <cs> is returned by specificT::configSection() call of
   // dev::shmimMonitor<parentT,specificT>
   // - N.B. specificT defaults to shmimT (as it does in this class);
   //        shmimT::configSection() returns std::string("shmimMonitor")
   // So in TOML file, the shmim configuration could look like this:
   //
   //     [shmimMonitor]
   //     threadPrio = N
   //     cpuset = M
   //     shmimName = fpga0
   //
   shmimMonitorT::setupConfig(config);

   config.add("linux_dev", "", "linux_dev", argType::Required, "", "linux_dev", false, "string", "The full name (e.g. /dev/fsmfpga) of the device this app will drive.");
   config.add("setpoint_source", "", "setpoint_source", argType::Required, "", "setpoint_source", false, "string", "The source of the setpoint, either [indi] or [shm] or [disabled]");
   config.add("scaling.um_lo", "", "scaling.um_lo", argType::Required, "scaling", "um_lo", false, "real", "The lowest user setpoint, um [0.0]");
   config.add("scaling.um_hi", "", "scaling.um_hi", argType::Required, "scaling", "um_hi", false, "real", "The highest user setpoint, um [10.0]");
   config.add("scaling.int_lo", "", "scaling.int_lo", argType::Required, "scaling", "int_lo", false, "int", "The lowest integer device setpoint [0]");
   config.add("scaling.int_hi", "", "scaling.int_hi", argType::Required, "scaling", "int_hi", false, "int", "The hiwest integer device setpoint [1048575 = 2^20 - 1]");
}

// Load configuration from TOML file, using setup from setupConfig above
int CGFSMHIfpga::loadConfigImpl( mx::app::appConfigurator & _config )
{
   shmimMonitorT::loadConfig(_config);

   _config(m_linux_dev, "linux_dev");
   std::string spsrc;
   _config(spsrc, "setpoint_source");
   m_setpoint_source = (spsrc=="indi")
                       ? CGFSMHIfpga::SPSource::indi
                       : ( (spsrc=="shm")
                           ? CGFSMHIfpga::SPSource::shm
                           : ( (spsrc=="disabled")
                               ? CGFSMHIfpga::SPSource::disabled
                               : CGFSMHIfpga::SPSource::invalid
                             )
                         );
   // Also scaling limits
   m_Dac_scaling.limits_from_config(_config, "scaling.");
   return 0;
}

// Override of virtural loadConfig() is a wrapper for loadConfigImpl()
void CGFSMHIfpga::loadConfig()
{
   loadConfigImpl(config);
}

int CGFSMHIfpga::appStartup()
{
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   // TBD:  open FPGA device, map device memory

   // INDI interface to the setpoint source property
   createStandardIndiSelectionSw( m_indiP_setpoint_source, "setpoint_source", {"indi", "shm", "disabled"});
   m_indiP_setpoint_source["indi"] = m_setpoint_source == SPSource::indi ? pcf::IndiElement::On : pcf::IndiElement::Off;
   m_indiP_setpoint_source["shm"] = m_setpoint_source == SPSource::shm ? pcf::IndiElement::On : pcf::IndiElement::Off;
   m_indiP_setpoint_source["disabled"] = m_setpoint_source == SPSource::disabled ? pcf::IndiElement::On : pcf::IndiElement::Off;
   if( registerIndiPropertyNew( m_indiP_setpoint_source, INDI_NEWCALLBACK(m_indiP_setpoint_source)) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }

   // INDI interface to the micron (um) user input values
   REG_INDI_NEWPROP(m_indiP_Dac_um, "Dac_um", pcf::IndiProperty::Number);
   indi::addNumberElement<double>( m_indiP_Dac_um, "A",  0.0,  10.0, 1.0e-5,  "%f", "");
   indi::addNumberElement<double>( m_indiP_Dac_um, "B",  0.0,  10.0, 1.0e-5,  "%f", "");
   indi::addNumberElement<double>( m_indiP_Dac_um, "C",  0.0,  10.0, 1.0e-6,  "%f", "");
   m_indiP_Dac_um["A"].set<double>(m_DacA_um = 0.0);
   m_indiP_Dac_um["B"].set<double>(m_DacB_um = 0.0);
   m_indiP_Dac_um["C"].set<double>(m_DacC_um = 0.0);

   // INDI read-only interface to the output values to the DACs
   REG_INDI_NEWPROP_NOCB(m_indiP_Dac_setpoint, "Dac_setpoint", pcf::IndiProperty::Number);
   indi::addNumberElement<double>(m_indiP_Dac_setpoint, "A", 0, 1048575, 1, "%d", "");
   indi::addNumberElement<double>(m_indiP_Dac_setpoint, "B", 0, 1048575, 1, "%d", "");
   indi::addNumberElement<double>(m_indiP_Dac_setpoint, "C", 0, 1048575, 1, "%d", "");
   m_indiP_Dac_setpoint["A"].set<uint32_t>(m_DacA_setpoint = 0);
   m_indiP_Dac_setpoint["B"].set<uint32_t>(m_DacB_setpoint = 0);
   m_indiP_Dac_setpoint["C"].set<uint32_t>(m_DacC_setpoint = 0);

   // INDI interface to the Dacs' scaling
   REG_INDI_NEWPROP(m_indiP_Dac_scaling, "Dac_scaling", pcf::IndiProperty::Number);
   indi::addNumberElement<double>(m_indiP_Dac_scaling, "um_lo",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1, "%.6f", "");
   indi::addNumberElement<double>(m_indiP_Dac_scaling, "um_hi",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1, "%.6f", "");
   indi::addNumberElement<double>(m_indiP_Dac_scaling, "int_lo", 0, 0xFFFFFFFF, 1, "%d", "");
   indi::addNumberElement<double>(m_indiP_Dac_scaling, "int_hi", 0, 0xFFFFFFFF, 1, "%d", "");
   indi::addNumberElement<double>(m_indiP_Dac_scaling, "slope",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1, "%.6f", "");
   indi::addNumberElement<double>(m_indiP_Dac_scaling, "offset",  std::numeric_limits<double>::min(),  std::numeric_limits<double>::max(), 1, "%.6f", "");
   m_indiP_Dac_scaling["um_lo"].set<double>(m_Dac_scaling.um_lo());
   m_indiP_Dac_scaling["um_hi"].set<double>(m_Dac_scaling.um_hi());
   m_indiP_Dac_scaling["int_lo"].set<uint32_t>(m_Dac_scaling.int_lo());
   m_indiP_Dac_scaling["int_hi"].set<uint32_t>(m_Dac_scaling.int_hi());
   m_indiP_Dac_scaling["slope"].set<double>(m_Dac_scaling.slope());
   m_indiP_Dac_scaling["offset"].set<double>(m_Dac_scaling.offset());

   //state(stateCodes::READY);
   state(stateCodes::OPERATING);

   return 0;
}

int CGFSMHIfpga::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   // TBD:  read um inputs from shmim
   return 0;
}

int CGFSMHIfpga::appShutdown()
{
   shmimMonitorT::appShutdown();
   // TBD:  close FPGA device
   return 0;
}

// Scale floating point values to FPGA-ranged integers
int CGFSMHIfpga::updateDacs(bool update_INDI = false)
{
   // Scale to D/A counts, integer, 0-1M
   // N.B. setpoint values in engineering units are assigned elsewhere
   m_DacA_setpoint = m_Dac_scaling.scale(m_DacA_um);
   m_DacB_setpoint = m_Dac_scaling.scale(m_DacB_um);
   m_DacC_setpoint = m_Dac_scaling.scale(m_DacC_um);

   if (update_INDI || (CGFSMHIfpga::SPSource::indi == m_setpoint_source))
   {
      log<text_log>("set new D/A setpoints: {A:" + std::to_string(m_DacA_setpoint)
                                         + ",B:" + std::to_string(m_DacB_setpoint)
                                         + ",C:" + std::to_string(m_DacC_setpoint)
                                         + ",slope:" + std::to_string(m_Dac_scaling.slope())
                                         + ",offset:" + std::to_string(m_Dac_scaling.offset())
                                         + "}"
                   , logPrio::LOG_NOTICE
                   );

      // publish Dac setpoints
      m_indiP_Dac_setpoint["A"] = m_DacA_setpoint;
      m_indiP_Dac_setpoint["B"] = m_DacB_setpoint;
      m_indiP_Dac_setpoint["C"] = m_DacC_setpoint;
      if(m_indiDriver) { m_indiDriver->sendSetProperty (m_indiP_Dac_setpoint); }
   }

   return 0;
}

int CGFSMHIfpga::allocate(const dev::shmimT & placeholder)
{
   static_cast<void>(placeholder); //be unused

   std::lock_guard<std::mutex> guard(m_indiMutex);

   int err{0};

   if(shmimMonitorT::m_width != m_shmimWidth)
   {
      log<software_critical>({__FILE__,__LINE__, "shmim width is " +  std::to_string(shmimMonitorT::m_width) + " not " + std::to_string(m_shmimWidth)});
      ++err;
   }

   if(shmimMonitorT::m_height != m_shmimHeight)
   {
      log<software_critical>({__FILE__,__LINE__, "shmim height is " + std::to_string(shmimMonitorT::m_height) + " not " + std::to_string(m_shmimHeight)});
      ++err;
   }

   //if(shmimMonitorT::m_dataType != IMAGESTRUCT_DOUBLE)
   if(shmimMonitorT::m_dataType != IMAGESTRUCT_DOUBLE)
   {
      log<software_critical>({__FILE__,__LINE__, "shmim data type is " + std::to_string(shmimMonitorT::m_dataType) + " not " + std::to_string(m_shmimHeight)});
      ++err;
   }

   log<text_log>(err ? "allocate(...) failure" : "allocate(...) success", logPrio::LOG_NOTICE);

   if(err) return -1;

#if 0
   // I think this code may not be needed in this particular case
   // ImageStreamIO_openIm(...) et al. are called from caller of
   // this routine, and the size is fixed

   if(m_shmimOpened)
   {
      ImageStreamIO_closeIm(&m_shmimStream);
   }

   m_shmimOpened = false;
   m_shmimRestart = false; //Set this up front, since we're about to restart.

   if( ImageStreamIO_openIm(&m_shmimStream, m_shmimChannel.c_str()) == 0)
   {
      if(m_shmimStream.md[0].sem < 10)
      {
         ImageStreamIO_closeIm(&m_shmimStream);
      }
      else
      {
         m_shmimOpened = true;
      }
   }

   if(!m_shmimOpened)
   {
      log<software_error>({__FILE__, __LINE__, m_shmimChannel + " not opened."});
      return -1;
   }
   else
   {
      log<text_log>( "Opened " + m_shmimChannel + " " + std::to_string(m_shmimWidth) + " x " + std::to_string(m_shmimHeight) + " with data type: " + std::to_string(m_shmimDataType));
   }
#endif//0

   return 0;

} // int CGFSMHIfpga::allocate(const dev::shmimT & placeholder)

inline
int CGFSMHIfpga::processImage( void* curr_src,
                               const dev::shmimT & placeholder
                             )
{
   static_cast<void>(placeholder);

   // Poor-man's mutex
   while(shmimMonitorT::m_imageStream.md->write == 1);


   if(SPSource::disabled != m_setpoint_source)
   {
      // Put three micron (um) setpoints from shared memory into
      // local attributes
      double* ptr = (double*) curr_src;
      m_DacA_um = *(ptr++);
      m_DacB_um = *(ptr++);
      m_DacC_um = *(ptr++);

      // Update the D/A converters
      updateDacs();
   }

   if(SPSource::indi == m_setpoint_source)
   {
      // Update three micron (um) INDI properties, and log this call
      m_indiP_Dac_um["A"] = m_DacA_um;
      m_indiP_Dac_um["B"] = m_DacB_um;
      m_indiP_Dac_um["C"] = m_DacC_um;

      m_indiP_Dac_um.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_Dac_um);

      log<text_log>("Setpoints updated from shmim via processImage(...)", logPrio::LOG_NOTICE);
   }


   return 0;
}

// Callback to change source of input um values
INDI_NEWCALLBACK_DEFN(CGFSMHIfpga, m_indiP_setpoint_source )(const pcf::IndiProperty &ipRecv)
{
   if(ipRecv.getName() != m_indiP_setpoint_source.getName())
   {
      log<software_error>({__FILE__, __LINE__, "invalid indi property received"});
      return -1;
   }

   CGFSMHIfpga::SPSource spsrc = CGFSMHIfpga::SPSource::invalid;

   if(ipRecv.find("indi"))
   {
      if(ipRecv["indi"].getSwitchState() == pcf::IndiElement::On)
      {
         spsrc = CGFSMHIfpga::SPSource::indi;
      }
   }

   if(ipRecv.find("shm"))
   {
      if(ipRecv["shm"].getSwitchState() == pcf::IndiElement::On)
      {
         if(spsrc != CGFSMHIfpga::SPSource::invalid)
         {
            log<text_log>("can not set source to multiple values", logPrio::LOG_ERROR);
         }
         else spsrc = CGFSMHIfpga::SPSource::shm;
      }
   }

   if(ipRecv.find("disabled"))
   {
      if(ipRecv["disabled"].getSwitchState() == pcf::IndiElement::On)
      {
         if(spsrc != CGFSMHIfpga::SPSource::invalid)
         {
            log<text_log>("can not set source to multiple values", logPrio::LOG_ERROR);
         }
         else spsrc = CGFSMHIfpga::SPSource::disabled;
      }
   }

   if(spsrc != CGFSMHIfpga::SPSource::invalid)
   {
      m_setpoint_source = spsrc;

      m_indiP_setpoint_source["indi"] = spsrc == SPSource::indi ? pcf::IndiElement::On : pcf::IndiElement::Off;
      m_indiP_setpoint_source["shm"] = spsrc == SPSource::shm ? pcf::IndiElement::On : pcf::IndiElement::Off;
      m_indiP_setpoint_source["disabled"] = spsrc == SPSource::disabled ? pcf::IndiElement::On : pcf::IndiElement::Off;
      m_indiP_setpoint_source.setState (INDI_OK);
      if (m_indiDriver) { m_indiDriver->sendSetProperty (m_indiP_setpoint_source); }
   }

   // Ensure values displayed in INDI are correct
   if (m_indiDriver && m_setpoint_source == SPSource::indi)
   {
      m_indiP_Dac_um["A"] = m_DacA_um;
      m_indiP_Dac_um["B"] = m_DacB_um;
      m_indiP_Dac_um["C"] = m_DacC_um;
      m_indiP_Dac_um.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_Dac_um);

      m_indiP_Dac_setpoint["A"] = m_DacA_setpoint;
      m_indiP_Dac_setpoint["B"] = m_DacB_setpoint;
      m_indiP_Dac_setpoint["C"] = m_DacC_setpoint;
      m_indiP_Dac_setpoint.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_Dac_setpoint);
   }

   return 0;
}

// Callback to specify an input um value
INDI_NEWCALLBACK_DEFN(CGFSMHIfpga, m_indiP_Dac_um)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == m_indiP_Dac_um.getName())
   {

      if(SPSource::indi == m_setpoint_source)
      {
         // Received a new value for a micron (um) setpoint from INDI
         if(ipRecv.find("A"))
         {
            m_indiP_Dac_um["A"] = m_DacA_um = ipRecv["A"].get<double>();
         }
         if(ipRecv.find("B"))
         {
            m_indiP_Dac_um["B"] = m_DacB_um = ipRecv["B"].get<double>();
         }
         if(ipRecv.find("C"))
         {
            m_indiP_Dac_um["C"] = m_DacC_um = ipRecv["C"].get<double>();
         }
         updateDacs(true);
      }

      m_indiP_Dac_um.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_Dac_um);

      return 0;
   }
   return -1;
}


INDI_NEWCALLBACK_DEFN(CGFSMHIfpga, m_indiP_Dac_scaling)(const pcf::IndiProperty &ipRecv)
{

   if (ipRecv.getName() == m_indiP_Dac_scaling.getName())
   {
      double um_lo{m_Dac_scaling.um_lo()};
      double um_hi{m_Dac_scaling.um_hi()};
      uint32_t int_lo{m_Dac_scaling.int_lo()};
      uint32_t int_hi{m_Dac_scaling.int_hi()};
      // received a new value for property val
      if(ipRecv.find("um_lo"))
      {
         m_indiP_Dac_scaling["um_lo"] = um_lo = ipRecv["um_lo"].get<double>();
      }
      if(ipRecv.find("um_hi"))
      {
         m_indiP_Dac_scaling["um_hi"] = um_hi = ipRecv["um_hi"].get<double>();
      }
      if(ipRecv.find("int_lo"))
      {
         m_indiP_Dac_scaling["int_lo"] = int_lo = ipRecv["int_lo"].get<uint32_t>();
      }
      if(ipRecv.find("int_hi"))
      {
         m_indiP_Dac_scaling["int_hi"] = int_hi = ipRecv["int_hi"].get<uint32_t>();
      }

      m_Dac_scaling.setLimits(um_lo, um_hi, int_lo, int_hi);

      m_indiP_Dac_scaling["slope"] = m_Dac_scaling.slope();
      m_indiP_Dac_scaling["offset"] = m_Dac_scaling.offset();

      m_indiP_Dac_scaling.setState (pcf::IndiProperty::Ok);
      m_indiDriver->sendSetProperty (m_indiP_Dac_scaling);

      return 0;
   }
   return -1;
}
} //namespace app
} //namespace MagAOX

#endif //CGFSMHIfpga_hpp
