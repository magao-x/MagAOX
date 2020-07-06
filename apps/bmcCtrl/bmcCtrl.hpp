/** \file bmcCtrl.hpp
  * \brief The MagAO-X BMC DM controller header file
  *
  * \ingroup bmcCtrl_files
  */



/*
Open questions:
* Fix actuator mapping issue (don't want a bunch of if statements)


Test:
* bias code

*/


// #define _GLIBCXX_USE_CXX11_ABI 0


#ifndef bmcCtrl_hpp
#define bmcCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


/* BMC SDK C Header */
#include <BMCApi.h>


/** \defgroup bmcCtrl 
  * \brief The MagAO-X application to control a BMC DM
  *
  * <a href="..//apps_html/page_module_bmcCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup bmcCtrl_files
  * \ingroup bmcCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X BMC DM Controller
/** 
  * \ingroup bmcCtrl
  */
class bmcCtrl : public MagAOXApp<true>, public dev::dm<bmcCtrl,float>, public dev::shmimMonitor<bmcCtrl>
{

   //Give the test harness access.
   friend class bmcCtrl_test;
   
   friend class dev::dm<bmcCtrl,float>;
   
   friend class dev::shmimMonitor<bmcCtrl>;
   
   typedef float realT;  ///< This defines the datatype used to signal the DM using the ImageStreamIO library.
   
   size_t m_nsat {0};
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_serialNumber; ///< The BMC serial number used to open the correct DM profile
   
   ///@}

public:
   /// Default c'tor.
   bmcCtrl();

   /// D'tor.
   ~bmcCtrl() noexcept;

   /// Setup the configuration system.
   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   /// Load the configuration
   virtual void loadConfig();

   /// Startup function
   /** Sets up INDI, and starts the shmim thread.
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for bmcCtrl.
   /** 
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
   virtual int appLogic();

   /// Shutdown the app.
   /** 
     *
     */
   virtual int appShutdown();

   /** \name DM Base Class Interface
     *
     *@{
     */
   
   /// Initialize the DM and prepare for operation.
   /** Application is in state OPERATING upon successful conclusion.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */ 
   int initDM();
   
   /// Zero all commands on the DM
   /** This does not update the shared memory buffer.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */
   int zeroDM();
   
   /// Send a command to the DM
   /** This is called by the shmim monitoring thread in response to a semaphore trigger.
     * 
     * \returns 0 on success 
     * \returns -1 on error
     */
   int commandDM(void * curr_src);
   
   /// Release the DM, making it safe to turn off power.
   /** The application will be state READY at the conclusion of this.
     *  
     * \returns 0 on success 
     * \returns -1 on error
     */
   int releaseDM();
   
   ///@}
   
   /** \name BMC Interface
     * \todo document these members
     *@{
     */
   
protected:
   double m_act_gain {0}; ///< Actuator gain (microns/volt)
   double m_volume_factor {0}; ///< the volume factor to convert from displacement to commands
   uint32_t m_nbAct {0}; ///< The number of actuators
   
   int * m_actuator_mapping {nullptr}; ///< Array containing the mapping from 2D grid position to linear index in the command vector
   
   double * m_dminputs {nullptr}; ///< Pre-allocated command vector, used only in commandDM
   
   DM m_dm = {}; ///< BMC SDK handle for the DM.
   bool m_dmopen {false}; ///< Track whether the DM connection has been opened
   
public:
   
   /// Parse the BMC calibration file
   /** \returns 0 on success
     * \returns -1 on error
     */  
   int parse_calibration_file(); 
   
   /// Read the actuator mapping from a FITS file
   /**
     * \todo convert this to use mxlib::fitsFile
     *
     * \returns 0 on success
     * \returns -1 on error
     */ 
   int get_actuator_mapping();
   
   ///@}
};

bmcCtrl::bmcCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   return;
}

bmcCtrl::~bmcCtrl() noexcept
{
   if(m_actuator_mapping) free(m_actuator_mapping);
   if(m_dminputs) free(m_dminputs);
   
}   
   
void bmcCtrl::setupConfig()
{
   config.add("dm.serialNumber", "", "dm.serialNumber", argType::Required, "dm", "serialNumber", false, "string", "The BMC serial number used to find correct DM Profile.");
   config.add("dm.calibRelDir", "", "dm.calibRelDir", argType::Required, "dm", "calibRelDir", false, "string", "Used to find the default config directory.");
   dev::dm<bmcCtrl,float>::setupConfig(config);
   
}

int bmcCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   config(m_calibRelDir, "dm.calibRelDir");
   config(m_serialNumber, "dm.serialNumber");
      
   //m_calibRelDir = "dm/bmc_2k";
   
   dev::dm<bmcCtrl,float>::loadConfig(_config);
   
   return 0;
}

void bmcCtrl::loadConfig()
{
   loadConfigImpl(config);
   
}

int bmcCtrl::appStartup()
{
   if(parse_calibration_file() < 0)
   {
      log<software_critical>({__FILE__,__LINE__});
      return -1;
   }
   
   if(m_act_gain == 0 || m_volume_factor == 0)
   {
      log<software_critical>({__FILE__,__LINE__, "calibration not loaded properly"});
      return -1;
   }
   
   dev::dm<bmcCtrl,float>::appStartup();
   shmimMonitor<bmcCtrl>::appStartup();
   
   return 0;
}

int bmcCtrl::appLogic()
{
   dev::dm<bmcCtrl,float>::appLogic();
   shmimMonitor<bmcCtrl>::appLogic();
   
   if(state()==stateCodes::POWEROFF) return 0;
   
   if(state()==stateCodes::POWERON)
   {
      sleep(5);
      return initDM();
   }
   
   if(m_nsat > 0)
   {
      log<text_log>("Saturated actuators in last second: " + std::to_string(m_nsat), logPrio::LOG_WARNING);
   }
   m_nsat = 0;
   
   return 0;
}

int bmcCtrl::appShutdown()
{
   if(m_dmopen) releaseDM();
      
   dev::dm<bmcCtrl,float>::appShutdown();
   shmimMonitor<bmcCtrl>::appShutdown();
   
   return 0;
}


int bmcCtrl::initDM()
{
   if(m_dmopen)
   {
      log<text_log>("DM is already initialized.  Release first.", logPrio::LOG_ERROR);
      return -1;
   }

   std::string ser = mx::ioutils::toUpper(m_serialNumber);
   BMCRC ret = NO_ERR;
   ret = BMCOpen(&m_dm, ser.c_str());

   if(ret == NO_ERR) m_dmopen = true; // remember that the DM connection has been opened 

   if(ret != NO_ERR)
   {
      const char *err;
      err = BMCErrorString(ret);
      log<text_log>(std::string("DM initialization failed: ") + err, logPrio::LOG_ERROR);
      
      m_dm = {};
      return -1;
   }
   
   if (!m_dmopen)
   {
      log<text_log>("DM initialization failed. Couldn't open DM handle.", logPrio::LOG_ERROR);
      return -1;
   }
   
   log<text_log>("BMC " + m_serialNumber + " initialized", logPrio::LOG_NOTICE);

   // Get number of actuators
   m_nbAct = m_dm.ActCount;


   // Load the DM map
   uint32_t *map_lut;
   map_lut = (uint32_t *)malloc(sizeof(uint32_t)*MAX_DM_SIZE);
   ret = BMCLoadMap(&m_dm, NULL, map_lut);

   if(ret != NO_ERR)
   {
      const char *err;
      err = BMCErrorString(ret);
      log<text_log>(std::string("DM initialization failed. Couldn't load map.") + err, logPrio::LOG_ERROR);
      
      m_dm = {};
      return -1;
   }

   
   if(m_dminputs) free(m_dminputs);
   m_dminputs = (double*) calloc( m_nbAct, sizeof( double ) );
   
   if(zeroDM() < 0)
   {
      log<text_log>("DM initialization failed.  Error zeroing DM.", logPrio::LOG_ERROR);
      return -1;
   }
   
   /* get actuator mapping from 2D cacao image to 1D vector for BMC input */
   if(m_actuator_mapping) free(m_actuator_mapping);
   m_actuator_mapping = (int *) malloc(m_nbAct * sizeof(int)); /* memory for actuator mapping */

    /* initialize to -1 to allow for handling addressable but ignored actuators */
    for (uint32_t idx = 0; idx < m_nbAct; ++idx)
    {
        m_actuator_mapping[idx] = -1;
    }
   
   if(get_actuator_mapping() < 0)
   {
      log<text_log>("DM initialization failed.  Failed to get actuator mapping.", logPrio::LOG_ERROR);
      return -1;
   }

   if(m_actuator_mapping == nullptr)
   {
      log<text_log>("DM initialization failed.  null pointer.", logPrio::LOG_ERROR);
      return -1;
   }
   
   state(stateCodes::OPERATING);
   
   return 0;
}

int bmcCtrl::zeroDM()
{
   if(!m_dmopen)
   {
      log<text_log>("DM not initialized (NULL pointer)", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(m_nbAct == 0)
   {
      log<text_log>("DM not initialized (number of actuators)", logPrio::LOG_ERROR);
      return -1;
   }

   double * dminputs = (double*) calloc( m_nbAct, sizeof( double ) );
   
   /* Send the all 0 command to the DM */
   BMCRC ret = BMCSetArray(&m_dm, dminputs, NULL);

   /* Release memory */
   free( dminputs );

   if(ret != NO_ERR)
   {
      const char *err;
      err = BMCErrorString(ret);
      log<text_log>(std::string("Error zeroing DM: ") + err, logPrio::LOG_ERROR);
      return -1;
   }
   
   log<text_log>("DM zeroed");
   return 0;
}

int bmcCtrl::commandDM(void * curr_src)
{
   //This is based on Kyle Van Gorkoms original sendCommand function.
   
   /*This loop performs the following steps:
     1) converts from float to double
     2) convert to volume-normalized displacement
     3) convert to squared fractional voltage (0 to +1)
     4) calculate the mean
   */


   // want to rework the logic here so that we don't have to check
   // if every actuator is addressable.
   // Loop over addressable only?
   //double mean = 0;
   for (uint32_t idx = 0; idx < m_nbAct; ++idx)
   {
     int address = m_actuator_mapping[idx];
     if(address == -1)
     {
        m_dminputs[idx] = 0.; // addressable but ignored actuators set to 0
     } 
     else 
     {
        m_dminputs[idx] = ((double)  ((realT *) curr_src)[address]) * m_volume_factor/m_act_gain;
        //mean += m_dminputs[idx];
     }
   }
   //mean /= m_nbAct;

   /*This loop performas the following steps:
      1) remove mean from each actuator input (and add midpoint bias)
      2) clip to fractional values between 0 and 1.
      3) take the square root to approximate the voltage-displacement curve
   */
   int nsat = 0;
   for (uint32_t idx = 0 ; idx < m_nbAct ; ++idx)
   {
      //m_dminputs[idx] -= mean - 0.5;
      if (m_dminputs[idx] > 1)
      {
         ++m_nsat;
         m_dminputs[idx] = 1;
      } else if (m_dminputs[idx] < 0)
      {
         ++m_nsat;
         m_dminputs[idx] = 0;
      }
      m_dminputs[idx] = sqrt(m_dminputs[idx]);
   }

   
   //for (uint32_t idx = 0 ; idx < m_nbAct ; ++idx){
   // printf("Acuator %d: %f\n", idx+1, m_dminputs[idx]);
   //}
   //printf("Mean %f\n", mean);
    
   /* Finally, send the command to the DM */
   BMCRC ret = BMCSetArray(&m_dm, m_dminputs, NULL);

   /* Return immediately upon success. Otherwise, log the error
   message first and then return the failure code. */
   if(ret != NO_ERR)
   {
      const char *err;
      err = BMCErrorString(ret);
      log<text_log>(std::string("DM command failed: ") + err, logPrio::LOG_ERROR);
      return -1;
   }

   return ret;
    
}

int bmcCtrl::releaseDM()
{
   // Safe DM shutdown on interrupt

   if(!m_dmopen)
   {
      log<text_log>("dm is not initialized", logPrio::LOG_ERROR);
      return -1;
   }
   
   state(stateCodes::READY);
   
   if(!shutdown())
   {
      pthread_kill(m_smThread.native_handle(), SIGUSR1);
   }
   
   sleep(1); ///\todo need to trigger shmimMonitor loop to pause it.
   
   if(zeroDM() < 0)
   {
      log<text_log>("DM release failed.  Error zeroing DM.", logPrio::LOG_ERROR);
      return -1;
   }

    // Zero all actuators (this is probably redundant after zeroing the DM above)
    BMCRC ret = NO_ERR;
    ret = BMCClearArray(&m_dm);

   if(ret != NO_ERR)
   {
      const char *err;
      err = BMCErrorString(ret);
      log<text_log>(std::string("DM reset failed: ") + err, logPrio::LOG_ERROR);
      return -1;
   }
   
   // Close BMC connection
   ret = BMCClose(&m_dm);

   if(ret == NO_ERR) m_dmopen = false;

   if(ret != NO_ERR)
   {
      const char *err;
      err = BMCErrorString(ret);
      log<text_log>(std::string("DM release failed: ") + err, logPrio::LOG_ERROR);
      return -1;
   }

   m_dm = {};
   
   log<text_log>("BMC " + m_serialNumber + " reset and released", logPrio::LOG_NOTICE);
   
   return 0;
}

/* Read in a configuration file with user-calibrated
values to determine the conversion from physical to
fractional stroke as well as the volume displaced by
the influence function. */
int bmcCtrl::parse_calibration_file()
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    double * calibvals;

    std::string calibpath = m_calibPath + "/" + "bmc_2k_userconfig.txt";
    
    // open file
    fp = fopen(calibpath.c_str(), "r");
    if (fp == NULL)
    {
       log<text_log>("Could not read configuration file at " + calibpath, logPrio::LOG_ERROR);
        return -1;
    }

    calibvals = (double*) malloc(2*sizeof(double));
    int idx = 0;
    while ((read = getline(&line, &len, fp)) != -1)
    {
        // grab first value from each line
        calibvals[idx] = strtod(line, NULL);
        idx++;
    }

    fclose(fp);

    // assign stroke and volume factors
    m_act_gain = calibvals[0];
    m_volume_factor = calibvals[1];

    free(calibvals);
    
    log<text_log>("BMC " + m_serialNumber + ": Using stroke and volume calibration from " + calibpath);
    std::cerr << m_act_gain << " " << m_volume_factor << "\n";
    return 0;
}

int bmcCtrl::get_actuator_mapping() //const char * serial, int nbAct, int * actuator_mapping)
{
    /* This function closely follows the CFITSIO imstat
    example */

    fitsfile *fptr;  /* FITS file pointer */
    int status = 0;  /* CFITSIO status value MUST be initialized to zero! */
    

    // get file path to actuator map
    std::string calibpath = m_calibPath + "/" + "bmc_2k_actuator_mapping.fits";

    if ( !fits_open_image(&fptr, calibpath.c_str(), READONLY, &status) )
    {
      int hdutype, naxis;
      long naxes[2];   
       
      if (fits_get_hdu_type(fptr, &hdutype, &status) || hdutype != IMAGE_HDU) { 
        printf("Error: this program only works on images, not tables\n");
        return(1);
      }

      fits_get_img_dim(fptr, &naxis, &status);
      fits_get_img_size(fptr, 2, naxes, &status);

      if (status || naxis != 2) { 
        printf("Error: NAXIS = %d.  Only 2-D images are supported.\n", naxis);
        return(1);
      }

      int * pix = (int *) malloc(naxes[0] * sizeof(int)); /* memory for 1 row */

      if (pix == NULL) {
        printf("Memory allocation error\n");
        return(1);
      }

      long fpixel[2];
      //totpix = naxes[0] * naxes[1];
      fpixel[0] = 1;  /* read starting with first pixel in each row */

      /* process image one row at a time; increment row # in each loop */
      int ij = 0;/* actuator mapping index */
      for (fpixel[1] = 1; fpixel[1] <= naxes[1]; fpixel[1]++)
      {  
         /* give starting pixel coordinate and number of pixels to read */
         if (fits_read_pix(fptr, TINT, fpixel, naxes[0],0, pix,0, &status))
            break;   /* jump out of loop on error */

         // get indices of active actuators in order
         for (int ii = 0; ii < naxes[0]; ii++) {
           if (pix[ii] > 0) {
                m_actuator_mapping[pix[ii] - 1] = ij;
           }
           ij++;
         }
      }
      fits_close_file(fptr, &status);
      
      free(pix);
    }

    if (status)  {
        fits_report_error(stderr, status); /* print any error message */
    }


    log<text_log>("BMC " + m_serialNumber + ": Using actuator mapping from " + calibpath);
    return 0;
}

} //namespace app
} //namespace MagAOX

#endif //bmcCtrl_hpp
