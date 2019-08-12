/** \file alpaoCtrl.hpp
  * \brief The MagAO-X ALPAO DM controller header file
  *
  * \ingroup alpaoCtrl_files
  */

#ifndef alpaoCtrl_hpp
#define alpaoCtrl_hpp


#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"


/* Alpao SDK C Header */
#include <asdkWrapper.h>


/** \defgroup alpaoCtrl 
  * \brief The MagAO-X application to control an ALPAO DM
  *
  * <a href="..//apps_html/page_module_alpaoCtrl.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup alpaoCtrl_files
  * \ingroup alpaoCtrl
  */

namespace MagAOX
{
namespace app
{

/// The MagAO-X ALPAO DM Controller
/** 
  * \ingroup alpaoCtrl
  */
class alpaoCtrl : public MagAOXApp<true>, public dev::dm<alpaoCtrl,float>, public dev::shmimMonitor<alpaoCtrl>
{

   //Give the test harness access.
   friend class alpaoCtrl_test;
   
   friend class dev::dm<alpaoCtrl,float>;
   
   friend class dev::shmimMonitor<alpaoCtrl>;
   
   typedef float realT;  ///< This defines the datatype used to signal the DM using the ImageStreamIO library.
   
protected:

   /** \name Configurable Parameters
     *@{
     */
   
   std::string m_serialNumber; ///< The ALPAO serial number used to find the default config directory.
   
   ///@}




public:
   /// Default c'tor.
   alpaoCtrl();

   /// D'tor.
   ~alpaoCtrl() noexcept;

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

   /// Implementation of the FSM for alpaoCtrl.
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
   
   /** \name ALPAO Interface
     * \todo document these members
     *@{
     */
   
protected:
   Scalar m_max_stroke {0}; ///< The maximum allowable stroke
   Scalar m_volume_factor {0}; ///< the volume factor to convert from displacement to commands
   UInt m_nbAct {0}; ///< The number of actuators
   
   int * m_actuator_mapping {nullptr}; ///< Array containing the mapping from 2D grid position to linear index in the command vector
   
   Scalar * m_dminputs {nullptr}; ///< Pre-allocated command vector, used only in commandDM
   
   asdkDM * m_dm {nullptr}; ///< ALPAO SDK handle for the DM.
   
public:
   
   /// Parse the ALPAO calibration file
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

alpaoCtrl::alpaoCtrl() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   m_powerMgtEnabled = true;
   return;
}

alpaoCtrl::~alpaoCtrl() noexcept
{
   if(m_actuator_mapping) free(m_actuator_mapping);
   if(m_dminputs) free(m_dminputs);
   
}   
   
void alpaoCtrl::setupConfig()
{
   config.add("dm.serialNumber", "", "dm.serialNumber", argType::Required, "dm", "serialNumber", false, "string", "The ALPAO serial number used to find the default config directory.");
   dev::dm<alpaoCtrl,float>::setupConfig(config);
   
}

int alpaoCtrl::loadConfigImpl( mx::app::appConfigurator & _config )
{
   config(m_serialNumber, "dm.serialNumber");
   
   m_calibRelDir = "dm/alpao_";
   
   std::string ser = mx::ioutils::toLower(m_serialNumber);
   
   m_calibRelDir += ser;
   dev::dm<alpaoCtrl,float>::loadConfig(_config);
   
   return 0;
}

void alpaoCtrl::loadConfig()
{
   loadConfigImpl(config);
   
   
}

int alpaoCtrl::appStartup()
{
   if(parse_calibration_file() < 0)
   {
      log<software_critical>({__FILE__,__LINE__});
      return -1;
   }
   
   if(m_max_stroke == 0 || m_volume_factor == 0)
   {
      log<software_critical>({__FILE__,__LINE__, "calibration not loaded properly"});
      return -1;
   }
   
   dev::dm<alpaoCtrl,float>::appStartup();
   shmimMonitor<alpaoCtrl>::appStartup();
   
   return 0;
}

int alpaoCtrl::appLogic()
{
   dev::dm<alpaoCtrl,float>::appLogic();
   shmimMonitor<alpaoCtrl>::appLogic();
   
   if(state()==stateCodes::POWEROFF) return 0;
   
   if(state()==stateCodes::POWERON)
   {
      sleep(5);
      return initDM();
   }
   
   
   return 0;
}

int alpaoCtrl::appShutdown()
{
   if(m_dm) releaseDM();
      
   dev::dm<alpaoCtrl,float>::appShutdown();
   shmimMonitor<alpaoCtrl>::appShutdown();
   

   
   return 0;
}

int alpaoCtrl::initDM()
{
   if(m_dm != nullptr)
   {
      log<text_log>("DM is already initialized.  Release first.", logPrio::LOG_ERROR);
      return -1;
   }
   
   std::string ser = mx::ioutils::toUpper(m_serialNumber);
   m_dm = asdkInit(ser.c_str());
   
   acs::UInt aerr = 0;
   asdkGetLastError(&aerr, nullptr, 0);
   if(aerr)
   {
      char err[1024];
      asdkGetLastError(&aerr, err, sizeof(err));
      log<text_log>(std::string("DM initialization failed: ") + err, logPrio::LOG_ERROR);
      
      m_dm = nullptr;
      return -1;
   }
   
   if (m_dm == NULL)
   {
      log<text_log>("DM initialization failed.  NULL pointer.", logPrio::LOG_ERROR);
      return -1;
   }
   
   log<text_log>("ALPAO " + m_serialNumber + " initialized", logPrio::LOG_NOTICE);

   // Get number of actuators
   Scalar tmp;
   if(asdkGet( m_dm, "NbOfActuator", &tmp ) < 0)
   {
      char err[1024];
      asdkGetLastError(&aerr, err, sizeof(err));
      log<text_log>(std::string("Getting number of actuators failed: ") + err, logPrio::LOG_ERROR);
      return -1;
   }
   m_nbAct = tmp;
   
   if(m_dminputs) free(m_dminputs);
   m_dminputs = (Scalar*) calloc( m_nbAct, sizeof( Scalar ) );
   
   if(zeroDM() < 0)
   {
      log<text_log>("DM initialization failed.  Error zeroing DM.", logPrio::LOG_ERROR);
      return -1;
   }
   
   /* get actuator mapping from 2D cacao image to 1D vector for ALPAO input */
   if(m_actuator_mapping) free(m_actuator_mapping);
   m_actuator_mapping = (int *) malloc(m_nbAct * sizeof(int)); /* memory for actuator mapping */
   
   if(get_actuator_mapping() < 0)
   {
      log<text_log>("DM initialization failed.  Failed to get actuator mapping.", logPrio::LOG_ERROR);
      return -1;
   }

   if( m_actuator_mapping == nullptr)
   {
      log<text_log>("DM initialization failed.  null pointer.", logPrio::LOG_ERROR);
      return -1;
   }
   
   state(stateCodes::OPERATING);
   
   return 0;
}

int alpaoCtrl::zeroDM()
{
   if(m_dm == nullptr)
   {
      log<text_log>("DM not initialized (NULL pointer)", logPrio::LOG_ERROR);
      return -1;
   }
   
   if(m_nbAct == 0)
   {
      log<text_log>("DM not initialized (number of actuators)", logPrio::LOG_ERROR);
      return -1;
   }

   Scalar * dminputs = (Scalar*) calloc( m_nbAct, sizeof( Scalar ) );
   
   /* Send the all 0 command to the DM */
   int ret = asdkSend(m_dm, dminputs);

   /* Release memory */
   free( dminputs );

   if(ret < 0)
   {
      log<text_log>("Error zeroing DM", logPrio::LOG_ERROR);
      return -1;
   }
   
   log<text_log>("DM zeroed");
   return 0;
}

int alpaoCtrl::commandDM(void * curr_src)
{
   COMPL_STAT ret;

   //This is based on Kyle Van Gorkoms original sendCommand function.
   
   /*This loop performs the following steps:
     1) converts from float to double (ALPAO Scalar)
     2) convert to volume-normalized displacement (microns)
     3) convert to fractional stroke (-1 to +1) that the ALPAO SDK expects
     4) calculate the mean
   */
   Scalar mean = 0;
   for (UInt idx = 0; idx < m_nbAct; ++idx)
   {
     m_dminputs[idx] = ((Scalar)  ((realT *) curr_src)[m_actuator_mapping[idx]]) * m_volume_factor/m_max_stroke;
     mean += m_dminputs[idx];
   }
   mean /= m_nbAct;

   /*This loop performas the following steps:
      1) remove mean from each actuator input
      2) clip to fractional values between -1 and 1.
         The ALPAO SDK doesn't seem to check for this, which
         is scary and a little odd.
   */
   for (UInt idx = 0 ; idx < m_nbAct ; ++idx)
   {
      m_dminputs[idx] -= mean;
      if (m_dminputs[idx] > 1)
      {
         printf("Actuator %d saturated!\n", idx + 1);
         m_dminputs[idx] = 1;
      } else if (m_dminputs[idx] < -1)
      {
         printf("Actuator %d saturated!\n", idx + 1);
         m_dminputs[idx] = - 1;
      }
   }
    

   /* Finally, send the command to the DM */
   ret = asdkSend(m_dm, m_dminputs);

   return ret;
    
}

int alpaoCtrl::releaseDM()
{
   // Safe DM shutdown on interrupt

   if(m_dm == nullptr)
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
   
   // Reset and release ALPAO
   asdkReset(m_dm);
    
   acs::UInt aerr = 0;
   asdkGetLastError(&aerr, nullptr, 0);
   if(aerr)
   {
      char err[1024];
      asdkGetLastError(&aerr, err, sizeof(err));
      log<text_log>(std::string("DM reset failed: ") + err, logPrio::LOG_ERROR);
      return -1;
   }
   
   asdkRelease(m_dm); ///\todo error check
   
   aerr = 0;
   asdkGetLastError(&aerr, nullptr, 0);
   if(aerr)
   {
      char err[1024];
      asdkGetLastError(&aerr, err, sizeof(err));
      log<text_log>(std::string("DM release failed: ") + err, logPrio::LOG_ERROR);
      return -1;
   }

   m_dm = nullptr;
   
   log<text_log>("ALPAO " + m_serialNumber + " reset and released", logPrio::LOG_NOTICE);
   
   return 0;
}

/* Read in a configuration file with user-calibrated
values to determine the conversion from physical to
fractional stroke as well as the volume displaced by
the influence function. */
int alpaoCtrl::parse_calibration_file() //const char * serial, Scalar *max_stroke, Scalar *volume_factor)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    Scalar * calibvals;

    std::string ser = mx::ioutils::toLower(m_serialNumber);
    
    std::string calibpath = m_calibPath + "/" + ser + "_userconfig.txt";
    
    // open file
    fp = fopen(calibpath.c_str(), "r");
    if (fp == NULL)
    {
       log<text_log>("Could not read configuration file at " + calibpath, logPrio::LOG_ERROR);
        return -1;
    }

    calibvals = (Scalar*) malloc(2*sizeof(Scalar));
    int idx = 0;
    while ((read = getline(&line, &len, fp)) != -1)
    {
        // grab first value from each line
        calibvals[idx] = strtod(line, NULL);
        idx++;
    }

    fclose(fp);

    // assign stroke and volume factors
    m_max_stroke = calibvals[0];
    m_volume_factor = calibvals[1];

    free(calibvals);
    
    log<text_log>("ALPAO " + m_serialNumber + ": Using stroke and volume calibration from " + calibpath);
    std::cerr << m_max_stroke << " " << m_volume_factor << "\n";
    return 0;
}

int alpaoCtrl::get_actuator_mapping() //const char * serial, int nbAct, int * actuator_mapping)
{
    /* This function closely follows the CFITSIO imstat
    example */

    fitsfile *fptr;  /* FITS file pointer */
    int status = 0;  /* CFITSIO status value MUST be initialized to zero! */
    
    

    // get file path to actuator map
    std::string ser = mx::ioutils::toLower(m_serialNumber);
    
    std::string calibpath = m_calibPath + "/" + ser + "_actuator_mapping.fits";
    
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
      for (fpixel[1] = naxes[1]; fpixel[1] >= 1; fpixel[1]--)
      {  
         /* give starting pixel coordinate and number of pixels to read */
         if (fits_read_pix(fptr, TINT, fpixel, naxes[0],0, pix,0, &status))
            break;   /* jump out of loop on error */

         // get indices of active actuators in order
         for (int ii = 0; ii < naxes[0]; ii++) {
           if (pix[ii] > 0) {
                m_actuator_mapping[ij] = (fpixel[1]-1) * naxes[0] + ii;
                ij++;
           }
         }
      }
      fits_close_file(fptr, &status);
      
      free(pix);
    }

    if (status)  {
        fits_report_error(stderr, status); /* print any error message */
    }

    

    log<text_log>("ALPAO " + m_serialNumber + ": Using actuator mapping from " + calibpath);
    return 0;
}

} //namespace app
} //namespace MagAOX

#endif //alpaoCtrl_hpp
