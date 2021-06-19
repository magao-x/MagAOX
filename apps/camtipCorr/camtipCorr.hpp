/** \file camtipCorr.hpp
  * \brief The MagAO-X image registrator and Strehl ratio monitor
  *
  * \ingroup app_files
  */

#pragma once

#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <fftw3.h>
#include "reg_functions.hpp" 
#include "cc-functions.hpp"

namespace MagAOX::app
{


   
/** \defgroup camtipCorr Image registrator and Strehl ratio monitor
  * \brief Calculates the movement in images over time and fetches their Strehl ratio
  *
  * \ingroup apps
  *
  */

/** \defgroup camtipCorr_files Image monitoring
  * \ingroup camtipCorr
  */

/** MagAO-X application to monitor image quality.
  *
  * \ingroup camtipCorr
  * 
  */
class camtipCorr : public MagAOXApp<false>, public dev::shmimMonitor<camtipCorr>
{

   friend class dev::shmimMonitor<camtipCorr>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<camtipCorr> shmimMonitorT;
      
   ///Floating point type in which to do all calculations.
   typedef float realT;
   
protected:

   size_t m_rows {0}, m_cols {0};
   size_t m_sz {5};
   double m_data[3] {0, 0, 0};

   double*        m_input      {nullptr};
   double*        m_cc_array   {nullptr};
   fftw_complex*  m_output     {nullptr};
   fftw_complex*  m_image0_fft {nullptr};
   fftw_complex*  m_cc_fft     {nullptr};

   fftw_plan m_planF, m_planB;

   std::string m_shmemKey {"camtip"};
   IMAGE m_image;    // input stream

   int m_semNum {0};
   std::string m_resChannel {"coord-shifts"};
   uint32_t m_imsize[3];
   IMAGE m_shifts;   // output stream

   bool m_template  {true};
   bool m_imOpened  {false};
   bool m_imRestart {false};

   uint8_t m_dataType;
   size_t m_typeSize; 

   size_t m_xctr;
   size_t m_yctr;

public:
   /// Default c'tor.
   camtipCorr();

   /// D'tor, declared and defined for noexcept.   ~camtipCorr() noexcept
   ~camtipCorr() noexcept
   {}

   virtual void setupConfig();

   /// Implementation of loadConfig logic, separated for testing.
   /** This is called by loadConfig().
     */
   int loadConfigImpl( mx::app::appConfigurator & _config /**< [in] an application configuration from which to load values*/);

   virtual void loadConfig();

   /// Startup function
   /**
     *
     */
   virtual int appStartup();

   /// Implementation of the FSM for camtipCorr.
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

   
   int allocate( const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
   int processImage( void * curr_src,          ///< [in] pointer to start of current frame.
                     const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                   );
};


//===============================/
//          FUNCTIONS            /
//===============================/
inline
camtipCorr::camtipCorr() 
: MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{}

inline
void camtipCorr::setupConfig()
{
   shmimMonitorT::setupConfig(config);
}


inline
int camtipCorr::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   
   return 0;
}

inline
void camtipCorr::loadConfig()
{
   loadConfigImpl(config);
}

inline
int camtipCorr::appStartup()
{
  
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
  
   state(stateCodes::OPERATING);  
   return 0;
}

inline
int camtipCorr::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   
   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }
   
   
   return 0;
}

inline
int camtipCorr::appShutdown()
{
   shmimMonitorT::appShutdown(); 
   
   return 0;
}

inline
int camtipCorr::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   
   if(m_imOpened)
   {
      ImageStreamIO_closeIm(&m_image);
   }
   
   m_imOpened = false;
   m_imRestart = false;
   if( ImageStreamIO_openIm(&m_image, m_shmemKey.c_str()) == 0)
   {
      if(m_image.md[0].sem < 10) 
      {
            ImageStreamIO_closeIm(&m_image);
      }
      else
      {
         m_imOpened = true;
      }
   }
      
   if(!m_imOpened) {

      log<software_error>({__FILE__, __LINE__, m_resChannel + " not opened."});
      return -1;

   } else {
   
      m_imsize[0] = 3;
      m_imsize[1] = 1;
      m_imsize[2] = 1;
    
      std::cerr << m_imsize[0] << " " << m_imsize[1] << " " << m_imsize[2] << "\n";
    
      ImageStreamIO_createIm(&m_shifts, m_resChannel.c_str(), 1, 
                          m_imsize, _DATATYPE_DOUBLE, 1, 0); 
      // need code to determine whether m_shifts is open or not
      m_rows = m_image.md[0].size[0];
      m_cols = m_image.md[0].size[1];
   
      m_input    = (double *)fftw_malloc(m_rows * m_cols * sizeof(double));
      m_cc_array = (double *)fftw_malloc(m_rows * m_cols * sizeof(double));

      m_output = (fftw_complex *)fftw_malloc(m_rows * (m_cols / 2 + 1) * sizeof(fftw_complex));
      m_image0_fft = (fftw_complex *)fftw_malloc(m_rows * (m_cols / 2 + 1) * sizeof(fftw_complex));
      m_cc_fft = (fftw_complex *)fftw_malloc(m_rows * (m_cols / 2 + 1) * sizeof(fftw_complex));

      m_planF = fftw_plan_dft_r2c_2d(m_rows, m_cols, m_input, m_output, FFTW_MEASURE); // use wisdom here
      m_planB = fftw_plan_dft_c2r_2d(m_rows, m_cols, m_cc_fft, m_cc_array, FFTW_MEASURE);

      memset(m_cc_fft, 0, m_rows * (m_cols / 2 + 1) * sizeof(fftw_complex)); 

      m_dataType = m_image.md->datatype;
      m_typeSize = ImageStreamIO_typesize(m_dataType);
      
      log<text_log>( "Opened " + m_resChannel + " " + std::to_string(m_cols) + " x " 
                               + std::to_string(m_rows) + " with data type: " 
                               + std::to_string(m_dataType)); 
       
   }
  
   ///\todo size checks here.
   
   //state(stateCodes::OPERATING);
   return 0;
}

inline
int camtipCorr::processImage( void * curr_src, 
                              const dev::shmimT & dummy 
                            )
{
   static_cast<void>(dummy); //be unused
   
   switch(m_template) { 
   case true:
      copy_image(m_input, &m_image);
      fftw_execute(m_planF);
      image0_fft_fill(m_image0_fft, m_output, m_rows, m_cols / 2 + 1);
      memset(m_cc_fft, 0, m_rows * (m_cols / 2 + 1) * sizeof(fftw_complex));
      m_template = false;
      break;

   case false:
      copy_image(m_input, &m_image);
      fftw_execute(m_planF);
      point_multiply(m_image0_fft, m_output, m_cc_fft, m_rows, m_cols / 2 + 1);
      fftw_execute(m_planB); 
      memset(m_cc_fft, 0, (m_cols / 2 + 1) * m_rows * sizeof(fftw_complex));
      GaussFit(m_rows, m_cols, m_cc_array, m_sz, m_data);
      m_data[2] = getStrehlMod(m_input, m_rows, m_cols, m_xctr, m_yctr) / 1; //FPK_FTOT;
      std::cout << "SR: " << m_data[1] << "\n";

      m_shifts.md[0].write=1;
      memcpy(m_shifts.array.D, m_data, 3 * sizeof(double)); 
      std::cout << "post-memcpy\n";
      m_shifts.md[0].cnt0++;               
      m_shifts.md[0].write=0;             
      ImageStreamIO_sempost(&m_shifts,-1); 
      std::cout << "False end\n"; 
      break;
   } 
   
   return 0;
}


} // namespace magAOX::app 
