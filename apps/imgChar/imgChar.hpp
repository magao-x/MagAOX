/** \file imgChar.hpp
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

// Pointer to linked theoretical Strehl amplitude values
extern double _binary_sa_dat_start;
extern double _binary_sa_dat_end;


namespace MagAOX::app
{


   
/** \defgroup  imgChar Image registrator and Strehl ratio monitor
  * \brief  Calculates the tilt in the x and y directions, as well as the
  *         strehl ratio of the image
  *
  * \ingroup  apps
  *
  */

/** \defgroup imgChar_files Image monitoring
  * \ingroup imgChar
  */

/** MagAO-X application to monitor image quality.
  *
  * \ingroup imgChar
  * 
  */
class imgChar : public MagAOXApp<true>, 
                public dev::shmimMonitor<imgChar>,
                public dev::frameGrabber<imgChar>
{

   friend class dev::shmimMonitor<imgChar>;
   
   //The base shmimMonitor type
   typedef dev::shmimMonitor<imgChar> shmimMonitorT;
     
   friend class dev::frameGrabber<imgChar>;

   //The base frameGrabber type
   typedef dev::frameGrabber<imgChar> frameGrabberT;

   //Datatypes 
   typedef double realT;
   
   typedef fftw_complex complexT;


   protected:

      size_t m_rows {0}, m_cols {0};
      size_t m_sz {5};
      realT m_data[3]; // [0] = current y-shift
                       // [1] = current x-shift 
                       // [2] = current Strehl ratio

      realT*         m_input      {nullptr};
      realT*         m_cc_array   {nullptr};
      fftw_complex*  m_output     {nullptr};
      fftw_complex*  m_image0_fft {nullptr};
      fftw_complex*  m_cc_fft     {nullptr};

      fftw_plan m_planF, m_planB;

      bool m_template  {true};

      uint8_t m_dataType;
      size_t  m_typeSize; 

      size_t m_xctr;
      size_t m_yctr;

   public:
      /// Default c'tor.
      imgChar();

      /// D'tor, declared and defined for noexcept.   ~imgChar() noexcept
      ~imgChar() noexcept
      {}

      virtual void setupConfig();

      /// Implementation of loadConfig logic, separated for testing.
      /** This is called by loadConfig().
        */
      int loadConfigImpl(mx::app::appConfigurator& _config /**< [in] an application configuration from which to load values*/);

      virtual void loadConfig();

      /// Startup function
      /**
        *
        */
      virtual int appStartup();

      /// Implementation of the FSM for imgChar.
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

   
      int allocate(const dev::shmimT & dummy /**< [in] tag to differentiate shmimMonitor parents.*/);
   
      int processImage(void * curr_src,          ///< [in] pointer to start of current frame.
                       const dev::shmimT & dummy ///< [in] tag to differentiate shmimMonitor parents.
                      );


   protected: //frameGrabber functionality
      static constexpr bool c_frameGrabber_flippable = false;

      sem_t m_smSemaphore; ///< synchronizes the fg and sm threads

      bool m_update {false};

      float fps();

      int configureAcquisition(); 
      int startAcquisition();
      int acquireAndCheckValid();
      int loadImageIntoStream(void * dest);
      int reconfig();

   protected:
      pcf::IndiProperty m_indiP_shifts;
      
      realT m_rx      {0};
      realT m_ry      {0};
      realT m_rstrehl {0};

      realT m_xshiftRMS  {0};
      realT m_yshiftRMS  {0};
      realT m_strehlMean {0};
      realT m_strehlRMS  {0};
      
      uint64_t n;

      pcf::IndiProperty m_indiP_modRadius;
      float m_modRadius {0};
      INDI_SETCALLBACK_DECL( imgChar, m_indiP_modRadius);

      pcf::IndiProperty m_indiP_fps;
      float m_fps {0};
      INDI_SETCALLBACK_DECL( imgChar, m_indiP_fps);

   private:
      double* sa_ptr;
};



// =============================== //
//            FUNCTIONS            //
// =============================== //
inline
imgChar::imgChar() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   sa_ptr = &_binary_sa_dat_start;
}



inline
void imgChar::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   frameGrabberT::setupConfig(config);
}



inline
int imgChar::loadConfigImpl( mx::app::appConfigurator & _config )
{
   
   shmimMonitorT::loadConfig(_config);
   return 0;
}



inline
void imgChar::loadConfig()
{
   loadConfigImpl(config);
   frameGrabberT::loadConfig(config); 
}



inline
int imgChar::appStartup()
{
  
   if (shmimMonitorT::appStartup() < 0)
      return log<software_error, -1>({__FILE__, __LINE__});

   if (frameGrabberT::appStartup() < 0) 
      return log<software_error, -1>({__FILE__, __LINE__});


   if(sem_init(&m_smSemaphore, 0, 0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, 
                              "Initializing S.M. semaphore."});
      return -1;
   }

   createROIndiNumber( m_indiP_shifts, "Shifts", "Shift [pixels]");

   indi::addNumberElement<realT>( m_indiP_shifts, "x-rms", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "y-rms", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-mean", -20, 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-rms", -20, 120., 0, "%0.2f");

   registerIndiPropertyReadOnly(m_indiP_shifts);

   REG_INDI_SETPROP( m_indiP_fps, "camtip", "fps");
   REG_INDI_SETPROP( m_indiP_modRadius, "modwfs", "modRadius");
 
   state(stateCodes::OPERATING);  

   return 0;
}



inline
int imgChar::appLogic()
{
   if (shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
 
   if(frameGrabberT::appLogic() < 0)
   {
      return log<software_error, -1>({__FILE__, __LINE__});
   }  

   std::unique_lock<std::mutex> lock(m_indiMutex);
   
   if(shmimMonitorT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   if(frameGrabberT::updateINDI() < 0)
   {

      log<software_error>({__FILE__, __LINE__});
      state(stateCodes::ERROR);

      return 0;
   }

   updateIfChanged(m_indiP_shifts, "x-rms", m_xshiftRMS);
   updateIfChanged(m_indiP_shifts, "y-rms", m_yshiftRMS);
   updateIfChanged(m_indiP_shifts, "strehl-mean", m_strehlMean);
   updateIfChanged(m_indiP_shifts, "strehl-rms", m_strehlRMS);
   
   return 0;
}



inline
int imgChar::appShutdown() 
{

   shmimMonitorT::appShutdown(); 
   frameGrabberT::appShutdown();
 
   return 0;
}



inline
int imgChar::allocate(const dev::shmimT & dummy)
{
   static_cast<void>(dummy);

      // setup the FFT data structures 
      m_rows = shmimMonitorT::m_height;
      m_cols = shmimMonitorT::m_width;
      size_t realArrSize {m_rows * m_cols * sizeof(realT)};
      size_t fftArrSize {m_rows * (m_cols / 2 + 1) * sizeof(complexT)};
   
      m_input      = (realT *)fftw_malloc(realArrSize);
      m_cc_array   = (realT *)fftw_malloc(realArrSize);

      m_output     = (complexT *)fftw_malloc(fftArrSize);
      m_image0_fft = (complexT *)fftw_malloc(fftArrSize);
      m_cc_fft     = (complexT *)fftw_malloc(fftArrSize);

      m_planF = fftw_plan_dft_r2c_2d(
                  m_rows, m_cols, m_input, m_output, FFTW_MEASURE);

      m_planB = fftw_plan_dft_c2r_2d(
                  m_rows, m_cols, m_cc_fft, m_cc_array, FFTW_MEASURE);

      memset(m_cc_fft, 0, fftArrSize); 

      m_dataType = shmimMonitorT::m_dataType;
      m_typeSize = ImageStreamIO_typesize(m_dataType);

      m_xshiftRMS  = 0;
      m_yshiftRMS  = 0;
      m_strehlMean = 0;
      m_strehlRMS  = 0;

      m_rx      = 0;
      m_ry      = 0;
      m_rstrehl = 0;
      n         = 1;      
  
   return 0;
}



inline
int imgChar::processImage(void * curr_src, const dev::shmimT & dummy)
{
   static_cast<void>(dummy); //be unused
   m_update = false;

   size_t memSz = m_rows * (m_cols / 2 + 1) * sizeof(complexT);  
 
   switch (m_template) 
   { 
         case true:
         copy_image0(m_input, curr_src, m_rows, m_cols, m_dataType, &m_xctr, &m_yctr);
         fftw_execute(m_planF);
         image0_fft_fill(m_image0_fft, m_output, m_rows, m_cols / 2 + 1);
         memset(m_cc_fft, 0, memSz);
         m_template = false;
         break;

         case false:
         copy_image(m_input, curr_src, m_rows, m_cols, m_dataType);
         fftw_execute(m_planF);
         point_multiply(m_image0_fft, m_output, m_cc_fft, m_rows, m_cols/2+1);
         fftw_execute(m_planB); 
         memset(m_cc_fft, 0, memSz);

         GaussFit(m_rows, m_cols, m_cc_array, m_sz, m_data);
         if (m_modRadius == 0) 
         {
            m_data[2] = max(curr_src, m_rows * m_cols, m_dataType); 
            m_data[2] /= sa_ptr[0];
         } 
         else 
         {
            m_data[2] = getStrehlMod(m_input, m_rows, m_cols, m_xctr, m_yctr); 
            m_data[2] /= sa_ptr[(size_t)(40*m_modRadius)];
            std::cout << m_data[2] << "\n";
         }

         m_rx = ((m_data[1] * m_data[1]) + (n - 1) * m_rx) / n;
         m_ry = ((m_data[0] * m_data[0]) + (n - 1) * m_ry) / n;
         m_rstrehl = ((m_data[2] * m_data[2]) + (n - 1) * m_rstrehl) / n;
         m_strehlMean = m_strehlMean + ( (m_data[2] - m_strehlMean) / n);

         m_xshiftRMS = sqrt(m_rx);
         m_yshiftRMS = sqrt(m_ry);
         m_strehlRMS = sqrt(m_rstrehl);
         ++n;
         break;
   }
 
   if(sem_post(&m_smSemaphore) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno, 0, 
                              "Error posting to semaphore."});
      return -1;
   }

   m_update = true;
  
   return 0;
}



INDI_SETCALLBACK_DEFN( imgChar, m_indiP_modRadius)(const pcf::IndiProperty &ipRecv)
{
   if (ipRecv.getDevice() != m_indiP_modRadius.getDevice() || ipRecv.getName() != m_indiP_modRadius.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if (ipRecv.find("current") != true )
   {
      return 0;
   }

   m_indiP_modRadius = ipRecv;
   
   m_modRadius = ipRecv["current"].get<float>();
   
   return 0;
}



INDI_SETCALLBACK_DEFN( imgChar, m_indiP_fps)(const pcf::IndiProperty &ipRecv)`
{
   if (ipRecv.getDevice() != m_indiP_fps.getDevice() || ipRecv.getName() != m_indiP_fps.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }

   if (ipRecv.find("current") != true)
   {
      return 0;
   }

   m_indiP_fps = ipRecv;

   m_fps = ipRecv["current"].get<float>();

   return 0;
}



// ============================ //
//    frameGrabber functions    //
// ============================ //
inline
float imgChar::fps()
{
   return m_fps;
}



inline
int imgChar::configureAcquisition()
{
   if (shmimMonitorT::m_width == 0
       || shmimMonitorT::m_height == 0
       || shmimMonitorT::m_dataType == 0)
   {
      sleep(1);
      return -1;
   }
   
   frameGrabberT::m_width = 3;
   frameGrabberT::m_height = 1;
   frameGrabberT::m_dataType = _DATATYPE_DOUBLE;
   
   std::cerr << "shmimMonitorT::m_dataType: " 
             << (int) shmimMonitorT::m_dataType 
             << "\n";
   std::cerr << "frameGrabberT::m_dataType: " 
             << (int) frameGrabberT::m_dataType 
             << "\n";

   return 0;
}


inline
int imgChar::startAcquisition()
{
   state(stateCodes::OPERATING); 
   return 0;
}


inline
int imgChar::acquireAndCheckValid()
{
   timespec ts;
         
   if (clock_gettime(CLOCK_REALTIME, &ts) < 0) {

      log<software_critical>({__FILE__, __LINE__, errno, 0, "clock_gettime"});
      return -1;
   }
         
   ts.tv_sec += 1;
        
   if (sem_timedwait(&m_smSemaphore, &ts) == 0)
   {
      if (m_update)
      {
         clock_gettime(CLOCK_REALTIME, &m_currImageTimestamp);
         return 0;
      }
      else
      {
         return 1;
      }
   }
   else
   {
      return 1;
   }

   return -1;
}


inline
int imgChar::loadImageIntoStream(void * dest)
{
   size_t memSz = frameGrabberT::m_width 
                * frameGrabberT::m_height 
                * frameGrabberT::m_typeSize;

   memcpy(dest, m_data, memSz); 
   return 0;
}


inline
int imgChar::reconfig()
{
   return 0;
}

} // namespace magAOX::app 
