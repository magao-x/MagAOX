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

      uint64_t idx_1s;
      uint64_t idx_10buf;

      std::vector<realT> x_1sec; // 1-second long buffer of x-shifts^2
      std::vector<realT> y_1sec; // 1-second long buffer of y-shifts^2
      std::vector<realT> s_1sec; // 1-second long buffer of Strehl ratios

      std::array<realT, 10> x_10_1s;  // 10 element long buffer holding 1 second rms shifts
      std::array<realT, 10> y_10_1s;  // 10 element long buffer holding 1 second rms shifts
      std::array<realT, 10> s_10_1sR; // 10 element long buffer holding 1 second rms Strehl
      std::array<realT, 10> s_10_1sM; // 10 element long buffer holding 1 second mean Strehl

      realT m_xshiftRMS_1  {0};
      realT m_yshiftRMS_1  {0};
      realT m_strehlMean_1 {0};
      realT m_strehlRMS_1  {0};
 
      realT m_xshiftRMS_5  {0};
      realT m_yshiftRMS_5  {0};
      realT m_strehlMean_5 {0};
      realT m_strehlRMS_5  {0};
 
      realT m_xshiftRMS_10  {0};
      realT m_yshiftRMS_10  {0};
      realT m_strehlMean_10 {0};
      realT m_strehlRMS_10  {0};
      
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

   createROIndiNumber( m_indiP_shifts, "shifts", "Shift [pixels]");

   indi::addNumberElement<realT>( m_indiP_shifts, "x-rms1", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "y-rms1", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-mean1", -20, 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-rms1", -20, 120., 0, "%0.2f");

   indi::addNumberElement<realT>( m_indiP_shifts, "x-rms5", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "y-rms5", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-mean5", -20, 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-rms5", -20, 120., 0, "%0.2f");

   indi::addNumberElement<realT>( m_indiP_shifts, "x-rms10", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "y-rms10", -20., 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-mean10", -20, 120., 0, "%0.2f");
   indi::addNumberElement<realT>( m_indiP_shifts, "strehl-rms10", -20, 120., 0, "%0.2f");

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

   m_xshiftRMS_1  = 0;
   m_yshiftRMS_1  = 0;
   m_strehlMean_1 = 0;
   m_strehlRMS_1  = 0;

   for (size_t i{0}; i < (size_t)fps(); ++i)
   {
      m_xshiftRMS_1  += x_1sec[i];
      m_yshiftRMS_1  += y_1sec[i];
      m_strehlMean_1 += s_1sec[i];
      m_strehlRMS_1  += s_1sec[i] * s_1sec[i];
   }

   m_xshiftRMS_1  /= (size_t)fps();
   m_yshiftRMS_1  /= (size_t)fps();
   m_strehlMean_1 /= (size_t)fps();
   m_strehlRMS_1  /= (size_t)fps();

   x_10_1s[idx_10buf]  = m_xshiftRMS_1;
   y_10_1s[idx_10buf]  = m_yshiftRMS_1;
   s_10_1sR[idx_10buf] = m_strehlRMS_1;
   s_10_1sM[idx_10buf] = m_strehlMean_1;

   switch (idx_10buf)
   {
      case 4:
         m_xshiftRMS_5  = 0;
         m_yshiftRMS_5  = 0;
         m_strehlMean_5 = 0;
         m_strehlRMS_5  = 0;
     
         for (int i{0}; i < 5; ++i)
         {
            m_xshiftRMS_5  += x_10_1s[i];
            m_yshiftRMS_5  += y_10_1s[i];
            m_strehlMean_5 += s_10_1sM[i];
            m_strehlRMS_5  += s_10_1sR[i];
         }

         m_xshiftRMS_5  = sqrt(m_xshiftRMS_5 / 5);
         m_yshiftRMS_5  = sqrt(m_yshiftRMS_5 / 5);
         m_strehlMean_5 = m_strehlMean_5 / 5;
         m_strehlRMS_5  = sqrt(m_strehlRMS_5 / 5);
         break;


      case 9:
         m_xshiftRMS_5  = 0;
         m_yshiftRMS_5  = 0;
         m_strehlMean_5 = 0;
         m_strehlRMS_5  = 0;
 
         m_xshiftRMS_10  = 0;
         m_yshiftRMS_10  = 0;
         m_strehlMean_10 = 0;
         m_strehlRMS_10  = 0;

         for (int i{0}; i < 5; ++i)
         {
            m_xshiftRMS_5  += x_10_1s[i];
            m_yshiftRMS_5  += y_10_1s[i];
            m_strehlMean_5 += s_10_1sM[i];
            m_strehlRMS_5  += s_10_1sR[i];  
         }

         for (int i{0}; i < 10; ++i)
         {
            m_xshiftRMS_10  += x_10_1s[i];
            m_yshiftRMS_10  += y_10_1s[i];
            m_strehlMean_10 += s_10_1sM[i];
            m_strehlRMS_10  += s_10_1sR[i];
         }

         m_xshiftRMS_5  = sqrt(m_xshiftRMS_5 / 5);
         m_yshiftRMS_5  = sqrt(m_yshiftRMS_5 / 5);
         m_strehlMean_5 = m_strehlMean_5 / 5;
         m_strehlRMS_5  = sqrt(m_strehlRMS_5 / 5);

         m_xshiftRMS_10  = sqrt(m_xshiftRMS_10 / 10);
         m_yshiftRMS_10  = sqrt(m_yshiftRMS_10 / 10);
         m_strehlMean_10 = m_strehlMean_10 / 10;
         m_strehlRMS_10  = sqrt(m_strehlRMS_10 / 10);
         break;
     
      default: break;
   }

   m_xshiftRMS_1 = sqrt(m_xshiftRMS_1);
   m_yshiftRMS_1 = sqrt(m_yshiftRMS_1);
   m_strehlRMS_1 = sqrt(m_strehlRMS_1);

   updateIfChanged(m_indiP_shifts, "x-rms1", m_xshiftRMS_1);
   updateIfChanged(m_indiP_shifts, "y-rms1", m_yshiftRMS_1);
   updateIfChanged(m_indiP_shifts, "strehl-mean1", m_strehlMean_1);
   updateIfChanged(m_indiP_shifts, "strehl-rms1", m_strehlRMS_1);
 
   updateIfChanged(m_indiP_shifts, "x-rms5", m_xshiftRMS_5);
   updateIfChanged(m_indiP_shifts, "y-rms5", m_yshiftRMS_5);
   updateIfChanged(m_indiP_shifts, "strehl-mean5", m_strehlMean_5);
   updateIfChanged(m_indiP_shifts, "strehl-rms5", m_strehlRMS_5);  

   updateIfChanged(m_indiP_shifts, "x-rms10", m_xshiftRMS_10);
   updateIfChanged(m_indiP_shifts, "y-rms10", m_yshiftRMS_10);
   updateIfChanged(m_indiP_shifts, "strehl-mean10", m_strehlMean_10);
   updateIfChanged(m_indiP_shifts, "strehl-rms10", m_strehlRMS_10);

   ++idx_10buf;
   idx_10buf = ( idx_10buf % 10 );

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

   x_1sec.resize( (int)fps(), 0);
   y_1sec.resize( (int)fps(), 0);
   s_1sec.resize( (int)fps(), 0);

   m_xshiftRMS_1  = 0;
   m_yshiftRMS_1  = 0;
   m_strehlMean_1 = 0;
   m_strehlRMS_1  = 0;

   m_xshiftRMS_5  = 0;
   m_yshiftRMS_5  = 0;
   m_strehlMean_5 = 0;
   m_strehlRMS_5  = 0;

   m_xshiftRMS_10  = 0;
   m_yshiftRMS_10  = 0;
   m_strehlMean_10 = 0;
   m_strehlRMS_10  = 0;

   idx_1s    = 0;
   idx_10buf = 0;
   m_template = true;

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
            m_data[2] /= sa_ptr[0]; // need to account for wavelength
         } 
         else 
         {
            m_data[2] = getStrehlMod(m_input, m_rows, m_cols, m_xctr, m_yctr); 
            m_data[2] /= sa_ptr[(size_t)(40*m_modRadius)]; // need to account for wavelength, and change the 40
            std::cout << m_data[2] << "\n";
         }

         x_1sec[idx_1s] = m_data[1] * m_data[1];
         y_1sec[idx_1s] = m_data[0] * m_data[0];
         s_1sec[idx_1s] = m_data[2];

         ++idx_1s;
         idx_1s = idx_1s % ( (uint64_t)(fps()) );
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



INDI_SETCALLBACK_DEFN( imgChar, m_indiP_fps)(const pcf::IndiProperty &ipRecv)
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
