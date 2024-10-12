/** \file psfAcq.hpp
  * \brief The MagAO-X PSF Fitter application header
  *
  * \ingroup psfAcq_files
  */

#ifndef psfAcq_hpp
#define psfAcq_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/fit/fitGaussian.hpp>
#include <mx/improc/imageFilters.hpp>

/** \defgroup psfAcq 
  * \brief The MagAO-X PSF fitter.
  *
  * <a href="../handbook/operating/software/apps/psfAcq.html">Application Documentation</a>
  *
  * \ingroup apps
  *
  */

/** \defgroup psfAcq_files
  * \ingroup psfAcq
  */

namespace MagAOX
{
namespace app
{

struct darkShmimT 
{
   static std::string configSection()
   {
      return "darkShmim";
   };
   
   static std::string indiPrefix()
   {
      return "dark";
   };
};


//TESTING
struct Star {
    float x, y;          // Star coordinates
    float max;          // Star brightness
    float fwhm;         // Star FWHM
};
//TESTING


/// The MagAO-X PSF Fitter
/** 
  * \ingroup psfAcq
  */
class psfAcq : public MagAOXApp<true>, public dev::shmimMonitor<psfAcq>, public dev::shmimMonitor<psfAcq,darkShmimT>, 
                                      public dev::frameGrabber<psfAcq>, public dev::telemeter<psfAcq>
{
   //Give the test harness access.
   friend class psfAcq_test;

   friend class dev::shmimMonitor<psfAcq>;
   friend class dev::shmimMonitor<psfAcq,darkShmimT>;
   friend class dev::frameGrabber<psfAcq>;

   friend class dev::telemeter<psfAcq>;

public:
   //The base shmimMonitor type
   typedef dev::shmimMonitor<psfAcq> shmimMonitorT;
   
   typedef dev::shmimMonitor<psfAcq,darkShmimT> darkShmimMonitorT;

   //The base frameGrabber type
   typedef dev::frameGrabber<psfAcq> frameGrabberT;

   //The base telemeter type
   typedef dev::telemeter<psfAcq> telemeterT;

   ///Floating point type in which to do all calculations.
   typedef float realT;
   
   /** \name app::dev Configurations
     *@{
     */
   
   static constexpr bool c_frameGrabber_flippable = false; ///< app:dev config to tell framegrabber these images can not be flipped
   
   ///@}

protected:

    /** \name Configurable Parameters
      *@{
      */
   
    std::string m_fpsSource; ///< Device name for getting fps if time-based averaging is used.  This device should have *.fps.current.

    uint16_t m_fitCircBuffMaxLength {3600}; ///< Maximum length of the latency measurement circular buffers
    float m_fitCircBuffMaxTime {5}; ///< Maximum time of the latency meaurement circular buffers

    float m_fwhmGuess {2};
   ///@}

    mx::improc::eigenImage<float> m_image;
    mx::improc::eigenImage<float> m_sm;

    mx::improc::eigenImage<float> m_dark;
   
    bool m_updated {false};
    float m_x {0};
    float m_y {0};
    int m_max_loops {5}; //default to detecting a max of 5 stars    
    int m_zero_area {8}; //default to zeroing an 8x8 pixel area around stars it finds
    float m_threshold = {7.0}; //how many sigma away from the mean you want to classify a detection, default to 7sigma
    float m_fwhm_threshold = {4.0}; //minumum fwhm to consider something a star

    std::vector<float> m_x_vec = {};
    int m_len_x_vec=1;
    std::vector<float> m_y_vec;
    std::vector<float> m_peak;
    std::vector<float> m_fwhm;
    
    
    bool first_pass = true;  //TESTING
    std::vector<Star> m_detectedStars; // vactor to store all the stars and properties

    float m_dx {0};
    float m_dy {0};

    float m_fps {0};

    mx::sigproc::circularBufferIndex<float, cbIndexT> m_xcb;
    mx::sigproc::circularBufferIndex<float, cbIndexT> m_ycb;

    std::vector<float> m_xcbD;
    std::vector<float> m_ycbD;

    float m_mnx {0};
    float m_rmsx {0};
    float m_mny  {0};
    float m_rmsy {0};

    //Working memory for poke fitting
    mx::math::fit::fitGaussian2Dsym<float> m_gfit;

public:
   /// Default c'tor.
   psfAcq();

   /// D'tor, declared and defined for noexcept.
   ~psfAcq() noexcept;
   
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

   /// Implementation of the FSM for psfAcq.
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

   // shmimMonitor interface:
   int allocate( const dev::shmimT &);
   
   int processImage( void* curr_src,
                     const dev::shmimT &
                    );
   
   // shmimMonitor interface for referenc:
   int allocate( const darkShmimT &);
   
   int processImage( void* curr_src,
                     const darkShmimT &
                   );

protected:

   std::mutex m_imageMutex;

   sem_t m_smSemaphore {0}; ///< Semaphore used to synchronize the fg \thread and the sm thread.

public:

   /** \name dev::frameGrabber interface
     *
     * @{
     */
   
   /// Implementation of the framegrabber configureAcquisition interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int configureAcquisition();
   
   /// Implementation of the framegrabber fps interface
   /**
     * \todo this needs to infer the stream fps and return it
     */  
   float fps()
   {
      return m_fps;
   }
   
   /// Implementation of the framegrabber startAcquisition interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int startAcquisition();
   
   /// Implementation of the framegrabber acquireAndCheckValid interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int acquireAndCheckValid();
   
   /// Implementation of the framegrabber loadImageIntoStream interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int loadImageIntoStream( void * dest  /**< [in] */);
   
   /// Implementation of the framegrabber reconfig interface
   /** 
     * \returns 0 on success
     * \returns -1 on error
     */
   int reconfig();
   
   ///@}

protected:

   /** \name INDI
     * @{
     */ 

/*   //FOR INDI
   std::vector<pcf::IndiProperty> m_indiP_x_pos; // INDI Property for stars x position
   std::vector<pcf::IndiProperty> m_indiP_y_pos; // INDI Property for stars y position
   std::vector<pcf::IndiProperty> m_indiP_brightness; // INDI Property for stars brightness
   std::vector<pcf::IndiProperty> m_indiP_fwhm; // INDI Property for stars fwhm
   //FOR INDI  */
   std::vector<pcf::IndiProperty> m_indiP_star; // INDI Property for stars


   pcf::IndiProperty m_indiP_dx;
   
   INDI_NEWCALLBACK_DECL(psfAcq, m_indiP_dx);
   
   pcf::IndiProperty m_indiP_dy;
   
   INDI_NEWCALLBACK_DECL(psfAcq, m_indiP_dy);

   pcf::IndiProperty m_indiP_fpsSource;
   INDI_SETCALLBACK_DECL(psfAcq, m_indiP_fpsSource);

   ///@}

   /** \name Telemeter Interface
     * 
     * @{
     */ 
   int checkRecordTimes();
   
   int recordTelem( const telem_fgtimings * );

   ///@}
};

inline
psfAcq::psfAcq() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
   darkShmimMonitorT::m_getExistingFirst = true;
   return;
}

inline
psfAcq::~psfAcq() noexcept
{

}

inline
void psfAcq::setupConfig()
{
   shmimMonitorT::setupConfig(config);
   darkShmimMonitorT::setupConfig(config);
   frameGrabberT::setupConfig(config);
   telemeterT::setupConfig(config);
   
   config.add("fitter.fpsSource", "", "fitter.fpsSource", argType::Required, "fitter", "fpsSource", false, "string", "Device name for getting fps if time-based averaging is used.  This device should have *.fps.current.");
   config.add("fitter.max_loops", "", "fitter.max_loops", argType::Required, "fitter", "max_loops", false, "int", "Setting the number of stars to detect in processImage function.");
   config.add("fitter.zero_area", "", "fitter.zero_area", argType::Required, "fitter", "zero_area", false, "int", "Setting the pixel area to zero out after detecting stars in processImage function.");
   config.add("fitter.threshold", "", "fitter.threshold", argType::Required, "fitter", "threshold", false, "float", "setting how many sigma away from the mean you want to classify a detection.");
   config.add("fitter.fwhm_threshold", "", "fitter.fwhm_threshold", argType::Required, "fitter", "fwhm_threshold", false, "float", "minumum fwhm to consider something a star.");

}


inline
int psfAcq::loadConfigImpl( mx::app::appConfigurator & _config )
{
   shmimMonitorT::loadConfig(_config);
   darkShmimMonitorT::loadConfig(_config);

   frameGrabberT::loadConfig(_config);
   telemeterT::loadConfig(_config);

    _config(m_fpsSource, "fitter.fpsSource");
    _config(m_max_loops, "fitter.max_loops"); //Max number of stars to detect in processImage
    _config(m_zero_area, "fitter.zero_area"); //pixel area to zero out in processImage when a star is detected
    _config(m_threshold, "fitter.threshold"); //how many sigma away from the mean you want to classify a detection
    _config(m_fwhm_threshold, "fitter.fwhm_threshold"); //how many sigma away from the mean you want to classify a detection

   return 0;
}

inline
void psfAcq::loadConfig()
{
   loadConfigImpl(config);
}

inline
int psfAcq::appStartup()
{
   if(shmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }
   
   if(darkShmimMonitorT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(sem_init(&m_smSemaphore, 0,0) < 0)
   {
      log<software_critical>({__FILE__, __LINE__, errno,0, "Initializing S.M. semaphore"});
      return -1;
   }

   if(frameGrabberT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(telemeterT::appStartup() < 0)
   {
      return log<software_error,-1>({__FILE__, __LINE__});
   }

   if(m_fpsSource != "")
   {
      REG_INDI_SETPROP(m_indiP_fpsSource, m_fpsSource, std::string("fps"));
   }

   CREATE_REG_INDI_NEW_NUMBERF(m_indiP_dx, "dx", -100, 100, 1e-2, "%0.02f", "", "");
   m_indiP_dx["current"].setValue(m_dx);
   m_indiP_dx["target"].setValue(m_dx);

   CREATE_REG_INDI_NEW_NUMBERF(m_indiP_dy, "dy", -100, 100, 1e-2, "%0.02f", "", "");
   m_indiP_dy["current"].setValue(m_dy);
   m_indiP_dy["target"].setValue(m_dy);

   state(stateCodes::OPERATING);
   
   return 0;
}

inline
int psfAcq::appLogic()
{
   if( shmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }
   
   if( darkShmimMonitorT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( frameGrabberT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }

   if( telemeterT::appLogic() < 0)
   {
      return log<software_error,-1>({__FILE__,__LINE__});
   }


    if( state() == stateCodes::OPERATING && m_xcb.size() > 0 )
    {
        if(m_xcb.size() >= m_xcb.maxEntries())
        {
            cbIndexT refEntry = m_xcb.earliest();
         
            m_xcbD.resize(m_xcb.maxEntries()-1);
            m_ycbD.resize(m_xcb.maxEntries()-1);

            for(size_t n=0; n <= m_atimesD.size(); ++n)
            {
                m_xcbD[n] = m_xcb.at(refEntry,n);
                m_ycbD[n] = m_ycb.at(refEntry,n);
            }
         
            m_mnx = mx::math::vectorMean(m_xcbD);
            m_rmsx = sqrt(mx::math::vectorVariance(m_xcbD, m_mnx));
         
            m_mny = mx::math::vectorMean(m_ycbD);
            m_rmsy = sqrt(mx::math::vectorVariance(m_ycbD, m_mny));

        }
        else
        {
            m_mnx = 0;
            m_rmsx = 0;
            m_mny = 0;
            m_rmsy = 0;
        }
    }
    else
    {
        m_mnx = 0;
        m_rmsx = 0;
        m_mny = 0;
        m_rmsy = 0;
    }


   shmimMonitorT::updateINDI();
   darkShmimMonitorT::updateINDI();
   
   if(frameGrabberT::updateINDI() < 0)
   {
      log<software_error>({__FILE__, __LINE__});
   }

   updateIfChanged(m_indiP_dx, "current", m_dx);
   updateIfChanged(m_indiP_dy, "current", m_dy);

   return 0;
}

inline
int psfAcq::appShutdown()
{
   shmimMonitorT::appShutdown();
   darkShmimMonitorT::appShutdown();
   frameGrabberT::appShutdown();
   telemeterT::appShutdown();

   return 0;
}

inline
int psfAcq::allocate(const dev::shmimT & dummy)
{
    static_cast<void>(dummy);
   
    std::lock_guard<std::mutex> guard(m_imageMutex);

    m_image.resize(shmimMonitorT::m_width, shmimMonitorT::m_height);
    m_image.setZero();

    m_sm.resize(m_image.rows(), m_image.cols());

    if(m_fitCircBuffMaxLength == 0 || m_fitCircBuffMaxTime == 0 || m_fps <= 0)
    {
        m_xcb.maxEntries(0);
        m_ycb.maxEntries(0);
    }
    else 
    {
        //Set up the fit circ. buffs
        cbIndexT cbSz = m_fitCircBuffMaxTime * m_fps;
        if(cbSz > m_fitCircBuffMaxLength) cbSz = m_fitCircBuffMaxLength;
        if(cbSz < 3) cbSz = 3; //Make variance meaningful
        m_xcb.maxEntries(cbSz);
        m_ycb.maxEntries(cbSz);
    }

   m_updated = false;
   return 0;
}
   

//TESTING
// Function to calculate Euclidean distance between two stars
float calculateDistance(float x1, float y1, float x2, float y2) {
   return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}
//TESTING

inline
int psfAcq::processImage( void* curr_src,
                          const dev::shmimT & dummy
                        )
{
    static_cast<void>(dummy);
   
    std::unique_lock<std::mutex> lock(m_imageMutex);
   
    if(m_dark.rows() == m_image.rows() && m_dark.cols() == m_image.cols())
    {
        for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
        {
            m_image.data()[nn] = ((uint16_t*)curr_src) [nn] - m_dark.data()[nn];
        }
    }
    else
    {
        for(unsigned nn=0; nn < shmimMonitorT::m_width*shmimMonitorT::m_height; ++nn)
        {
            m_image.data()[nn] = ((uint16_t*)curr_src) [nn];
        }
    }

    lock.unlock();

    // fit
    //std::cerr << m_image.sum() << "\n";

    static size_t old_x_size = 0;
    float max;
    //int max_loops=5;
    int x=0;
    int y=0;
    int N_loops=0;
    m_x_vec = {};
    m_y_vec = {};
    m_peak = {};
    m_fwhm = {};

    max = m_image.maxCoeff(&x, &y);

    //mx::improc::medianSmooth(m_sm, x, y, max, m_image, 3);

    //mx::improc::imageCenterOfLight(m_x, m_y, m_image);
    //std::cerr << __LINE__ << std::endl;


    /*if(fabs(m_x-x) > 2 || fabs(m_y-y) > 2)
    {
        std::cerr << "skip frame\n";
        return 0;
    }*/

    
    eigenImage<float> llcorn = m_image.block(0,0,32,32); //calc std dev of 32x32 block in lower left corner
    float mean = llcorn.mean(); // Calculate the mean
    float variance = (llcorn.array() - mean).square().sum() / (llcorn.size()); //calculate variance
    float stddev = std::sqrt(variance); // Calculate the standard deviation
    float z_score = (max - mean) / stddev; //how many std dev away from mean 

    float fwhm = mx::math::func::sigma2fwhm(m_fwhmGuess); //getting intial fwhm before entering while loop    

    std::size_t numStars = m_detectedStars.size();
   if (numStars == 0){ //TESTING This runs when the vector of stars is empty (usually the first time)

    while ((z_score > m_threshold) && (fwhm > m_fwhm_threshold) && (N_loops < m_max_loops)) { //m_max_loops, m_fwhm_threshold, and m_threshold are configurable variables
        m_gfit.set_itmax(1000); 
        //add if statement in case it goes out of bounds, adjust x and y to be in image
        // m_zero_area is used to zero out the pixel array once a star is detected but can also be used to set up a sub image around the max pixel
        if (x <= m_zero_area){
          x = m_zero_area;
        }
        if (x >= (m_image.rows() - m_zero_area)){
          x = m_image.rows() - m_zero_area;
        }
        if (y <= m_zero_area){
          y = m_zero_area;
        }
        if (y >= (m_image.cols() - m_zero_area)){
          y = m_image.cols() - m_zero_area;
        }
        eigenImage<float> subImage = m_image.block(x-m_zero_area, y-m_zero_area, m_zero_area*2, m_zero_area*2); //set m_image to subImage to speed up gaussian, x,y is position of max pixel
        m_gfit.setArray(subImage.data(), subImage.rows(), subImage.cols());
        //m_gfit.setArray(m_image.data(), m_image.rows(), m_image.cols());
        m_gfit.setGuess(0, max, x, y, mx::math::func::sigma2fwhm(m_fwhmGuess));
        m_gfit.fit();
        m_x = m_gfit.x0();
        m_y = m_gfit.y0();
        m_x_vec.push_back(m_gfit.x0()); // add star to the vector 
        m_y_vec.push_back(m_gfit.y0()); // add star to the vector
        m_peak.push_back(max); // add max pixel value to vector
        m_fwhm.push_back(fwhm); // add max pixel value to vector   
        float fwhm = mx::math::func::sigma2fwhm(m_fwhmGuess);

        //TESTING
        Star newStar;
        newStar.x = m_x;  // Adding attributes to the new star 
        newStar.y = m_y;
        newStar.max = max;
        newStar.fwhm = fwhm;
        m_detectedStars.push_back(newStar);    
        //TESTING

        int x_value = static_cast<int>(m_x); // convert m_x to an int so we can 0 out a rectangular area around the detected star
        int y_value = static_cast<int>(m_y);
        for (int i = x_value; i <= (x_value + m_zero_area); i++) { //zeroing out area around the star centered at m_x and m_y(8x8 pixel area)
            for (int j = y_value; j <= (y_value + m_zero_area); j++) {
                m_image(i-(m_zero_area/2), j-(m_zero_area/2)) = 0; // m_zero_area is defaulted to 8 to zero out a pixel array around the star
            }
        }

        max = m_image.maxCoeff(&x, &y);
        N_loops = N_loops + 1;
        z_score = (max - mean) / stddev;
    }
    
    //first_pass = false; //TESTING
   }


//TESTING

   else{
   int track = 0;
   // In here is where we track the stars using cross correlation between the first frame and subsequent frames
   while ((z_score > m_threshold) && (fwhm > m_fwhm_threshold) && (N_loops < m_max_loops)) { //m_max_loops, m_fwhm_threshold, and m_threshold are configurable variables
        std::cout << "This is the " << track << " time through the loop" << std::endl;
        track++;
        m_gfit.set_itmax(1000); 
        //add if statement in case it goes out of bounds, adjust x and y to be in image
        // m_zero_area is used to zero out the pixel array once a star is detected but can also be used to set up a sub image around the max pixel
        if (x <= m_zero_area){
          x = m_zero_area;
        }
        if (x >= (m_image.rows() - m_zero_area)){
          x = m_image.rows() - m_zero_area;
        }
        if (y <= m_zero_area){
          y = m_zero_area;
        }
        if (y >= (m_image.cols() - m_zero_area)){
          y = m_image.cols() - m_zero_area;
        }
        eigenImage<float> subImage = m_image.block(x-m_zero_area, y-m_zero_area, m_zero_area*2, m_zero_area*2); //set m_image to subImage to speed up gaussian, x,y is position of max pixel
        m_gfit.setArray(subImage.data(), subImage.rows(), subImage.cols());
        m_gfit.setGuess(0, max, x, y, mx::math::func::sigma2fwhm(m_fwhmGuess));
        float fwhm = mx::math::func::sigma2fwhm(m_fwhmGuess);
        m_gfit.fit();
        m_x = m_gfit.x0();
        m_y = m_gfit.y0();

        int x_value = static_cast<int>(m_x); // convert m_x to an int so we can 0 out a rectangular area around the detected star
        int y_value = static_cast<int>(m_y);
        for (int i = x_value; i <= (x_value + m_zero_area); i++) { //zeroing out area around the star centered at m_x and m_y(8x8 pixel area)
            for (int j = y_value; j <= (y_value + m_zero_area); j++) {
                m_image(i-(m_zero_area/2), j-(m_zero_area/2)) = 0; // m_zero_area is defaulted to 8 to zero out a pixel array around the star
            }
        }

        // This simple for loop calculate the distance from the detected star to the cloest star already in the list and updates the values 
        float closest_dist = 1000.0; //distance between new stars should be a small positive number so this updates
        int n = 0;
        int star_number = 0;
        for (const Star& star : m_detectedStars) {
            float dist = calculateDistance(star.x, star.y, m_x, m_y);
            if (dist < closest_dist){
               closest_dist = dist;
               star_number = n;
            }
            n++;
        }

        //star_number keeps track of what star in the vector was the closest and updates its values
        std::cout << "Star " << star_number << " New X-value: " << m_detectedStars[star_number].x << std::endl;
        m_detectedStars[star_number].x = m_x;
        m_detectedStars[star_number].y = m_y;
        m_detectedStars[star_number].max = max;
        m_detectedStars[star_number].fwhm = fwhm;
        
        max = m_image.maxCoeff(&x, &y);
        N_loops = N_loops + 1;
        z_score = (max - mean) / stddev;
        /*
        std::cout << "Z-score:          " << z_score << std::endl;
        std::cout << "FWHM:             " << fwhm << std::endl;
        std::cout << "N_loops:          " << N_loops << std::endl;
        std::cout << "m_threshold:      " << m_threshold << std::endl;
        std::cout << "m_fwhm_threshold: " << m_fwhm_threshold << std::endl;
        std::cout << "m_max_loops:      " << m_max_loops << std::endl;
        */
        }
    }


//TESTING


    if (old_x_size != m_x_vec.size()){ //if statement to update only if number of stars changes  
      if (m_indiP_star.size() > 0)
      {
         for (size_t n = 0; n < m_indiP_star.size(); ++n)
         {
            // Delete old Y properties
            if (m_indiDriver) m_indiDriver->sendDelProperty(m_indiP_star[n]);
            if (!m_indiNewCallBacks.erase(m_indiP_star[n].createUniqueKey()))
            {
                  log<software_error>({__FILE__, __LINE__, "Failed to erase " + m_indiP_star[n].createUniqueKey()});
            }
         }
         m_indiP_star.clear();  // Clear the Y position container
      }
      //set and senSetProperty should run every time. Not Sure about this
      //problem of two stars equal brightness, first pass thru use max brightness, then switch to keep track using coordinates
      //set stars to be ~6 sigma (in config then restart)
      size_t starCount = m_detectedStars.size();
      m_indiP_star.resize(starCount);


      // Create and register new properties for X and Y positions
      for (size_t n = 0; n < starCount; ++n){

         std::string starPrefix = "star_" + std::to_string(n);

         createROIndiNumber( m_indiP_star[n], starPrefix, "Star " + std::to_string(n) + " Properties", "Star Acq");
         m_indiP_star[n].add (pcf::IndiElement("x"));
         m_indiP_star[n]["x"].set(m_detectedStars[n].x);
         m_indiP_star[n].add (pcf::IndiElement("y"));
         m_indiP_star[n]["y"].set(m_detectedStars[n].y); 
         m_indiP_star[n].add (pcf::IndiElement("peak"));
         m_indiP_star[n]["peak"].set(m_detectedStars[n].max); 
         m_indiP_star[n].add (pcf::IndiElement("fwhm"));
         m_indiP_star[n]["fwhm"].set(m_detectedStars[n].fwhm);   
         registerIndiPropertyReadOnly(m_indiP_star[n]);
         if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_star[n]);

/*
         createROIndiNumber( m_indiP_star[n], starPrefix, "Star " + std::to_string(n) + " Properties", "Star Acq");
         m_indiP_star[n].add (pcf::IndiElement("x"));
         m_indiP_star[n]["x"].set(m_x_vec[n]);
         m_indiP_star[n].add (pcf::IndiElement("y"));
         m_indiP_star[n]["y"].set(m_y_vec[n]); 
         m_indiP_star[n].add (pcf::IndiElement("peak"));
         m_indiP_star[n]["peak"].set(m_peak[n]); 
         m_indiP_star[n].add (pcf::IndiElement("fwhm"));
         m_indiP_star[n]["fwhm"].set(m_fwhm[n]);   
         registerIndiPropertyReadOnly(m_indiP_star[n]);
         if(m_indiDriver) m_indiDriver->sendSetProperty (m_indiP_star[n]);
*/
      }
    }   
      //FOR INDI
    
    if (old_x_size != m_x_vec.size()){
        old_x_size = m_x_vec.size();
        m_reconfig=true;
    }
    m_len_x_vec = m_x_vec.size();

    m_updated = true;

    //signal framegrabber
    //Now tell the f.g. to get going
    if(sem_post(&m_smSemaphore) < 0)
    {
        log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
        return -1;
    }
         
    //Update the latency circ. buffs
    if(m_xcb.maxEntries()  >  0)
    {
        m_xcb.nextEntry(m_x);
        m_ycb.nextEntry(m_y);
    }

    return 0;
}
   
inline
int psfAcq::allocate(const darkShmimT & dummy)
{
    static_cast<void>(dummy);
   
    std::lock_guard<std::mutex> guard(m_imageMutex);
   
    if(darkShmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT)
    {
        return log<software_error,-1>({__FILE__, __LINE__, "dark is not float"});
    }

    m_dark.resize( darkShmimMonitorT::m_width, darkShmimMonitorT::m_height );
    m_dark.setZero();

    return 0;
}   
   
inline
int psfAcq::processImage( void* curr_src,
                          const darkShmimT & dummy
                        )
{
    static_cast<void>(dummy);
   
    std::unique_lock<std::mutex> lock(m_imageMutex);

    for(unsigned nn=0; nn < darkShmimMonitorT::m_width*darkShmimMonitorT::m_height; ++nn)
    {
        m_dark.data()[nn] += ((float*)curr_src) [nn];
    }

    lock.unlock();

    log<text_log>("dark updated", logPrio::LOG_INFO);

    return 0;
}

inline
int psfAcq::configureAcquisition()
{
   
   frameGrabberT::m_width = 2;
   frameGrabberT::m_height = m_x_vec.size() + 2; //Ask Jared if this is fine, need this to != 0 else throws error is Line 1040 (/usr/local/include/eigen3/Eigen/src/Core/DenseCoeffsBase.h)
   frameGrabberT::m_dataType = _DATATYPE_FLOAT;
   
   return 0;
}

inline
int psfAcq::startAcquisition()
{
   return 0;
}

inline
int psfAcq::acquireAndCheckValid()
{
   timespec ts;
         
   if(clock_gettime(CLOCK_REALTIME, &ts) < 0)
   {
      log<software_critical>({__FILE__,__LINE__,errno, 0, "clock_gettime"}); 
      return -1;
   }
         
   ts.tv_sec += 1;
        
   if(sem_timedwait(&m_smSemaphore, &ts) == 0)
   {
      if( m_updated )
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


}

inline
int psfAcq::loadImageIntoStream(void * dest)
{
   //std::cout << "Width: " << frameGrabberT::m_width << ", Height: " << frameGrabberT::m_height << std::endl;
   //mx::improc::eigenMap<float> destMap(reinterpret_cast<float*>(dest), frameGrabberT::m_width, frameGrabberT::m_height);
   //Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> destMap(reinterpret_cast<float*>(dest), frameGrabberT::m_width, frameGrabberT::m_height);
   Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> destMap(reinterpret_cast<float*>(dest), m_x_vec.size(), 1+m_y_vec.size()); //1+ keeps it from crashing when only one star is detected
   /*
   std::cout << __LINE__ << std::endl;
   std::cout << "m_x_vec size: " << m_x_vec.size() << std::endl;
   std::cout << "m_y_vec size: " << m_y_vec.size() << std::endl;
   std::cout << "destMap rows: " << destMap.rows() << ", cols: " << destMap.cols() << std::endl;
   std::cout << "destMap size (total elements): " << destMap.size() << std::endl; 
   */
   for (size_t i = 1; i <= m_x_vec.size(); i++){
      // Using destMap to store the x, y values with the respective offsets
      //std::cout << __LINE__ << std::endl;
      destMap(2*i - 2) = m_x_vec[i-1] - m_dx;  // Store x-coordinate
      //std::cout << __LINE__ << std::endl;
      destMap(2*i - 1) = m_y_vec[i-1] - m_dy;  // Store y-coordinate
      //std::cout << __LINE__ << std::endl;
   }
   //std::cout << __LINE__ << std::endl;
   

/*
   ((float *) dest)[0] = m_x - m_dx;
   ((float *) dest)[1] = m_y - m_dy;
*/
   m_updated = false;

   return 0;
}

inline
int psfAcq::reconfig()
{
   return 0;
}

INDI_NEWCALLBACK_DEFN(psfAcq, m_indiP_dx)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_dx.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_dx, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_dx = target;
   
   log<text_log>("set dx = " + std::to_string(m_dx), logPrio::LOG_NOTICE);
   return 0;
}

INDI_NEWCALLBACK_DEFN(psfAcq, m_indiP_dy)(const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getName() != m_indiP_dy.getName())
   {
      log<software_error>({__FILE__,__LINE__, "wrong INDI property received."});
      return -1;
   }
   
   float target;
   
   if( indiTargetUpdate( m_indiP_dy, target, ipRecv, true) < 0)
   {
      log<software_error>({__FILE__,__LINE__});
      return -1;
   }
   
   m_dy = target;
   
   log<text_log>("set dy = " + std::to_string(m_dy), logPrio::LOG_NOTICE);
   return 0;
}

INDI_SETCALLBACK_DEFN( psfAcq, m_indiP_fpsSource )(const pcf::IndiProperty &ipRecv)
{
   if( ipRecv.getName() != m_indiP_fpsSource.getName())
   {
      log<software_error>({__FILE__, __LINE__, "Invalid INDI property."});
      return -1;
   }
   
   if( ipRecv.find("current") != true ) //this isn't valie
   {
      return 0;
   }
   
   std::lock_guard<std::mutex> guard(m_indiMutex);

   realT fps = ipRecv["current"].get<float>();
   
   if(fps != m_fps)
   {
      m_fps = fps;
      shmimMonitorT::m_restart = true;
   }

   return 0;
}

inline
int psfAcq::checkRecordTimes()
{
   return telemeterT::checkRecordTimes(telem_fgtimings());
}
   
inline
int psfAcq::recordTelem( const telem_fgtimings * )
{
   return recordFGTimings(true);
}



} //namespace app
} //namespace MagAOX

#endif //psfAcq_hpp
