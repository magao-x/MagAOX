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

struct Star
{
public:

    float x, y; // Star coordinates
    float max;  // Star brightness
    float fwhm; // Star FWHM
    float seeing; // Star's seeing

private:

    pcf::IndiProperty * m_prop {nullptr};

public:

    pcf::IndiProperty & prop()
    {
        if(m_prop == nullptr)
        {
            throw std::runtime_error("attempt to access nullptr prop");
        }

        return *m_prop;
    }

    void allocate()
    {
        m_prop = new pcf::IndiProperty;
    }

    void deallocate()
    {
        pcf::IndiProperty * mp = m_prop;
        m_prop = nullptr;
        delete mp;
    }


};

/// The MagAO-X PSF Fitter
/**
 * \ingroup psfAcq
 */
class psfAcq : public MagAOXApp<true>,
               public dev::shmimMonitor<psfAcq>,
               public dev::shmimMonitor<psfAcq, darkShmimT>
{
    // Give the test harness access.
    friend class psfAcq_test;

    friend class dev::shmimMonitor<psfAcq>;
    friend class dev::shmimMonitor<psfAcq, darkShmimT>;

  public:
    // The base shmimMonitor type
    typedef dev::shmimMonitor<psfAcq> shmimMonitorT;

    typedef dev::shmimMonitor<psfAcq, darkShmimT> darkShmimMonitorT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

    /** \name app::dev Configurations
     *@{
     */
    ///@}

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_fpsSource; ///< Device name for getting fps if time-based averaging is used.  This device should have
                             ///< *.fps.current.

    uint16_t m_fitCircBuffMaxLength{ 3600 }; ///< Maximum length of the latency measurement circular buffers
    float m_fitCircBuffMaxTime{ 5 };         ///< Maximum time of the latency meaurement circular buffers

    float m_fwhmGuess{ 4 };
    ///@}

    mx::improc::eigenImage<float> m_image;
    mx::improc::eigenImage<float> m_sm;

    mx::improc::eigenImage<float> m_dark;

    double m_acqQuitTime {0};
    double m_acqPauseTime{2};

    int m_current_acq_star {-1};
    int m_temp_acq_star {-1};
    bool m_updated{ false };
    float m_x{ 0 };
    float m_y{ 0 };
    int m_max_loops{ 5 };        // default to detecting a max of 5 stars
    int m_zero_area{ 8 };        // default to zeroing an 8x8 pixel area around stars it finds
    float m_threshold = { 7.0 }; // how many sigma away from the mean you want to classify a detection, default to
                                 // 7sigma
    float m_fwhm_threshold = { 4.0 }; // minumum fwhm to consider something a star
    float m_max_fwhm = { 40.0 }; // max fwhm to consider a star

    std::vector<float> m_first_x_vals = {};
    std::vector<float> m_first_y_vals = {};
    std::vector<Star> m_detectedStars; // vector to store all the stars and properties

    float m_dx{ 0 };
    float m_dy{ 0 };

    double m_plate_scale = .0795336;
    int m_old_num_stars{ 0 };
    int m_num_stars{ 0 };
    float m_seeing{ 0 };
    int m_acquire_star{ -1 }; // Testing for user to select star
    int m_seeing_star { -1 }; // default star to calc seeing
    int m_x_center{};     // 'center' of image or hot spot
    int m_y_center{};

    float m_fps{ 0 };

    void resetAcq(); // class member for resetAcq function

    // Working memory for poke fitting
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
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

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
    int allocate( const dev::shmimT & );

    int processImage( void *curr_src, const dev::shmimT & );

    // shmimMonitor interface for referenc:
    int allocate( const darkShmimT & );

    int processImage( void *curr_src, const darkShmimT & );

  protected:
    std::mutex m_imageMutex;

  protected:
    /** \name INDI
     * @{
     */

    pcf::IndiProperty m_indiP_num_stars;

    pcf::IndiProperty m_indiP_seeing;

    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( psfAcq, m_indiP_fpsSource );

    // For user to select star
    pcf::IndiProperty m_indiP_acquire_star;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_acquire_star );

    // For user to set what star they want to use to calc seeing
    pcf::IndiProperty m_indiP_seeing_star;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_seeing_star );

    // toggling
    pcf::IndiProperty m_indiP_restartAcq;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_restartAcq );

    pcf::IndiProperty m_indiP_recordSeeing;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_recordSeeing );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem( const telem_fgtimings * );

    ///@}
};

inline psfAcq::psfAcq() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    darkShmimMonitorT::m_getExistingFirst = true;
    return;
}

inline psfAcq::~psfAcq() noexcept
{
    for(size_t n = 0; n < m_detectedStars.size(); ++n)
    {
        m_detectedStars[n].deallocate();
    }
}

inline void psfAcq::setupConfig()
{
    shmimMonitorT::setupConfig( config );
    darkShmimMonitorT::setupConfig( config );

    config.add(
        "fitter.fpsSource",
        "",
        "fitter.fpsSource",
        argType::Required,
        "fitter",
        "fpsSource",
        false,
        "string",
        "Device name for getting fps if time-based averaging is used.  This device should have *.fps.current." );
    config.add( "fitter.max_loops",
                "",
                "fitter.max_loops",
                argType::Required,
                "fitter",
                "max_loops",
                false,
                "int",
                "Setting the number of stars to detect in processImage function." );
    config.add( "fitter.zero_area",
                "",
                "fitter.zero_area",
                argType::Required,
                "fitter",
                "zero_area",
                false,
                "int",
                "Setting the pixel area to zero out after detecting stars in processImage function." );
    config.add( "fitter.threshold",
                "",
                "fitter.threshold",
                argType::Required,
                "fitter",
                "threshold",
                false,
                "float",
                "setting how many sigma away from the mean you want to classify a detection." );
    config.add( "fitter.fwhm_threshold",
                "",
                "fitter.fwhm_threshold",
                argType::Required,
                "fitter",
                "fwhm_threshold",
                false,
                "float",
                "minumum fwhm to consider something a star." );
    config.add( "acquisition.x_center",
                "",
                "acquisition.x_center",
                argType::Required,
                "acquisition",
                "x_center",
                false,
                "int",
                "X value for 'center' of image." );
    config.add( "acquisition.y_center",
                "",
                "acquisition.y_center",
                argType::Required,
                "acquisition",
                "y_center",
                false,
                "int",
                "Y value for 'center' of image." );
}

inline int psfAcq::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::loadConfig( _config );
    darkShmimMonitorT::loadConfig( _config );

    _config( m_fpsSource, "fitter.fpsSource" );
    _config( m_max_loops, "fitter.max_loops" ); // Max number of stars to detect in processImage
    _config( m_zero_area, "fitter.zero_area" ); // pixel area to zero out in processImage when a star is detected
    _config( m_threshold, "fitter.threshold" ); // how many sigma away from the mean you want to classify a detection
    _config( m_fwhm_threshold,
             "fitter.fwhm_threshold" ); // how many sigma away from the mean you want to classify a detection

    _config( m_x_center, "acquisition.x_center" );         // star number to acquire
    _config( m_y_center, "acquisition.y_center" );         // star number to acquire

    return 0;
}

inline void psfAcq::loadConfig()
{
    loadConfigImpl( config );
}

inline int psfAcq::appStartup()
{
    if( shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( darkShmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( m_fpsSource != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );
    }

    // creating toggling to restart the acquisition
    createStandardIndiRequestSw( m_indiP_restartAcq, "restart_acq", "Restart Acquisition", "psfAcq");
    registerIndiPropertyNew( m_indiP_restartAcq, INDI_NEWCALLBACK(m_indiP_restartAcq) );

    // INDI prop for seeing
    createROIndiNumber(m_indiP_seeing, "seeing");
    m_indiP_seeing.add(pcf::IndiElement("current"));
    m_indiP_seeing["current"].setValue( m_seeing ); //m_seeing gets assigned the seeing values of the star that is acquired
    registerIndiPropertyReadOnly( m_indiP_seeing );

    // create toggling for recording seeing
    createStandardIndiToggleSw( m_indiP_recordSeeing, "record_seeing", "Record Seeing");
    m_indiP_recordSeeing["toggle"].set(pcf::IndiElement::Off);
    if( registerIndiPropertyNew( m_indiP_recordSeeing, INDI_NEWCALLBACK(m_indiP_recordSeeing)) < 0)
    {
       log<software_error>({__FILE__,__LINE__});
       return -1;
    }

    // INDI prop for user to select star
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_acquire_star, "acquire_star", 0, 20, 1, "%d", "", "" );
    m_indiP_acquire_star["current"].setValue( m_acquire_star );
    m_indiP_acquire_star["target"].setValue( m_acquire_star );

    // INDI prop for user to select seeing star
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_seeing_star, "seeing_star", 0, 20, 1, "%d", "", "" );
    m_indiP_seeing_star["current"].setValue( m_seeing_star );
    m_indiP_seeing_star["target"].setValue( m_seeing_star );

    // number of stars INDI prop
    createROIndiNumber(m_indiP_num_stars, "num_stars");
    m_indiP_num_stars.add(pcf::IndiElement("current"));
    m_indiP_num_stars["current"].setValue( m_num_stars );
    registerIndiPropertyReadOnly( m_indiP_num_stars );

    state( stateCodes::OPERATING );

    return 0;
}

inline int psfAcq::appLogic()
{
    if( shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( darkShmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    shmimMonitorT::updateINDI();
    darkShmimMonitorT::updateINDI();

    updateIfChanged( m_indiP_num_stars, "current", m_num_stars );
    updateIfChanged( m_indiP_seeing, "current", m_seeing );

    for( size_t n = 0; n < m_detectedStars.size() ; ++n )
    {

       updateIfChanged( m_detectedStars[n].prop(), "x", m_detectedStars[n].x );
       updateIfChanged( m_detectedStars[n].prop(), "y", m_detectedStars[n].y );
       updateIfChanged( m_detectedStars[n].prop(), "peak", m_detectedStars[n].max );
       updateIfChanged( m_detectedStars[n].prop(), "fwhm", m_detectedStars[n].fwhm );
    }
    return 0;
}

inline int psfAcq::appShutdown()
{
    shmimMonitorT::appShutdown();
    darkShmimMonitorT::appShutdown();

    return 0;
}

inline int psfAcq::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    m_image.resize( shmimMonitorT::m_width, shmimMonitorT::m_height );
    m_image.setZero();

    m_sm.resize( m_image.rows(), m_image.cols() );

    m_updated = false;
    return 0;
}

// Function to calculate Euclidean distance between two stars
float calculateDistance( float x1, float y1, float x2, float y2 )
{
    return sqrt( ( x2 - x1 ) * ( x2 - x1 ) + ( y2 - y1 ) * ( y2 - y1 ) );
}

inline int psfAcq::processImage( void *curr_src, const dev::shmimT &dummy )
{
    if(mx::sys::get_curr_time() - m_acqQuitTime < m_acqPauseTime ) return 0; // Pausing while telescope moves to star
    static_cast<void>( dummy );

    std::unique_lock<std::mutex> lock( m_imageMutex );

    if( m_dark.rows() == m_image.rows() && m_dark.cols() == m_image.cols() )
    {
        for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
        {
            m_image.data()[nn] = ( (float *)curr_src )[nn] - m_dark.data()[nn];
        }
    }
    else
    {
        for( unsigned nn = 0; nn < shmimMonitorT::m_width * shmimMonitorT::m_height; ++nn )
        {
            m_image.data()[nn] = ( (float *)curr_src )[nn];
        }
    }

    lock.unlock();
    float max;
    float seeing = 0;
    // int max_loops=5;
    int x = 0;
    int y = 0;
    int N_loops = 0;
    // 1. find brightest star
    max = m_image.maxCoeff( &x, &y );

    std::cerr << __LINE__ << " " << max << " " << x << " " << y << "\n";

    // mx::improc::medianSmooth(m_sm, x, y, max, m_image, 3);

    // mx::improc::imageCenterOfLight(m_x, m_y, m_image);
    // std::cerr << __LINE__ << std::endl;

    /*if(fabs(m_x-x) > 2 || fabs(m_y-y) > 2)
    {
        std::cerr << "skip frame\n";
        return 0;
    }*/

    eigenImage<float> llcorn = m_image.block( 0, 0, 32, 32 ); // calc std dev of 32x32 block in lower left corner
    float mean = llcorn.mean();                               // Calculate the mean
    float variance = ( llcorn.array() - mean ).square().sum() / ( llcorn.size() ); // calculate variance
    float stddev = std::sqrt( variance );                                          // Calculate the standard deviation
    float z_score = ( max - mean ) / stddev;                                       // how many std dev away from mean
    float fwhm = m_fwhm_threshold + 1; // getting intial fwhm before entering while loop
    std::size_t numStars = m_detectedStars.size();
    if( numStars == 0 )
    { // This runs when the vector of stars is empty (usually the first time)
        while( ( z_score > m_threshold ) && ( fwhm > m_fwhm_threshold ) && ( N_loops < m_max_loops ) )
        { // m_max_loops, m_fwhm_threshold, and m_threshold are configurable variables
            N_loops = N_loops + 1;
            m_gfit.set_itmax( 1000 );
            // m_zero_area is used to zero out the pixel array once a star is detected but can also be used to set up a
            // sub image around the max pixel
            if( x < m_zero_area )
            {
                x = m_zero_area;
            }
            if( x >= ( m_image.rows() - m_zero_area ) )
            {
                x = m_image.rows() - m_zero_area;
            }
            if( y < m_zero_area )
            {
                y = m_zero_area;
            }
            if( y >= ( m_image.cols() - m_zero_area ) )
            {
                y = m_image.cols() - m_zero_area;
            }
            eigenImage<float> subImage = m_image.block(
                x - m_zero_area,
                y - m_zero_area,
                m_zero_area * 2,
                m_zero_area * 2 ); // set m_image to subImage to speed up gaussian, x,y is position of max pixel
            m_gfit.setArray( subImage.data(), subImage.rows(), subImage.cols() );
            // m_gfit.setArray(m_image.data(), m_image.rows(), m_image.cols());
            m_gfit.setGuess( 0, max, m_zero_area, m_zero_area, mx::math::func::fwhm2sigma( m_fwhmGuess ) );
            m_gfit.fit();
            m_x = ( x - m_zero_area ) + m_gfit.x0();
            m_y = ( y - m_zero_area ) + m_gfit.y0();
            m_first_x_vals.push_back( m_x ); // adding first detected x value to vector
            m_first_y_vals.push_back( m_y ); // adding first detected y value to vector
            fwhm = m_gfit.fwhm() ;
            max = m_gfit.G();
            seeing = fwhm * m_plate_scale;
            int x_value = static_cast<int>(
                m_x ); // convert m_x to an int so we can 0 out a rectangular area around the detected star
            int y_value = static_cast<int>( m_y );
            if (fwhm > m_fwhm_threshold && fwhm < m_max_fwhm ){
                Star newStar;
                newStar.x = m_x; // Adding attributes to the new star
                newStar.y = m_y;
                newStar.max = max;
                newStar.fwhm = fwhm;
                newStar.seeing = seeing;
                std::unique_lock<std::mutex> lock( m_indiMutex );
                newStar.allocate();
                m_detectedStars.push_back( newStar );
                int index = m_detectedStars.size() - 1;
                std::string starPrefix = "star_" + std::to_string( index );
                createROIndiNumber(m_detectedStars.back().prop(), starPrefix);  //, "Star " + std::to_string( m_detectedStars.size() ) + " Properties", "Star Acq" );
                m_detectedStars.back().prop().add( pcf::IndiElement( "x" ) );
                m_detectedStars.back().prop()["x"].set( m_detectedStars.back().x );
                m_detectedStars.back().prop().add( pcf::IndiElement( "y" ) );
                m_detectedStars.back().prop()["y"].set( m_detectedStars.back().y );
                m_detectedStars.back().prop().add( pcf::IndiElement( "peak" ) );
                m_detectedStars.back().prop()["peak"].set( m_detectedStars.back().max );
                m_detectedStars.back().prop().add( pcf::IndiElement( "fwhm" ) );
                m_detectedStars.back().prop()["fwhm"].set( m_detectedStars.back().fwhm );
                registerIndiPropertyReadOnly( m_detectedStars.back().prop() );
            }
            if( x_value < m_zero_area )
            {
                x_value = m_zero_area;
            }
            if( x_value >= ( m_image.rows() - m_zero_area ) )
            {
                x_value = m_image.rows() - m_zero_area;
            }
            if( y_value < m_zero_area )
            {
                y_value = m_zero_area;
            }
            if( y_value >= ( m_image.cols() - m_zero_area ) )
            {
                y_value = m_image.cols() - m_zero_area;
            }
            for( int i = x_value - m_zero_area; i < ( x_value + m_zero_area ); i++ )
            { // zeroing out area around the star centered at m_x and m_y(8x8 pixel area)
                for( int j = y_value - m_zero_area; j < ( y_value + m_zero_area ); j++ )
                {
                    m_image( i, j ) = 0; // m_zero_area is defaulted to 20 to zero out a pixel array around the star
                }
            }
            max = m_image.maxCoeff( &x, &y );
            z_score = ( max - mean ) / stddev;
        }
    }

    else
    {
        // In here is where we track the stars using cross correlation between the first frame and subsequent frames
        while( ( z_score > m_threshold ) && ( fwhm > m_fwhm_threshold ) && ( N_loops < m_max_loops ) )
        {
            N_loops = N_loops + 1;
            m_gfit.set_itmax( 1000 );
            // m_zero_area is used to zero out the pixel array once a star is detected but can also be used to set up a
            // sub image around the max pixel
            if( x < m_zero_area )
            {
                x = m_zero_area;
            }
            if( x >= ( m_image.rows() - m_zero_area ) )
            {
                x = m_image.rows() - m_zero_area;
            }
            if( y < m_zero_area )
            {
                y = m_zero_area;
            }
            if( y >= ( m_image.cols() - m_zero_area ) )
            {
                y = m_image.cols() - m_zero_area;
            }
            eigenImage<float> subImage = m_image.block(
                x - m_zero_area,
                y - m_zero_area,
                m_zero_area * 2,
                m_zero_area * 2 ); // set m_image to subImage to speed up gaussian, x,y is position of max pixel
            // 2. fit it's position
            m_gfit.setArray( subImage.data(), subImage.rows(), subImage.cols() );
            m_gfit.setGuess( 0, max, m_zero_area, m_zero_area, mx::math::func::fwhm2sigma( m_fwhmGuess ) );
            m_gfit.fit();
            fwhm = m_gfit.fwhm() ;
            max = m_gfit.G();
            seeing = fwhm * m_plate_scale;
            m_x = ( x - m_zero_area ) + m_gfit.x0();
            m_y = ( y - m_zero_area ) + m_gfit.y0();
            int x_value = static_cast<int>(
                m_x ); // convert m_x to an int so we can 0 out a rectangular area around the detected star
            int y_value = static_cast<int>( m_y );
            if( x_value < m_zero_area )
            {
                x_value = m_zero_area;
            }
            if( x_value >= ( m_image.rows() - m_zero_area ) )
            {
                x_value = m_image.rows() - m_zero_area;
            }
            if( y_value < m_zero_area )
            {
                y_value = m_zero_area;
            }
            if( y_value >= ( m_image.cols() - m_zero_area ) )
            {
                y_value = m_image.cols() - m_zero_area;
            }
            for( int i = x_value - m_zero_area; i < ( x_value + m_zero_area ); i++ )
            { // zeroing out area around the star centered at m_x and m_y(8x8 pixel area)
                for( int j = y_value - m_zero_area; j < ( y_value + m_zero_area ); j++ )
                {
                    m_image( i, j ) = 0; // m_zero_area is defaulted to 20 to zero out a pixel array around the star
                }
            }
            // 3. search through all known stars to figure out which one it corresponds to.  You can NOT assume it is in the order of the vector.
            // This simple for loop calculate the distance from the detected star to the cloest star already in the list
            // and updates the values
            int threshold_distance = 20; // distance between new stars should be a small positive number so this updates
            int tracker = 0; // tracks if the current star detected updated an already known star
            if (fwhm > m_fwhm_threshold && fwhm < m_max_fwhm ){
                for( Star &star : m_detectedStars )
                {
                    float dist = calculateDistance( star.x, star.y, x_value, y_value );
                    // 4. if it is found, update that star's data in the vector
                    if( dist < threshold_distance )
                    {
                        star.x = m_x;
                        star.y = m_y;
                        star.max = max;
                        star.fwhm = fwhm;
                        star.seeing = seeing;
                        tracker = 1;
                        continue;
                    }
                }
                if (tracker == 0 && m_detectedStars.size() < m_max_loops) { // 5. if it is not found, add a star and corresponding INDI Property using push_back
                    Star newStar;
                    newStar.x = m_x; // Adding attributes to the new star
                    newStar.y = m_y;
                    newStar.max = max;
                    newStar.fwhm = fwhm;
                    newStar.seeing = seeing;
                    std::unique_lock<std::mutex> lock( m_indiMutex );
                    newStar.allocate();
                    m_detectedStars.push_back( newStar );
                    int index = m_detectedStars.size() - 1;
                    std::string starPrefix = "star_" + std::to_string( index );
                    createROIndiNumber(m_detectedStars.back().prop(), starPrefix, "Star " + std::to_string( index ) + " Properties", "Star Acq" );
                    m_detectedStars.back().prop().add( pcf::IndiElement( "x" ) );
                    m_detectedStars.back().prop()["x"].set( m_detectedStars.back().x );
                    m_detectedStars.back().prop().add( pcf::IndiElement( "y" ) );
                    m_detectedStars.back().prop()["y"].set( m_detectedStars.back().y );
                    m_detectedStars.back().prop().add( pcf::IndiElement( "peak" ) );
                    m_detectedStars.back().prop()["peak"].set( m_detectedStars.back().max );
                    m_detectedStars.back().prop().add( pcf::IndiElement( "fwhm" ) );
                    m_detectedStars.back().prop()["fwhm"].set( m_detectedStars.back().fwhm );
                    registerIndiPropertyReadOnly( m_detectedStars.back().prop() );
                }
            }
            max = m_image.maxCoeff( &x, &y );
            z_score = ( max - mean ) / stddev;
        }
    }

    m_num_stars = m_detectedStars.size();
    // If statement that get the delta x and delta y from the 'center' of image
    static int delta_x;
    static int delta_y;
    double plate_scale = .0795336; // plate scale factor: deltatheta/deltapixel, calculated in python, arcsec/pixel
    //deltatheta -> Angular seperation between two stars in arcsec (from published data)
    //deltapixel -> Pixel seperation between same two stars on our detector
    if ( m_acquire_star != -1 &&  (m_acquire_star > m_detectedStars.size() - 1 || m_acquire_star < 0 )){
        std::cout << "Please enter a star number between 0 and " << m_detectedStars.size() - 1 << "." << std::endl;
        m_acquire_star = -1;
    }

    if ( m_seeing_star != -1 &&  (m_seeing_star > m_detectedStars.size() - 1 || m_seeing_star < 0 )){
        std::cout << "Please enter a star number between 0 and " << m_detectedStars.size() - 1 << "." << std::endl;
        m_seeing_star = -1;
    }

    if( m_acquire_star >= 0 && m_acquire_star < m_detectedStars.size())
    {
        m_acqQuitTime = mx::sys::get_curr_time();
        m_temp_acq_star = m_acquire_star;
        delta_x = m_detectedStars[m_acquire_star].x - m_x_center;
        delta_y = m_detectedStars[m_acquire_star].y - m_y_center;
        std::cout << "delta_x = " << delta_x << "    delta_y = " << delta_y << std::endl;

        // negative signs because we want to move scope opposite of how far it is from 'center'
        double x_arcsec = -1*delta_y * plate_scale; //positive x_arcsec moves up, negetive moves down
        double y_arcsec = -1*delta_x * plate_scale; //positive y_arcsec moves right, negetive moves left
        std::cout << "x_arcsec=" << x_arcsec << "  y_arcsec=" << y_arcsec << std::endl;

        // for moving telescope
        pcf::IndiProperty ip( pcf::IndiProperty::Number );

        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        //send telescope x and y offsets in acrsec
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = x_arcsec; //how far to move in y direction in arcsec?
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = y_arcsec; //how far to move in x direction in arcsec?

        sendNewProperty( ip );
        resetAcq();
        m_acquire_star = -1;
    }

    if ( m_current_acq_star >= 0 && m_current_acq_star <= m_num_stars ){
        m_seeing = m_detectedStars[m_current_acq_star].seeing;
    }

    m_updated = true;
    return 0;
}

inline int psfAcq::allocate( const darkShmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    if( darkShmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "dark is not float" } );
    }

    m_dark.resize( darkShmimMonitorT::m_width, darkShmimMonitorT::m_height );
    m_dark.setZero();

    return 0;
}

inline int psfAcq::processImage( void *curr_src, const darkShmimT &dummy )
{
    static_cast<void>( dummy );

    std::unique_lock<std::mutex> lock( m_imageMutex );

    for( unsigned nn = 0; nn < darkShmimMonitorT::m_width * darkShmimMonitorT::m_height; ++nn )
    {
        m_dark.data()[nn] += ( (float *)curr_src )[nn];
    }

    lock.unlock();

    log<text_log>( "dark updated", logPrio::LOG_INFO );

    return 0;
}

//delete m_detectedStars Properties
void psfAcq::resetAcq(){
    for(size_t n=0; n < m_detectedStars.size(); ++n)
    {
        if(m_indiDriver) m_indiDriver->sendDelProperty(m_detectedStars[n].prop());
        if(!m_indiNewCallBacks.erase(m_detectedStars[n].prop().createUniqueKey()))
        {
            log<software_error>({__FILE__, __LINE__, "failed to erase " + m_detectedStars[n].prop().createUniqueKey()});
        }
    }
    std::cout << "size=" << m_detectedStars.size() << std::endl;
    m_detectedStars.clear();
}

//for toggling Restart Acquisition
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_restartAcq )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_restartAcq, ipRecv);
    if(!ipRecv.find("request")) return 0;
    std::unique_lock<std::mutex> lock(m_indiMutex);

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        std::cout << "size=" << m_detectedStars.size() << std::endl;
        resetAcq();
        std::cout << "size=" << m_detectedStars.size() << std::endl;
        return 0;
    }
    else if( ipRecv["request"].getSwitchState() == pcf::IndiElement::Off)
    {
        return 0;
    }

    log<software_error>({__FILE__,__LINE__, "switch state fall through."});
    return -1;
}

//for toggling Recording Seeing
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_recordSeeing )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_recordSeeing, ipRecv);
    if(!ipRecv.find("toggle")) return 0;
    std::unique_lock<std::mutex> lock(m_indiMutex);

    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        m_current_acq_star = m_temp_acq_star;
        updateSwitchIfChanged(m_indiP_recordSeeing, "toggle", pcf::IndiElement::On, INDI_BUSY);
        //m_seeing = m_detectedStars[m_current_acq_star].seeing;
        return 0;
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
    {
        m_current_acq_star = -1;
        updateSwitchIfChanged(m_indiP_recordSeeing, "toggle", pcf::IndiElement::Off, INDI_IDLE);
        return 0;
    }


    log<software_error>({__FILE__,__LINE__, "switch state fall through."});
    return -1;
}

// For user to select acquisition star number
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_acquire_star )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_acquire_star.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_acquire_star, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_acquire_star = target;

    log<text_log>( "set acquire_star = " + std::to_string( m_acquire_star ), logPrio::LOG_NOTICE );
    return 0;
}

// For user to select seeing star number
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_seeing_star )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_seeing_star.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_seeing_star, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    m_seeing_star = target;

    log<text_log>( "set seeing_star = " + std::to_string( m_seeing_star ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_SETCALLBACK_DEFN( psfAcq, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_fpsSource.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid INDI property." } );
        return -1;
    }

    if( ipRecv.find( "current" ) != true ) // this isn't valie
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

    if( fps != m_fps )
    {
        m_fps = fps;
        shmimMonitorT::m_restart = true;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // psfAcq_hpp