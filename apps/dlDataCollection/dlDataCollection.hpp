/** \file dlDataCollection.hpp
 * \brief The MagAO-X generic ImageStreamIO stream integrator
 *
 * \ingroup app_files
 */

#ifndef dlDataCollection_hpp
#define dlDataCollection_hpp

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

using namespace nvinfer1;

bool loadCSV(const std::string& filename, int numValues, float*& dest) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return false;
    }

    std::string line;


    for (int i = 0; i < numValues; ++i) {
        std::getline(file, line);
        dest[i] = std::stof(line);
    }
    file.close();
    return true;
}


class ImageBuffer {
    private:
        size_t num_channels, num_rows, num_cols;     // Dimensions of each image (rows x columns)
        size_t num_images;             // Total number of images the buffer can hold
        float* buffer;                 // Buffer holding the images
        size_t size;                   // Total number of elements in each image (rows * columns)
        size_t head;                   // Index for the next write position
        bool is_full;                  // Flag to check if buffer is full

    public:
        ImageBuffer(size_t _num_channels, size_t _num_rows, size_t _num_cols, size_t _num_images)
            : num_channels(_num_channels), num_rows(_num_rows), num_cols(_num_cols),
              num_images(_num_images),
              size(_num_channels * _num_rows * _num_cols),
              head(0), is_full(false) {

            buffer = new float[size * num_images];
            memset( buffer, 0, sizeof( float) * size * num_images);
        }

        ~ImageBuffer(){
            delete buffer;
        }

        // Adds an image (array of floats) to the buffer (overwrites if full)
        void add(float* image) {
            memcpy(&buffer[head  * size], image, sizeof(float) * size);
            head = head + 1;
        }

        void clear(){
            head = 0;
            is_full = false;
        }

        void save(const std::string& filename) {
            std::ofstream file(filename, std::ios::out | std::ios::binary);
            if (!file) {
                std::cerr << "Error opening file for writing: " << filename << std::endl;
                return;
            }

            file.write(reinterpret_cast<char*>(buffer), sizeof(float) * size * num_images);

            file.close();
        }


};

namespace MagAOX
{
namespace app
{

class dlDataCollection : public MagAOXApp<true>, public dev::shmimMonitor<dlDataCollection>
{
    // Give the test harness access.
    friend class dlDataCollection_test;

    friend class dev::shmimMonitor<dlDataCollection>;

    // The base shmimMonitor type
    typedef dev::shmimMonitor<dlDataCollection> shmimMonitorT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

  public:
    /** \name app::dev Configurations
     *@{
     */

    ///@}

  protected:
    /** \name Configurable Parameters
     *@{
     */
    std::string dataDirs;     // Location where the data (onnx file, engine, WFS reference) is stored
    std::string ampsDir;     // Location where the data (onnx file, engine, WFS reference) is stored

    int Nperset {0}; // Number of images per dataset file
    int Nset {0}; // Number of images per dataset file
    int NumFrameSkip {0}; // Number of frames to skip to ensure latency is not an issue
    int Nmodes {0}; // Number of modes to control

    int inputC {0};
    int inputH {0};
    int inputW {0};

    float imageNorm; // Normalization constant for the image intensities
    float modalNorm; // Normalization constant for the modal coefficients

    int m_pupPix;      // Number of pixels in the pupil used for the Neural Network
    int pup_offset1_x; // Horizontal offset to the first set of pupils
    int pup_offset1_y; // Vertical offset to the first set of pupils
    int pup_offset2_x; // Horizontal offset to the second set of pupils
    int pup_offset2_y; // Horizontal offset to the second set of pupils
    int pixels_per_quadrant;
    unsigned long frame_counter{ 0 };
    int long frame_wait{ 0 };
    int long frame_saved{ 0 };

    int Npup{ 4 };        // Number of pupils
    float *modeval{ nullptr };
    float *pp_image{ nullptr };
    float *randomAmps{ nullptr };
    ImageBuffer* imagebuffer{ nullptr };
	eigenImage<realT> m_shaped_command;	// 50x50

    size_t m_pwfsWidth{ 0 };  ///< The width of the image
    size_t m_pwfsHeight{ 0 }; ///< The height of the image.

    uint8_t m_pwfsDataType{ 0 }; ///< The ImageStreamIO type code.
    size_t m_pwfsTypeSize{ 0 };  ///< The size of the type, in bytes.

    // variables for sending the output to aol_modevals
    std::string m_modevalChannel;
    IMAGE m_modevalStream;
    uint32_t m_modevalWidth {0}; ///< The width of the shmim
    uint32_t m_modevalHeight {0}; ///< The height of the shmim
    uint8_t m_modevalDataType {0}; ///< The ImageStreamIO type code.
    size_t m_modevalTypeSize {0};  ///< The size of the type, in bytes.

    bool m_modevalOpened {false};
    bool m_modevalRestart {false};
    int dataset_i{ 0 };
    int Nact_across{ 50 };
    int zeroPad { 2 };

  public:
    /// Default c'tor.
    dlDataCollection();

    /// D'tor, declared and defined for noexcept.
    ~dlDataCollection() noexcept
    {
    }

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

    /// Implementation of the FSM for dlDataCollection.
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

    // Custom functions
    int send_to_shmim();

    void loadRandomAmps(int dataset_i);

    // void build_engine(){};

  protected:
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    int processImage( void *curr_src,          ///< [in] pointer to start of current frame.
                      const dev::shmimT &dummy ///< [in] tag to differentiate shmimMonitor parents.
    );
};

inline int dlDataCollection::send_to_shmim()
{
	int ki = 0;
	for(uint32_t col_i=0; col_i < Nact_across; ++col_i){
		for(uint32_t row_i=0; row_i < Nact_across; ++row_i){
			m_shaped_command(row_i, col_i) = modeval[ki];
			ki += 1;
		}
	}

    m_modevalStream.md[0].write = 1;
	memcpy(m_modevalStream.array.raw, m_shaped_command.data(),  Nmodes * sizeof(float));
    m_modevalStream.md[0].cnt0++;
    m_modevalStream.md[0].write = 0;

    ImageStreamIO_sempost( &m_modevalStream, -1 );

    return 0;
}

inline dlDataCollection::dlDataCollection() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

inline void dlDataCollection::setupConfig()
{
    std::cout << "setupConfig()" << std::endl;
    shmimMonitorT::setupConfig( config );

    config.add( "parameters.dataDirs",
                "",
                "parameters.dataDirs",
                argType::Required,
                "parameters",
                "dataDirs",
                false,
                "string",
                "The path to the directory with the onnx model." );

    config.add( "parameters.ampsDir",
                "",
                "parameters.ampsDir",
                argType::Required,
                "parameters",
                "ampsDir",
                false,
                "string",
                "The path to the directory with the random amplitudes." );

    config.add( "parameters.Nperset",
                "",
                "parameters.Nperset",
                argType::Required,
                "parameters",
                "Nperset",
                false,
                "int",
                "Number of images per dataset file" );

    config.add( "parameters.Nset",
                "",
                "parameters.Nset",
                argType::Required,
                "parameters",
                "Nset",
                false,
                "int",
                "Number of dataset files to generate" );

    config.add( "parameters.NumFrameSkip",
                "",
                "parameters.NumFrameSkip",
                argType::Required,
                "parameters",
                "NumFrameSkip",
                false,
                "int",
                "Number of frames to skip to ensure latency is not an issue" );

    config.add( "parameters.imageNorm",
                "",
                "parameters.imageNorm",
                argType::Required,
                "parameters",
                "imageNorm",
                false,
                "float",
                "Normalization term for the preprocessed images." );

    config.add( "parameters.modalNorm",
                "",
                "parameters.modalNorm",
                argType::Required,
                "parameters",
                "modalNorm",
                false,
                "float",
                "Normalization term for the modal coefficients." );

    config.add( "parameters.channel",
                "",
                "parameters.channel",
                argType::Required,
                "parameters",
                "channel",
                false,
                "string",
                "The output channel." );

    config.add( "parameters.Nmodes",
                "",
                "parameters.Nmodes",
                argType::Required,
                "parameters",
                "Nmodes",
                false,
                "int",
                "Number of modes to control." );

    config.add( "parameters.m_pupPix",
                "",
                "parameters.m_pupPix",
                argType::Required,
                "parameters",
                "m_pupPix",
                false,
                "int",
                "Number of pixels across a PWFS pupil." );

    config.add( "parameters.pup_offset1_x",
                "",
                "parameters.pup_offset1_x",
                argType::Required,
                "parameters",
                "pup_offset1_x",
                false,
                "int",
                "Horizontal offset to the top left of the closest set op PWFS pupils." );

    config.add( "parameters.pup_offset1_y",
                "",
                "parameters.pup_offset1_y",
                argType::Required,
                "parameters",
                "pup_offset1_y",
                false,
                "int",
                "Vertical offset to the top left of the closest set op PWFS pupils." );

    config.add( "parameters.pup_offset2_x",
                "",
                "parameters.pup_offset2_x",
                argType::Required,
                "parameters",
                "pup_offset2_x",
                false,
                "int",
                "Horizontal offset to the top left of the furthest set op PWFS pupils." );

    config.add( "parameters.pup_offset2_y",
                "",
                "parameters.pup_offset2_y",
                argType::Required,
                "parameters",
                "pup_offset2_y",
                false,
                "int",
                "Vertical offset to the top left of the furthest set op PWFS pupils." );

}

inline int dlDataCollection::loadConfigImpl( mx::app::appConfigurator &_config )
{
    std::cout << "loadConfigImpl()" << std::endl;
    shmimMonitorT::loadConfig( config );

    _config( dataDirs, "parameters.dataDirs" );
    _config( ampsDir, "parameters.ampsDir" );
    _config( Nperset, "parameters.Nperset" );
    _config( Nset, "parameters.Nset" );
    _config( NumFrameSkip, "parameters.NumFrameSkip" );

    _config( imageNorm, "parameters.imageNorm" );
    _config( modalNorm, "parameters.modalNorm" );
    _config( m_modevalChannel, "parameters.channel");

    _config( Nmodes, "parameters.Nmodes" );
    _config( m_pupPix, "parameters.m_pupPix" );
    _config( pup_offset1_x, "parameters.pup_offset1_x" );
    _config( pup_offset1_y, "parameters.pup_offset1_y" );
    _config( pup_offset2_x, "parameters.pup_offset2_x" );
    _config( pup_offset2_y, "parameters.pup_offset2_y" );

    if( true )
    {
        std::cout << "Debug configuration loading: " << std::endl;
        std::cout << "dataDirs: " << dataDirs << std::endl;
        std::cout << "ampsDir: " << ampsDir << std::endl;
        std::cout << "Nperset: " << Nperset << std::endl;
        std::cout << "Nset: " << Nset << std::endl;
        std::cout << "NumFrameSkip: " << NumFrameSkip << std::endl;
        std::cout << "imageNorm: " << imageNorm << std::endl;
        std::cout << "modalNorm: " << modalNorm << std::endl;
        std::cout << "modeval Channel: " << m_modevalChannel << std::endl;

        std::cout << "m_pupPix: " << m_pupPix << std::endl;
        std::cout << "pup_offset1_x: " << pup_offset1_x << std::endl;
        std::cout << "pup_offset1_y: " << pup_offset1_y << std::endl;
        std::cout << "pup_offset2_x: " << pup_offset2_x << std::endl;
        std::cout << "pup_offset2_y: " << pup_offset2_y << std::endl;
    }

    return 0;
}

inline void dlDataCollection::loadConfig()
{
    loadConfigImpl( config );
}

inline int dlDataCollection::appStartup()
{
    if( shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    // state(stateCodes::READY);
    state( stateCodes::OPERATING );
    return 0;
}

inline int dlDataCollection::appLogic()
{
    if( shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    if( shmimMonitorT::updateINDI() < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
    }

    return 0;
}

inline int dlDataCollection::appShutdown()
{
    shmimMonitorT::appShutdown();
    for( int i = 0; i < Nmodes; ++i )
    {
        modeval[i] = 0;
    }
    send_to_shmim();

    if( pp_image )
    {
        delete[] pp_image;
    }
    if( randomAmps )
    {
        delete[] randomAmps;
    }

    if( modeval )
    {
        delete[] modeval;
    }

    return 0;
}

inline int dlDataCollection::allocate( const dev::shmimT &dummy )
{
    std::cout << "allocate()" << std::endl;
    static_cast<void>( dummy ); // be unused

    // Wavefront sensor setup
    m_pwfsWidth = shmimMonitorT::m_width;
    m_pwfsHeight = shmimMonitorT::m_height;
    std::cout << "Width: " << m_pwfsWidth << " Height: " << m_pwfsHeight << std::endl;

    pixels_per_quadrant = m_pupPix * m_pupPix;
    std::cout << "Pixels: " << pixels_per_quadrant << std::endl;
    pp_image = new float[Npup * pixels_per_quadrant];
    modeval = new float[Nmodes];
    randomAmps = new float[Nperset * Nmodes];
    imagebuffer = new ImageBuffer(Npup, m_pupPix, m_pupPix, Nperset);
    memset( pp_image, 0, sizeof( float ) * Npup * pixels_per_quadrant );
    memset( modeval, 0, sizeof( float) * Nmodes);
    memset( randomAmps, 0, sizeof( float) * Nperset * Nmodes);
    loadRandomAmps(0);
	m_shaped_command.resize(Nact_across, Nact_across);
    std::cout << "Close shmims" << std::endl;
    // Allocate the DM shmim interface
    if(m_modevalOpened){
        ImageStreamIO_closeIm(&m_modevalStream);
    }

    std::cout << "Open shmims" << std::endl;
    m_modevalOpened = false;
    m_modevalRestart = false; //Set this up front, since we're about to restart.

    if( ImageStreamIO_openIm(&m_modevalStream, m_modevalChannel.c_str()) == 0){
        if(m_modevalStream.md[0].sem < 10){
            ImageStreamIO_closeIm(&m_modevalStream);
        }else{
            m_modevalOpened = true;
        }
    }

    std::cout << "Done!" << std::endl;
    if(!m_modevalOpened){
        log<text_log>( m_modevalChannel + " not opened.", logPrio::LOG_NOTICE);
        return -1;
    }else{
        m_modevalWidth = m_modevalStream.md->size[0];
        m_modevalHeight = m_modevalStream.md->size[1];

        m_modevalDataType = m_modevalStream.md->datatype;
        m_modevalTypeSize = sizeof(float);

        log<text_log>( "Opened " + m_modevalChannel + " " + std::to_string(m_modevalWidth) + " x " + std::to_string(m_modevalHeight) + " with data type: " + std::to_string(m_modevalDataType), logPrio::LOG_NOTICE);
    }


    return 0;
}

inline void dlDataCollection::loadRandomAmps(int dataset_index){
    std::string filename = ampsDir + "modeval_dataset_" + std::to_string(dataset_index) + ".csv";
    std::cout << "Loading dataset from: " << filename << std::endl;
    loadCSV(filename, Nperset*Nmodes, randomAmps);
}

inline int dlDataCollection::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    // aol_imwfs2 is reference and dark subtracted and is power normalized.
    //Eigen::Map<eigenImage<unsigned short>> pwfsIm(static_cast<unsigned short *>( curr_src ), m_pwfsHeight, m_pwfsWidth );
    Eigen::Map<eigenImage<float>> pwfsIm(reinterpret_cast<float*>(curr_src), m_pwfsHeight, m_pwfsWidth);

    // Split up the four pupils for the Neural Network.
    int ki = 0;

    for( int col_i = -zeroPad; col_i < (m_pupPix - zeroPad); ++col_i )
    {
        for( int row_i = -zeroPad; row_i < (m_pupPix - zeroPad); ++row_i )
        {
            if ((col_i < 0) or (col_i >= (m_pupPix - 2*zeroPad)) or (row_i < 0) or (row_i >= (m_pupPix - 2*zeroPad))){
                pp_image[ki] = 0;
                pp_image[ki + pixels_per_quadrant] = 0;
                pp_image[ki + 2 * pixels_per_quadrant] = 0;
                pp_image[ki + 3 * pixels_per_quadrant] = 0;
            }
            else {
                pp_image[ki] = imageNorm * (realT)pwfsIm(pup_offset1_y + row_i, pup_offset1_x + col_i );
                pp_image[ki + pixels_per_quadrant] = imageNorm * (realT)pwfsIm( pup_offset1_y + row_i, pup_offset2_x + col_i );
                pp_image[ki + 2 * pixels_per_quadrant] = imageNorm * (realT)pwfsIm( pup_offset2_y + row_i, pup_offset1_x + col_i );
                pp_image[ki + 3 * pixels_per_quadrant] = imageNorm * (realT)pwfsIm( pup_offset2_y + row_i, pup_offset2_x + col_i );
            }
            ++ki;
        }
    }

    for( int i = 0; i < Nmodes; ++i )
    {
        modeval[i] = randomAmps[Nmodes * frame_saved + i];
    }

    // Send to DM
    send_to_shmim();


    // Check if we need to add image to buffer
    if (frame_wait==NumFrameSkip){
        imagebuffer->add(pp_image);
        frame_wait=0;
        frame_saved++;
    }

    // Save dataset
    if ((frame_saved%Nperset==0) and (frame_saved!=0)){
        std::cout << "Saving dataset " << dataset_i << std::endl;
        imagebuffer->save(dataDirs + "images_dataset_" + std::to_string(dataset_i) + ".bin");
        imagebuffer->clear();
        dataset_i++;
        if (dataset_i > Nset){
            appShutdown();
         }
        loadRandomAmps(dataset_i);
        frame_saved = 0;
    }
    frame_wait++;
    frame_counter++;



    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // dlDataCollection_hpp
