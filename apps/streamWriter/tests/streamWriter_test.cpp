/** \file streamWriter_test.cpp
 * \brief Catch2 tests for the streamWriter app.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup streamWriter_files
 */

#include "../../../tests/testXWC.hpp"

#include <cstdio>

#include <xrif/xrif.h>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace streamWriterFaultInject
{

/// Fault behavior for wrapped XRIF calls.
enum class xrifFaultMode
{
    none,
    fail,
    passThroughError
};

/// Fault specification for a wrapped XRIF call.
struct xrifFault
{
    xrifFaultMode mode{ xrifFaultMode::none }; ///< Current fault behavior.
    int           triggerCall{ 0 };            ///< 1-based call to fault, or `-1` for every call.
    xrif_error_t  error{ XRIF_ERROR_BADARG };  ///< Error code returned when the fault triggers.
    int           callCount{ 0 };              ///< Number of wrapped calls observed so far.
};

/// Fault specification for wrapped `fwrite()` calls.
struct fwriteFault
{
    bool   enabled{ false }; ///< Whether the wrapper should short-write.
    int    triggerCall{ 0 }; ///< 1-based `fwrite()` call to short-write, or `-1` for every call.
    size_t resultNmemb{ 0 }; ///< Number of items reported as written when the fault triggers.
    int    callCount{ 0 };   ///< Number of wrapped calls observed so far.
};

/// Aggregate fault-injection state for `streamWriter` XRIF and file-write tests.
struct state
{
    xrifFault   configure;         ///< Fault for `xrif_configure()`.
    xrifFault   setSize;           ///< Fault for `xrif_set_size()`.
    xrifFault   allocateRaw;       ///< Fault for `xrif_allocate_raw()`.
    xrifFault   allocateReordered; ///< Fault for `xrif_allocate_reordered()`.
    xrifFault   setLz4;            ///< Fault for `xrif_set_lz4_acceleration()`.
    xrifFault   encode;            ///< Fault for `xrif_encode()`.
    xrifFault   writeHeader;       ///< Fault for `xrif_write_header()`.
    fwriteFault write;             ///< Fault for `fwrite()`.
};

/// Access the current process-local fault-injection state.
state &faults();

/// Restore all wrapped calls to their pass-through behavior.
void reset();

/// Return `true` when the XRIF fault should trigger for this call.
bool shouldFault( xrifFault &fault );

/// Return `true` when the `fwrite()` fault should trigger for this call.
bool shouldFault( fwriteFault &fault );

/// Scope helper that automatically resets fault-injection state on entry and exit.
struct resetScope
{
    /// Reset all fault-injection state on construction.
    resetScope()
    {
        reset();
    }

    /// Reset all fault-injection state on destruction.
    ~resetScope()
    {
        reset();
    }
};

} // namespace streamWriterFaultInject

xrif_error_t
streamWriter_test_xrif_configure( xrif_t handle, int difference_method, int reorder_method, int compress_method );
xrif_error_t streamWriter_test_xrif_set_size(
    xrif_t handle, xrif_dimension_t w, xrif_dimension_t h, xrif_dimension_t d, xrif_dimension_t f, xrif_typecode_t c );
xrif_error_t streamWriter_test_xrif_allocate_raw( xrif_t handle );
xrif_error_t streamWriter_test_xrif_allocate_reordered( xrif_t handle );
xrif_error_t streamWriter_test_xrif_set_lz4_acceleration( xrif_t handle, int32_t lz4_accel );
xrif_error_t streamWriter_test_xrif_encode( xrif_t handle );
xrif_error_t streamWriter_test_xrif_write_header( char *header, xrif_t handle );
size_t       streamWriter_test_fwrite( const void *ptr, size_t size, size_t nmemb, FILE *stream );
/// \endcond

#define xrif_configure streamWriter_test_xrif_configure
#define xrif_set_size streamWriter_test_xrif_set_size
#define xrif_allocate_raw streamWriter_test_xrif_allocate_raw
#define xrif_allocate_reordered streamWriter_test_xrif_allocate_reordered
#define xrif_set_lz4_acceleration streamWriter_test_xrif_set_lz4_acceleration
#define xrif_encode streamWriter_test_xrif_encode
#define xrif_write_header streamWriter_test_xrif_write_header
#define fwrite streamWriter_test_fwrite
#define protected public
#include "../streamWriter.hpp"
#undef protected
#undef fwrite
#undef xrif_write_header
#undef xrif_encode
#undef xrif_set_lz4_acceleration
#undef xrif_allocate_reordered
#undef xrif_allocate_raw
#undef xrif_set_size
#undef xrif_configure

#include "../../../tests/testMacrosINDI.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
namespace streamWriterFaultInject
{

state &faults()
{
    static state s_faults;
    return s_faults;
}

void reset()
{
    faults() = state{};
}

bool shouldFault( xrifFault &fault )
{
    ++fault.callCount;
    return fault.mode != xrifFaultMode::none && ( fault.triggerCall == -1 || fault.callCount == fault.triggerCall );
}

bool shouldFault( fwriteFault &fault )
{
    ++fault.callCount;
    return fault.enabled && ( fault.triggerCall == -1 || fault.callCount == fault.triggerCall );
}

} // namespace streamWriterFaultInject

xrif_error_t
streamWriter_test_xrif_configure( xrif_t handle, int difference_method, int reorder_method, int compress_method )
{
    auto &fault = streamWriterFaultInject::faults().configure;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_configure( handle, difference_method, reorder_method, compress_method );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_configure( handle, difference_method, reorder_method, compress_method );
}

xrif_error_t streamWriter_test_xrif_set_size(
    xrif_t handle, xrif_dimension_t w, xrif_dimension_t h, xrif_dimension_t d, xrif_dimension_t f, xrif_typecode_t c )
{
    auto &fault = streamWriterFaultInject::faults().setSize;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_set_size( handle, w, h, d, f, c );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_set_size( handle, w, h, d, f, c );
}

xrif_error_t streamWriter_test_xrif_allocate_raw( xrif_t handle )
{
    auto &fault = streamWriterFaultInject::faults().allocateRaw;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_allocate_raw( handle );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_allocate_raw( handle );
}

xrif_error_t streamWriter_test_xrif_allocate_reordered( xrif_t handle )
{
    auto &fault = streamWriterFaultInject::faults().allocateReordered;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_allocate_reordered( handle );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_allocate_reordered( handle );
}

xrif_error_t streamWriter_test_xrif_set_lz4_acceleration( xrif_t handle, int32_t lz4_accel )
{
    auto &fault = streamWriterFaultInject::faults().setLz4;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_set_lz4_acceleration( handle, lz4_accel );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_set_lz4_acceleration( handle, lz4_accel );
}

xrif_error_t streamWriter_test_xrif_encode( xrif_t handle )
{
    auto &fault = streamWriterFaultInject::faults().encode;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_encode( handle );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_encode( handle );
}

xrif_error_t streamWriter_test_xrif_write_header( char *header, xrif_t handle )
{
    auto &fault = streamWriterFaultInject::faults().writeHeader;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        if( fault.mode == streamWriterFaultInject::xrifFaultMode::passThroughError )
        {
            xrif_error_t rv = ::xrif_write_header( header, handle );
            if( rv != XRIF_NOERROR )
            {
                return rv;
            }
        }

        return fault.error;
    }

    return ::xrif_write_header( header, handle );
}

size_t streamWriter_test_fwrite( const void *ptr, size_t size, size_t nmemb, FILE *stream )
{
    auto &fault = streamWriterFaultInject::faults().write;

    if( streamWriterFaultInject::shouldFault( fault ) )
    {
        return ::fwrite( ptr, size, std::min( nmemb, fault.resultNmemb ), stream );
    }

    return ::fwrite( ptr, size, nmemb, stream );
}
/// \endcond

using namespace MagAOX::app;

namespace libXWCTest
{

/** \addtogroup streamWriter_unit_test
 * \brief Additional unit tests for the streamWriter application.
 *
 * \ingroup application_unit_test
 */

/// Namespace for `streamWriter` unit tests.
/** \ingroup streamWriter_unit_test
 */
namespace streamWriterTest
{

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
struct streamWriter_test : public streamWriter
{
    streamWriter *m_sw;

    /// Construct a test harness instance with the minimum app identity needed by the callbacks.
    streamWriter_test( const std::string &device )
    {
        m_configName = device;
        m_outName    = device;

        XWCTEST_SETUP_INDI_NEW_PROP( writing );
    }
};
/// \endcond

/// \cond DOXYGEN_SUPPRESS_TEST_HARNESS
struct streamWriter_data_test
{
    streamWriter_test *m_sw;

    /// Bind the helper view to a `streamWriter` test harness.
    streamWriter_data_test( streamWriter_test *sw )
    {
        m_sw = sw;
    }

    /// Return the configured raw image output directory.
    std::string rawimageDir()
    {
        return m_sw->m_rawimageDir;
    }

    /// Build a valid `writing` toggle request property with the requested switch state.
    pcf::IndiProperty writingToggle( pcf::IndiElement::SwitchStateType state )
    {
        pcf::IndiProperty ip;
        m_sw->createStandardIndiToggleSw( ip, "writing" );
        ip["toggle"].setSwitchState( state );

        return ip;
    }

    /// Allocate circular buffers for a synthetic stream with the requested geometry and chunking.
    int setup_circbufs( int width, int height, int dataType, int circBuffLength, int writeChunkLength )
    {
        m_sw->m_typeSize = ImageStreamIO_typesize( dataType );

        m_sw->m_maxCircBuffLength   = circBuffLength;
        m_sw->m_maxCircBuffSize     = ( width * height * m_sw->m_typeSize * circBuffLength + 1 ) / 1048576.0;
        m_sw->m_maxWriteChunkLength = writeChunkLength;

        m_sw->m_width    = width;
        m_sw->m_height   = height;
        m_sw->m_dataType = dataType;

        return m_sw->allocate_circbufs();
    }

    /// Allocate the XRIF encoder state after the circular buffers have been configured.
    int setup_xrif()
    {
        m_sw->initialize_xrif();
        return m_sw->allocate_xrif();
    }

    /// Fill the circular buffers with deterministic image and timing data for round-trip comparisons.
    int fill_circbuf_uint16()
    {
        // fill in image data with increasing 256 bit vals.
        for( size_t pp = 0; pp < m_sw->m_circBuffLength; ++pp )
        {
            uint16_t v = pp;
            for( size_t rr = 0; rr < m_sw->m_width; ++rr )
            {
                for( size_t cc = 0; cc < m_sw->m_height; ++cc )
                {
                    ( (uint16_t *)
                          m_sw->m_rawImageCircBuff )[pp * m_sw->m_width * m_sw->m_height + rr * m_sw->m_height + cc] =
                        v;
                    ++v;
                }
            }

            // fitsFile<uint16_t> ff;
            // ff.write("cb.fits", m_sw->m_rawImageCircBuff);

            // Fill in timing values with unique vals.
            uint64_t *curr_timing = m_sw->m_timingCircBuff + 5 * pp;
            curr_timing[0]        = pp;                                 // image number
            curr_timing[1]        = pp + 1000;                          // atime sec
            curr_timing[2]        = pp + 2000;                          // atime nsec
            curr_timing[3]        = pp + m_sw->m_circBuffLength + 1000; // wtime sec
            curr_timing[4]        = pp + m_sw->m_circBuffLength + 2000; // wtime nsec
        }
        return 0;
    }

    /// Encode the requested frame window into an XRIF output file.
    int write_frames( int start, int stop )
    {
        m_sw->m_rawimageDir = "/tmp";

        m_sw->m_currSaveStart       = start;
        m_sw->m_currSaveStop        = stop;
        m_sw->m_currSaveStopFrameNo = stop;

        m_sw->m_writing = WRITING;
        return m_sw->doEncode();
    }

    /// Decode the saved image and timing archives and compare them against the requested frame window.
    int comp_frames_uint16( size_t start, size_t stop )
    {
        std::cout << "Reading: " << m_sw->m_outFilePath << "\n";

        xrif_t xrif = nullptr;
        if( xrif_new( &xrif ) != XRIF_NOERROR )
        {
            std::cerr << "Error allocating XRIF image decoder.\n";
            return -1;
        }

        xrif_t xrif_timing = nullptr;
        if( xrif_new( &xrif_timing ) != XRIF_NOERROR )
        {
            std::cerr << "Error allocating XRIF timing decoder.\n";
            xrif_delete( xrif );
            return -1;
        }

        char header[XRIF_HEADER_SIZE];

        FILE *fp_xrif = fopen( m_sw->m_outFilePath.c_str(), "rb" );
        if( fp_xrif == nullptr )
        {
            std::cerr << "Error opening " << m_sw->m_outFilePath << " for readback.\n";
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        size_t nr = fread( header, 1, XRIF_HEADER_SIZE, fp_xrif );

        if( nr != XRIF_HEADER_SIZE )
        {
            std::cerr << "Error reading header of " << m_sw->m_outFilePath << "\n";
            fclose( fp_xrif );
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        uint32_t header_size;
        xrif_read_header( xrif, &header_size, header );

        int rv = 0;
        if( xrif_width( xrif ) != m_sw->m_width )
        {
            std::cerr << "width mismatch\n";
            rv = -1;
        }

        if( xrif_height( xrif ) != m_sw->m_height )
        {
            std::cerr << "height mismatch\n";
            rv = -1;
        }

        if( xrif_depth( xrif ) != 1 )
        {
            std::cerr << "depth mismatch\n";
            rv = -1;
        }

        if( xrif_frames( xrif ) != stop - start )
        {
            std::cerr << "frames mismatch\n";
            rv = -1;
        }

        xrif_allocate( xrif );

        nr = fread( xrif->raw_buffer, 1, xrif->compressed_size, fp_xrif );

        if( nr != xrif->compressed_size )
        {
            std::cerr << "error reading compressed image buffer.\n";
            fclose( fp_xrif );
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        if( xrif_decode( xrif ) != XRIF_NOERROR )
        {
            std::cerr << "error decoding compressed image buffer.\n";
            fclose( fp_xrif );
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        size_t badpix = 0;

        for( size_t n = 0; n < m_sw->m_width * m_sw->m_height * m_sw->m_typeSize * ( stop - start ); ++n )
        {
            if( m_sw->m_rawImageCircBuff[start * m_sw->m_width * m_sw->m_height * m_sw->m_typeSize + n] !=
                xrif->raw_buffer[n] )
                ++badpix;
        }

        if( badpix > 0 )
        {
            std::cerr << "Buffers don't match: " << badpix << " bad pixels.\n";
            rv = -1;
        }

        char timing_header[XRIF_HEADER_SIZE];
        nr = fread( timing_header, 1, XRIF_HEADER_SIZE, fp_xrif );
        if( nr != XRIF_HEADER_SIZE )
        {
            std::cerr << "Error reading timing header of " << m_sw->m_outFilePath << "\n";
            fclose( fp_xrif );
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        uint32_t timing_header_size;
        xrif_read_header( xrif_timing, &timing_header_size, timing_header );

        if( xrif_width( xrif_timing ) != 5 )
        {
            std::cerr << "timing width mismatch\n";
            rv = -1;
        }

        if( xrif_height( xrif_timing ) != 1 )
        {
            std::cerr << "timing height mismatch\n";
            rv = -1;
        }

        if( xrif_depth( xrif_timing ) != 1 )
        {
            std::cerr << "timing depth mismatch\n";
            rv = -1;
        }

        if( xrif_frames( xrif_timing ) != stop - start )
        {
            std::cerr << "timing frame count mismatch\n";
            rv = -1;
        }

        xrif_allocate( xrif_timing );

        nr = fread( xrif_timing->raw_buffer, 1, xrif_timing->compressed_size, fp_xrif );
        if( nr != xrif_timing->compressed_size )
        {
            std::cerr << "error reading compressed timing buffer.\n";
            fclose( fp_xrif );
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        if( xrif_decode( xrif_timing ) != XRIF_NOERROR )
        {
            std::cerr << "error decoding compressed timing buffer.\n";
            fclose( fp_xrif );
            xrif_delete( xrif );
            xrif_delete( xrif_timing );
            return -1;
        }

        size_t badtiming = 0;
        for( size_t n = 0; n < 5 * ( stop - start ); ++n )
        {
            if( m_sw->m_timingCircBuff[start * 5 + n] != reinterpret_cast<uint64_t *>( xrif_timing->raw_buffer )[n] )
            {
                ++badtiming;
            }
        }

        if( badtiming > 0 )
        {
            std::cerr << "Timing buffers don't match: " << badtiming << " bad entries.\n";
            rv = -1;
        }

        fclose( fp_xrif );
        xrif_delete( xrif );
        xrif_delete( xrif_timing );

        return rv;
    }
};
/// \endcond

/// Verify the streamWriter INDI callback validator accepts only the expected property.
/**
 * \ingroup streamWriter_unit_test
 */
SCENARIO( "streamWriter INDI Callbacks", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    streamWriter::newCallBack_m_indiP_writing( pcf::IndiProperty() );
    streamWriter::doEncode();
    #endif
    // clang-format on

    XWCTEST_INDI_NEW_CALLBACK( streamWriter, writing );
}

/// Verify the streamWriter writing toggle transitions and stop-write flushes preserve the final queued frame.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter writing toggle transitions and stopped writes", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::newCallBack_m_indiP_writing( pcf::IndiProperty() ) );
    XWCTEST_DOXYGEN_REF( streamWriter::doEncode() );
    #endif
    // clang-format on

    SECTION( "toggle requests only change state for valid start and stop transitions" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        pcf::IndiProperty ipOn  = sw_test.writingToggle( pcf::IndiElement::On );
        pcf::IndiProperty ipOff = sw_test.writingToggle( pcf::IndiElement::Off );

        REQUIRE( sw.m_writing == NOT_WRITING );
        REQUIRE( sw.newCallBack_m_indiP_writing( ipOn ) == 0 );
        REQUIRE( sw.m_writing == START_WRITING );

        REQUIRE( sw.newCallBack_m_indiP_writing( ipOn ) == 0 );
        REQUIRE( sw.m_writing == START_WRITING );

        REQUIRE( sw.newCallBack_m_indiP_writing( ipOff ) == 0 );
        REQUIRE( sw.m_writing == STOP_WRITING );

        REQUIRE( sw.newCallBack_m_indiP_writing( ipOn ) == 0 );
        REQUIRE( sw.m_writing == STOP_WRITING );

        sw.m_writing = NOT_WRITING;

        REQUIRE( sw.newCallBack_m_indiP_writing( ipOff ) == 0 );
        REQUIRE( sw.m_writing == NOT_WRITING );
    }

    SECTION( "stopping without queued frames settles back to NOT_WRITING" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        pcf::IndiProperty ipOn  = sw_test.writingToggle( pcf::IndiElement::On );
        pcf::IndiProperty ipOff = sw_test.writingToggle( pcf::IndiElement::Off );

        REQUIRE( sw_test.setup_circbufs( 16, 16, XRIF_TYPECODE_UINT16, 8, 4 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );

        REQUIRE( sw.newCallBack_m_indiP_writing( ipOn ) == 0 );
        REQUIRE( sw.m_writing == START_WRITING );

        REQUIRE( sw.newCallBack_m_indiP_writing( ipOff ) == 0 );
        REQUIRE( sw.m_writing == STOP_WRITING );

        sw.m_currSaveStart       = 0;
        sw.m_currSaveStop        = 0;
        sw.m_currSaveStopFrameNo = 17;

        REQUIRE( sw.doEncode() == 0 );
        REQUIRE( sw.m_writing == NOT_WRITING );
    }

    SECTION( "stopping with pending frames encodes through the final queued frame" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        pcf::IndiProperty ipOff = sw_test.writingToggle( pcf::IndiElement::Off );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        sw.m_writing = WRITING;
        REQUIRE( sw.newCallBack_m_indiP_writing( ipOff ) == 0 );
        REQUIRE( sw.m_writing == STOP_WRITING );

        sw.m_rawimageDir         = "/tmp";
        sw.m_currSaveStart       = 5;
        sw.m_currSaveStop        = 8;
        sw.m_currSaveStopFrameNo = 8;

        REQUIRE( sw.doEncode() == 0 );
        REQUIRE( sw.m_writing == NOT_WRITING );
        REQUIRE( sw_test.comp_frames_uint16( 5, 8 ) == 0 );
    }
}

/// Verify streamWriter encode/setup helpers cover allocation and write-failure edge cases.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter allocation and encode edge cases", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::allocate_circbufs() );
    XWCTEST_DOXYGEN_REF( streamWriter::allocate_xrif() );
    XWCTEST_DOXYGEN_REF( streamWriter::doEncode() );
    #endif
    // clang-format on

    SECTION( "allocate_circbufs rejects frames that cannot fit in the requested maximum buffer size" )
    {
        streamWriter_test sw( "testdev" );

        sw.m_typeSize            = ImageStreamIO_typesize( XRIF_TYPECODE_UINT16 );
        sw.m_maxCircBuffLength   = 1024;
        sw.m_maxCircBuffSize     = 1024.0;
        sw.m_maxWriteChunkLength = 512;
        sw.m_width               = 16385;
        sw.m_height              = 16385;
        sw.m_dataType            = XRIF_TYPECODE_UINT16;

        REQUIRE( sw.allocate_circbufs() < 0 );
    }

    SECTION( "uncompressed XRIF encoding still preserves the saved frames" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        sw.m_compress = false;

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );
        REQUIRE( sw_test.write_frames( 0, 5 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 0, 5 ) == 0 );
    }

    SECTION( "doEncode returns immediately when not writing" )
    {
        streamWriter_test sw( "testdev" );

        sw.m_writing = NOT_WRITING;

        REQUIRE( sw.doEncode() == 0 );
    }

    SECTION( "doEncode rejects an all-zero timestamp for the first saved frame" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 16, 16, XRIF_TYPECODE_UINT16, 8, 4 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        std::fill_n( sw.m_timingCircBuff, 5 * sw.m_circBuffLength, 0ULL );

        sw.m_rawimageDir         = "/tmp";
        sw.m_currSaveStart       = 0;
        sw.m_currSaveStop        = 1;
        sw.m_currSaveStopFrameNo = 1;
        sw.m_writing             = WRITING;

        REQUIRE( sw.doEncode() < 0 );
    }

    SECTION( "doEncode reports directory creation failures below a non-directory save root" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        const std::filesystem::path blocker = std::filesystem::path( "/tmp" ) / "streamWriter_encode_blocker";

        std::filesystem::remove( blocker );
        {
            std::ofstream blockerOut( blocker.string() );
            blockerOut << "block";
        }

        REQUIRE( sw_test.setup_circbufs( 16, 16, XRIF_TYPECODE_UINT16, 8, 4 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        sw.m_rawimageDir         = blocker.string();
        sw.m_currSaveStart       = 0;
        sw.m_currSaveStop        = 1;
        sw.m_currSaveStopFrameNo = 1;
        sw.m_writing             = WRITING;

        REQUIRE( sw.doEncode() < 0 );

        std::filesystem::remove( blocker );
    }

    SECTION( "doEncode reports file open failures when the output name contains a path separator" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 16, 16, XRIF_TYPECODE_UINT16, 8, 4 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        sw.m_rawimageDir         = "/tmp";
        sw.m_outName             = "bad/name";
        sw.m_currSaveStart       = 0;
        sw.m_currSaveStop        = 1;
        sw.m_currSaveStopFrameNo = 1;
        sw.m_writing             = WRITING;

        REQUIRE( sw.doEncode() < 0 );
    }
}

/// Verify injected XRIF and file-write faults exercise `streamWriter` warning and failure handling.
/**
 * \ingroup streamWriter_unit_test
 */
TEST_CASE( "streamWriter fault injection covers XRIF setup and write warnings", "[streamWriter]" )
{
    // clang-format off
    #ifdef STREAMWRITER_TEST_DOXYGEN_REF
    XWCTEST_DOXYGEN_REF( streamWriter::initialize_xrif() );
    XWCTEST_DOXYGEN_REF( streamWriter::allocate_xrif() );
    XWCTEST_DOXYGEN_REF( streamWriter::doEncode() );
    #endif
    // clang-format on

    SECTION( "allocate_xrif fails when image XRIF configuration faults" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 16, 16, XRIF_TYPECODE_UINT16, 8, 4 ) == 0 );
        REQUIRE( sw.initialize_xrif() == 0 );

        streamWriterFaultInject::reset();
        streamWriterFaultInject::faults().configure.mode        = streamWriterFaultInject::xrifFaultMode::fail;
        streamWriterFaultInject::faults().configure.triggerCall = 1;

        REQUIRE( sw.allocate_xrif() < 0 );
    }

    SECTION( "allocate_xrif fails when timing XRIF reordered allocation faults" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 16, 16, XRIF_TYPECODE_UINT16, 8, 4 ) == 0 );
        REQUIRE( sw.initialize_xrif() == 0 );

        streamWriterFaultInject::reset();
        streamWriterFaultInject::faults().allocateReordered.mode        = streamWriterFaultInject::xrifFaultMode::fail;
        streamWriterFaultInject::faults().allocateReordered.triggerCall = 2;
        streamWriterFaultInject::faults().allocateReordered.error       = XRIF_ERROR_MALLOC;

        REQUIRE( sw.allocate_xrif() < 0 );
    }

    SECTION( "doEncode still recovers frames when image XRIF size reports a warning after setup" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        streamWriterFaultInject::faults().setSize.mode = streamWriterFaultInject::xrifFaultMode::passThroughError;
        streamWriterFaultInject::faults().setSize.triggerCall = 1;
        streamWriterFaultInject::faults().setSize.error       = XRIF_ERROR_INVALID_SIZE;

        REQUIRE( sw_test.write_frames( 0, 5 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 0, 5 ) == 0 );
    }

    SECTION( "doEncode still recovers frames when timing LZ4 setup reports a warning" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        streamWriterFaultInject::faults().setLz4.mode        = streamWriterFaultInject::xrifFaultMode::fail;
        streamWriterFaultInject::faults().setLz4.triggerCall = 2;
        streamWriterFaultInject::faults().setLz4.error       = XRIF_ERROR_BADARG;

        REQUIRE( sw_test.write_frames( 5, 10 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 5, 10 ) == 0 );
    }

    SECTION( "doEncode still recovers frames when timing encode reports an alert after encoding" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        streamWriterFaultInject::faults().encode.mode        = streamWriterFaultInject::xrifFaultMode::passThroughError;
        streamWriterFaultInject::faults().encode.triggerCall = 2;
        streamWriterFaultInject::faults().encode.error       = XRIF_ERROR_NOTIMPL;

        REQUIRE( sw_test.write_frames( 2, 5 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 2, 5 ) == 0 );
    }

    SECTION( "doEncode still recovers frames when timing header generation reports an alert" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        streamWriterFaultInject::faults().writeHeader.mode = streamWriterFaultInject::xrifFaultMode::passThroughError;
        streamWriterFaultInject::faults().writeHeader.triggerCall = 2;
        streamWriterFaultInject::faults().writeHeader.error       = XRIF_ERROR_NULLPTR;

        REQUIRE( sw_test.write_frames( 5, 8 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 5, 8 ) == 0 );
    }

    SECTION( "doEncode returns success but leaves a truncated archive after a short image-data write" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        streamWriterFaultInject::faults().write.enabled     = true;
        streamWriterFaultInject::faults().write.triggerCall = 2;
        streamWriterFaultInject::faults().write.resultNmemb = sw.m_xrif->compressed_size / 2;

        REQUIRE( sw_test.write_frames( 0, 5 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 0, 5 ) < 0 );
    }

    SECTION( "doEncode returns success but leaves a truncated archive after a short timing-data write" )
    {
        streamWriterFaultInject::resetScope faultScope;
        streamWriter_test                   sw( "testdev" );
        streamWriter_data_test              sw_test( &sw );

        REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, 10, 5 ) == 0 );
        REQUIRE( sw_test.setup_xrif() == 0 );
        REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

        streamWriterFaultInject::faults().write.enabled     = true;
        streamWriterFaultInject::faults().write.triggerCall = 4;
        streamWriterFaultInject::faults().write.resultNmemb =
            std::max<size_t>( 1, sw.m_xrif_timing->compressed_size / 2 );

        REQUIRE( sw_test.write_frames( 7, 8 ) == 0 );
        REQUIRE( sw_test.comp_frames_uint16( 7, 8 ) < 0 );
    }
}

/// Verify the streamWriter test harness exposes the expected default configuration state.
/**
 * \ingroup streamWriter_unit_test
 */
SCENARIO( "streamWriter Configuration", "[streamWriter]" )
{
    GIVEN( "A default constructed streamWriter" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        WHEN( "default configurations" )
        {
            REQUIRE( sw_test.rawimageDir() == "" );
        }
    }
}

/// Verify streamWriter encodes raw image buffers into XRIF archives without corrupting frame data.
/**
 * \ingroup streamWriter_unit_test
 */
SCENARIO( "streamWriter encoding data", "[streamWriter]" )
{
    GIVEN( "A default constructed streamWriter and a 120x120 uint16 stream" )
    {
        streamWriter_test      sw( "testdev" );
        streamWriter_data_test sw_test( &sw );

        WHEN( "writing full 1st chunk" )
        {
            int circBuffLength   = 10;
            int writeChunkLength = 5;
            REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, circBuffLength, writeChunkLength ) == 0 );
            REQUIRE( sw_test.setup_xrif() == 0 );

            REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

            REQUIRE( sw_test.write_frames( 0, 5 ) == 0 );

            REQUIRE( sw_test.comp_frames_uint16( 0, 5 ) == 0 );
        }

        WHEN( "writing full 2nd chunk" )
        {
            int circBuffLength   = 10;
            int writeChunkLength = 5;
            REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, circBuffLength, writeChunkLength ) == 0 );
            REQUIRE( sw_test.setup_xrif() == 0 );
            // REQUIRE( sw_test.setup_fname() == 0 );

            REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

            REQUIRE( sw_test.write_frames( 5, 10 ) == 0 );

            REQUIRE( sw_test.comp_frames_uint16( 5, 10 ) == 0 );
        }

        WHEN( "writing partial 1st chunk" )
        {
            int circBuffLength   = 10;
            int writeChunkLength = 5;
            REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, circBuffLength, writeChunkLength ) == 0 );
            REQUIRE( sw_test.setup_xrif() == 0 );
            // REQUIRE( sw_test.setup_fname() == 0 );

            REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

            REQUIRE( sw_test.write_frames( 2, 5 ) == 0 );

            REQUIRE( sw_test.comp_frames_uint16( 2, 5 ) == 0 );
        }

        WHEN( "writing partial 2nd chunk" )
        {
            int circBuffLength   = 10;
            int writeChunkLength = 5;
            REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, circBuffLength, writeChunkLength ) == 0 );
            REQUIRE( sw_test.setup_xrif() == 0 );
            // REQUIRE( sw_test.setup_fname() == 0 );

            REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

            REQUIRE( sw_test.write_frames( 5, 8 ) == 0 );

            REQUIRE( sw_test.comp_frames_uint16( 5, 8 ) == 0 );
        }

        WHEN( "writing a single middle frame" )
        {
            int circBuffLength   = 10;
            int writeChunkLength = 5;
            REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, circBuffLength, writeChunkLength ) == 0 );
            REQUIRE( sw_test.setup_xrif() == 0 );

            REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

            REQUIRE( sw_test.write_frames( 4, 5 ) == 0 );

            REQUIRE( sw_test.comp_frames_uint16( 4, 5 ) == 0 );
        }

        WHEN( "writing a shorter follow-up save after a longer save" )
        {
            int circBuffLength   = 10;
            int writeChunkLength = 5;
            REQUIRE( sw_test.setup_circbufs( 120, 120, XRIF_TYPECODE_UINT16, circBuffLength, writeChunkLength ) == 0 );
            REQUIRE( sw_test.setup_xrif() == 0 );

            REQUIRE( sw_test.fill_circbuf_uint16() == 0 );

            REQUIRE( sw_test.write_frames( 0, 5 ) == 0 );
            REQUIRE( sw_test.comp_frames_uint16( 0, 5 ) == 0 );

            REQUIRE( sw_test.write_frames( 7, 8 ) == 0 );
            REQUIRE( sw_test.comp_frames_uint16( 7, 8 ) == 0 );
        }
    }
}

} // namespace streamWriterTest

} // namespace libXWCTest
