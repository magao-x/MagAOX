/** \file gpuMonitor_test.cpp
 * \brief Hardware-free tests of GPU identity, CSV measurements, and sample validity.
 * \ingroup sysMonitor_unit_test
 */

#include "../../../tests/testXWC.hpp"
#include "../../../libMagAOX/libMagAOX.hpp"
#include "../gpuMonitor.hpp"

#include <limits>

namespace libXWCTest
{
namespace sysMonitorTest
{

/// \cond SYSMONITOR_GPU_TEST_HARNESS
namespace
{
const std::string gpuUUIDA = "GPU-01234567-89ab-cdef-0123-456789abcdef"; ///< First fixture device UUID.
const std::string gpuUUIDB = "GPU-11234567-89ab-cdef-0123-456789abcdef"; ///< Second fixture device UUID.

/// Build a query row with one fixture identity and supplied measurement fields.
std::string gpuRow( const std::string &metrics /**< [in] eight comma-separated measurements */ )
{
    return gpuUUIDA + ", 00000000:ab:01.0, " + metrics;
}
} // namespace
/// \endcond

/// Verify UUID normalization is stable and rejects malformed or colliding identities.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "GPU property keys preserve physical UUID identities", "[sysMonitor][gpuMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::gpuKey;
    #endif
    // clang-format on
    using MagAOX::app::sysMonitorDetail::gpuKey;
    REQUIRE( gpuKey( gpuUUIDA ) == "gpu_01234567_89ab_cdef_0123_456789abcdef" );
    REQUIRE( gpuKey( "GPU-01234567-89AB-CDEF-0123-456789ABCDEF" ) == gpuKey( gpuUUIDA ) );
    REQUIRE( gpuKey( gpuUUIDA ) != gpuKey( gpuUUIDB ) );

    for( const auto &invalid : { "",
                                 "GPU-0",
                                 "0",
                                 "GPU-01234567_89ab-cdef-0123-456789abcdef",
                                 "GPU-01234567-89ab-cdef-0123-456789abcdeg",
                                 "MIG-01234567-89ab-cdef-0123-456789abcdef" } )
    {
        INFO( invalid );
        REQUIRE( gpuKey( invalid ).empty() );
    }
}

/// Parse all eight measurement fields, retaining units, timestamps, and stable device order.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "GPU query parser normalizes measurements and device order", "[sysMonitor][gpuMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseGPUs;
    MagAOX::logger::telem_gpu::gpuT::operator==;
    #endif
    // clang-format on
    using MagAOX::app::sysMonitorDetail::parseGPUs;
    using MagAOX::logger::telem_gpu;
    std::vector<telem_gpu::gpuT> gpus;
    const auto                   rowA = gpuRow( "65, 75.5, 12.5, 100.25, 8192, 1500, 1510, 7000" );
    const auto                   rowB = gpuUUIDB + ",0000:02:00.0,0,0,100,0,16384,0,0,0";
    REQUIRE( parseGPUs( gpus, " \t" + rowB + "\r\n\n" + rowA + "\n", 1234567890.25 ) == 0 );
    REQUIRE( gpus.size() == 2 );
    REQUIRE( gpus[0].m_uuid == gpuUUIDA );
    REQUIRE( gpus[1].m_uuid == gpuUUIDB );
    REQUIRE( gpus[0].m_pci == "00000000:ab:01.0" );
    REQUIRE( gpus[0].m_status == "ok" );
    REQUIRE( gpus[0].m_valid == telem_gpu::AllValid );
    REQUIRE( gpus[0].m_temperature == 65 );
    REQUIRE( gpus[0].m_load == Approx( 0.755 ) );
    REQUIRE( gpus[0].m_memoryLoad == Approx( 0.125 ) );
    REQUIRE( gpus[0].m_memoryUsed == 100.25 );
    REQUIRE( gpus[0].m_memoryTotal == 8192 );
    REQUIRE( gpus[0].m_clockGraphics == 1500 );
    REQUIRE( gpus[0].m_clockSM == 1510 );
    REQUIRE( gpus[0].m_clockMemory == 7000 );
    REQUIRE( gpus[0].m_sampleTime == 1234567890.25 );
    REQUIRE( gpus[1].m_load == 0 );
    REQUIRE( gpus[1].m_memoryLoad == 1 );

    const auto first = gpus;
    REQUIRE( parseGPUs( gpus, rowA + "\n" + rowB, 1234567899.25 ) == 0 );
    REQUIRE( gpus == first );
    REQUIRE( gpus[0].m_sampleTime != first[0].m_sampleTime );
    gpus[0].m_valid = 0;
    REQUIRE_FALSE( gpus == first );
}

/// Unsupported metrics are independently invalid without discarding measurements from the same device.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "GPU unsupported fields retain explicit validity", "[sysMonitor][gpuMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseGPUs;
    #endif
    // clang-format on
    using MagAOX::app::sysMonitorDetail::parseGPUs;
    using MagAOX::logger::telem_gpu;
    std::vector<telem_gpu::gpuT> gpus;
    REQUIRE( parseGPUs( gpus, gpuRow( "N/A, 10, [N/A], 512, 4096, Not Supported, [Not Supported], 5000" ), 5 ) == 0 );
    REQUIRE( gpus.size() == 1 );
    REQUIRE( gpus[0].m_status == "partial" );
    REQUIRE( gpus[0].m_valid ==
             ( telem_gpu::Load | telem_gpu::MemoryUsed | telem_gpu::MemoryTotal | telem_gpu::ClockMemory ) );
    REQUIRE( gpus[0].m_temperature == 0 );
    REQUIRE( gpus[0].m_load == Approx( 0.1 ) );
    REQUIRE( gpus[0].m_memoryUsed == 512 );
    REQUIRE( gpus[0].m_clockMemory == 5000 );
    REQUIRE( parseGPUs( gpus, gpuRow( "N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" ), 6 ) == 0 );
    REQUIRE( gpus[0].m_valid == 0 );
    REQUIRE( gpus[0].m_sampleTime == 6 );
    REQUIRE( parseGPUs( gpus, " \t\r\n", 7 ) == 0 );
    REQUIRE( gpus.empty() );
}

/// Reject malformed complete responses atomically, including nonfinite values under production fast-math flags.
/** \ingroup sysMonitor_unit_test */
TEST_CASE( "GPU query parser rejects corrupt samples without replacing prior state", "[sysMonitor][gpuMonitor]" )
{
    // clang-format off
    #ifdef SYSMONITOR_TEST_DOXYGEN_REF
    MagAOX::app::sysMonitorDetail::parseGPUs;
    #endif
    // clang-format on
    using MagAOX::app::sysMonitorDetail::parseGPUs;
    using MagAOX::logger::telem_gpu;
    std::vector<telem_gpu::gpuT> gpus;
    const auto                   validRow = gpuRow( "65,75,12,100,8192,1500,1510,7000" );
    REQUIRE( parseGPUs( gpus, validRow, 10 ) == 0 );
    const auto prior = gpus;

    for( const auto &fields : { "65,75,12,100,8192,1500,1510",
                                "65,75,12,100,8192,1500,1510,7000,1",
                                "65,75,12,100,8192,1500,1510,",
                                "65,101,12,100,8192,1500,1510,7000",
                                "65,75,101,100,8192,1500,1510,7000",
                                "65,75,12,8193,8192,1500,1510,7000",
                                "65,75,12,0,0,1500,1510,7000",
                                "65,75,12,100,8192,-1,1510,7000",
                                "nan,75,12,100,8192,1500,1510,7000",
                                "65,inf,12,100,8192,1500,1510,7000",
                                "65,75,12,1e999,8192,1500,1510,7000",
                                "65C,75,12,100,8192,1500,1510,7000",
                                "65,75,12,100,8192,1500,,7000",
                                "-1,75,12,100,8192,1500,1510,7000" } )
    {
        INFO( fields );
        REQUIRE( parseGPUs( gpus, gpuRow( fields ), 11 ) == -1 );
        REQUIRE( gpus == prior );
        REQUIRE( gpus[0].m_sampleTime == 10 );
    }

    for( const auto &badRow :
         { "bad uuid,0000:01:00.0,65,75,12,100,8192,1500,1510,7000",
           "GPU-01234567-89ab-cdef-0123-456789abcdef,bad pci,65,75,12,100,8192,1500,1510,7000",
           "GPU-01234567-89ab-cdef-0123-456789abcdef,0000:01:00.8,65,75,12,100,8192,1500,1510,7000",
           "GPU-01234567-89ab-cdef-0123-456789abcdef,0000:01:20.0,65,75,12,100,8192,1500,1510,7000" } )
    {
        REQUIRE( parseGPUs( gpus, validRow + "\n" + badRow, 11 ) == -1 );
        REQUIRE( gpus == prior );
    }

    REQUIRE( parseGPUs( gpus, validRow + "\n" + validRow, 11 ) == -1 );
    REQUIRE( parseGPUs( gpus,
                        validRow + "\nGPU-01234567-89AB-CDEF-0123-456789ABCDEF,0000:02:00.0,"
                                   "65,75,12,100,8192,1500,1510,7000",
                        11 ) == -1 );
    REQUIRE( parseGPUs( gpus, validRow, -1 ) == -1 );
    REQUIRE( parseGPUs( gpus, validRow, std::numeric_limits<double>::infinity() ) == -1 );
    REQUIRE( parseGPUs( gpus, validRow, std::numeric_limits<double>::quiet_NaN() ) == -1 );
    REQUIRE( gpus == prior );
}

} // namespace sysMonitorTest
} // namespace libXWCTest
