/** \file cpuList_test.cpp
 * \brief Offline tests for strict Linux CPU-list normalization.
 * \ingroup cpuList_unit_test
 */
#include "../../../tests/testXWC.hpp"
#include "../cpuList.hpp"

namespace libXWCTest
{
namespace sysTest
{

/// Equivalent CPU lists normalize without imposing a machine-word CPU limit.
/** \ingroup cpuList_unit_test */
TEST_CASE( "CPU lists normalize singletons and ranges", "[cpuList]" )
{
    std::vector<unsigned> cpus;
    REQUIRE( MagAOX::sys::parseCpuList( cpus, " 130,64-66, 0, 65, 2 - 3\n" ) );
    REQUIRE( cpus == std::vector<unsigned>{ 0, 2, 3, 64, 65, 66, 130 } );
    REQUIRE( MagAOX::sys::cpuListString( cpus ) == "0,2-3,64-66,130" );
    REQUIRE( MagAOX::sys::parseCpuList( cpus, "1048575" ) );
    REQUIRE( MagAOX::sys::cpuListString( cpus ) == "1048575" );
    REQUIRE( MagAOX::sys::cpuListString( {} ).empty() );
}

/// Invalid input cannot replace a previously verified set or force unbounded range expansion.
/** \ingroup cpuList_unit_test */
TEST_CASE( "CPU lists reject incomplete and oversized inputs", "[cpuList]" )
{
    const std::vector<std::string> invalid = { "",
                                               " ",
                                               "-1",
                                               "+1",
                                               "1,",
                                               ",1",
                                               "1,,2",
                                               "3-1",
                                               "1-",
                                               "1-2-3",
                                               "1 2",
                                               "1abc",
                                               "1.0",
                                               "4294967296",
                                               "1048576",
                                               "0-65536",
                                               "0-65535,0" };

    for( const auto &text : invalid )
    {
        INFO( text );
        std::vector<unsigned> cpus{ 34 };
        REQUIRE_FALSE( MagAOX::sys::parseCpuList( cpus, text ) );
        REQUIRE( cpus == std::vector<unsigned>{ 34 } );
    }

    std::vector<unsigned> cpus;
    REQUIRE( MagAOX::sys::parseCpuList( cpus, "0-65535" ) );
    REQUIRE( cpus.size() == 65536 );
}

} // namespace sysTest
} // namespace libXWCTest
