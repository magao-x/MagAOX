/** \file cacaoWriter_test.hpp
 * \brief Deterministic FPS transport boundary for production NN handover tests.
 */
#ifndef nnReconstructor_cacaoWriter_test_hpp
#define nnReconstructor_cacaoWriter_test_hpp
#include "../cacaoWriter.hpp"
#include <stdexcept>
#include <vector>

/// \cond NNRECONSTRUCTOR_TEST_SUPPORT
namespace nnTest
{
inline bool              cacaoRunning = false, cacaoKnown = true, cacaoStopCompletes = true, cacaoStartCompletes = true,
                         cacaoDelivery = true;
inline std::vector<bool> cacaoCommands;
} // namespace nnTest
namespace MagAOX::app
{
inline cacaoWriter::cacaoWriter( const std::string &shmDirectory, const std::string &procDirectory )
    : m_shmDirectory( shmDirectory ), m_procDirectory( procDirectory )
{
}
inline bool cacaoWriter::running()
{
    if( !nnTest::cacaoKnown )
        throw std::runtime_error( "FPS evidence unavailable" );
    return nnTest::cacaoRunning;
}
inline void cacaoWriter::command( bool enabled )
{
    nnTest::cacaoCommands.push_back( enabled );
    if( !nnTest::cacaoDelivery )
        throw std::runtime_error( "FIFO unavailable" );
    if( enabled ? nnTest::cacaoStartCompletes : nnTest::cacaoStopCompletes )
        nnTest::cacaoRunning = enabled;
}
} // namespace MagAOX::app
/// \endcond
#endif
