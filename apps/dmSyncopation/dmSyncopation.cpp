/** \file dmSyncopation.cpp
 * \brief Entry point for synchronized auxiliary DM commands.
 */
#include "dmSyncopation.hpp"

/// Run the configured syncopation instance.
int main( int argc /**< [in] argument count */, char **argv /**< [in] command-line arguments */ )
{
    MagAOX::app::dmSyncopation app;
    return app.main( argc, argv );
}
