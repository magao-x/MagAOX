/** \file accelSim.cpp
 * \brief Entry point for the continuous accelerometer simulator.
 * \ingroup accelSim_files
 */
#include "accelSim.hpp"

/// Run the normal MagAO-X application lifecycle.
int main( int argc /**< [in] argument count */, char **argv /**< [in] command-line arguments */ )
{
    MagAOX::app::accelSim app;
    return app.main( argc, argv );
}
