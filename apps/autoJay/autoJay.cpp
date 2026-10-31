/** \file autoJay.cpp
 * \brief Automated J-test application entrypoint.
 * \ingroup autoJay_files
 */
#include "autoJay.hpp"

/// Run the dual-DM registration sensor.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Command-line arguments. */ )
{
    MagAOX::app::autoJay app;
    return app.main( argc, argv );
}
