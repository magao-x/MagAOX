/** \file loopTimingMonitor.cpp
 * \brief Entrypoint for the passive loop timing monitor.
 */
#include "loopTimingMonitor.hpp"

/// Run the configured timing monitor.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Argument values. */ )
{
    MagAOX::app::loopTimingMonitor app;
    return app.main( argc, argv );
}
