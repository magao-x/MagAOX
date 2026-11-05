/** \file accelFFCtrl.cpp
 * \brief Entry point for the non-actuating accelerometer observer.
 * \ingroup accelFFCtrl_files
 */
#include "accelFFCtrl.hpp"

/// Run the normal MagAO-X lifecycle; no output stream is created or opened.
int main( int argc /**< [in] argument count */, char **argv /**< [in] command-line arguments */ )
{
    MagAOX::app::accelFFCtrl app;
    return app.main( argc, argv );
}
