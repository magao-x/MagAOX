/** \file pupilXCFit.cpp
 * \brief Entry point for the pyramid pupil cross-correlation fitter.
 * \ingroup pupilXCFit_files
 */

#include "pupilXCFit.hpp"

/// Run the fitter with its application configuration.
int main( int argc /**< [in] argument count */, char **argv /**< [in] argument vector */ )
{
    MagAOX::app::pupilXCFit app;

    return app.main( argc, argv );
}
