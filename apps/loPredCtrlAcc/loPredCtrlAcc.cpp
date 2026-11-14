/** \file loPredCtrlAcc.cpp
 * \brief Standard installed MagAO-X application entrypoint, with native inputs and physical output inhibited.
 */
#include "loPredCtrlAcc_native.hpp"

/// Run the standard framework lifecycle using the invocation/configuration name as the INDI identity.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Standard MagAO-X arguments. */ )
{
    MagAOX::app::loPredCtrlAcc_native app;
    return app.main( argc, argv );
}
