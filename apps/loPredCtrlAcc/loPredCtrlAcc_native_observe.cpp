/** \file loPredCtrlAcc_native_observe.cpp
 * \brief Private native observation executable; no alternate output or control implementation.
 */
#include "loPredCtrlAcc_native_observe.hpp"

/// Run the guarded private observer using the standard MagAO-X application command line.
int main( int    argc /**< [in] Argument count. */,
          char **argv /**< [in] Private configuration and identity arguments. */ )
{
    MagAOX::app::loPredCtrlAcc_native_observe app;
    return app.main( argc, argv );
}
