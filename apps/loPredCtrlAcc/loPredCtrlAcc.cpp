/** \file loPredCtrlAcc.cpp
 * \brief Operational DDSPC application entrypoint with existing syncopation auxiliary output.
 */
#if defined( DDSPC_OPERATIONAL_BUILD ) && !defined( DDSPC_SHADOW_ONLY )
    #include "loPredCtrlAcc_operational.hpp"
using Controller = MagAOX::app::loPredCtrlAcc_operational;
#else
    #include "loPredCtrlAcc_native.hpp"
using Controller = MagAOX::app::loPredCtrlAcc_native;
#endif

/// Run the standard framework lifecycle using the invocation/configuration name as the INDI identity.
int main( int argc /**< [in] Argument count. */, char **argv /**< [in] Standard MagAO-X arguments. */ )
{
    Controller app;
    return app.main( argc, argv );
}
