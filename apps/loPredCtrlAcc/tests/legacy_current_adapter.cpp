/** \file legacy_current_adapter.cpp
 * \brief Execute current accelerometer-disabled Legacy code without touching native output streams.
 */
#include "legacy_replay.hpp"
#include "../loPredCtrlAcc.hpp"

/// \cond LEGACY_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
std::unique_ptr<LegacyModel> current_legacy_model( const LegacySettings &settings )
{
    // Last constructor argument is omitted deliberately: verify Legacy remains the default.
    return std::make_unique<LegacyModelAdapter<DDSPC::PredictiveController>>( settings, 0, 0 );
}

std::unique_ptr<LegacyApp> current_legacy_app()
{
    return std::make_unique<LegacyAppAdapter<MagAOX::app::loPredCtrlAcc, true>>();
}
} //namespace libXWCTest::loPredCtrlAccTest
/// \endcond
