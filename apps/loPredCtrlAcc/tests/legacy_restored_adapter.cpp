/** \file legacy_restored_adapter.cpp
 * \brief Compile the unchanged working-tree loPredCtrl with separate symbols and stream-free bases.
 */
#include "legacy_replay.hpp"
#define DDSPC DDSPCRestored
#define loPredCtrl loPredCtrlRestored
#include "restored/apps/loPredCtrl/loPredCtrl.hpp"
#undef loPredCtrl
#undef DDSPC

/// \cond LEGACY_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
std::unique_ptr<LegacyModel> restored_model( const LegacySettings &settings )
{
    return std::make_unique<LegacyModelAdapter<DDSPCRestored::PredictiveController>>( settings );
}

std::unique_ptr<LegacyApp> restored_app()
{
    return std::make_unique<LegacyAppAdapter<MagAOX::app::loPredCtrlRestored, false>>();
}
} //namespace libXWCTest::loPredCtrlAccTest
/// \endcond
