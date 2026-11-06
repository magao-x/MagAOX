/** \file legacy_original_adapter.cpp
 * \brief Compile the pinned pre-accelerometer source with separate symbols and stream-free bases.
 */
#include "legacy_replay.hpp"
#define DDSPC DDSPCOriginal
#define loPredCtrl loPredCtrlOriginal
#include "original/apps/loPredCtrl/loPredCtrl.hpp"
#undef loPredCtrl
#undef DDSPC

/// \cond LEGACY_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
std::unique_ptr<LegacyModel> original_model( const LegacySettings &settings )
{
    return std::make_unique<LegacyModelAdapter<DDSPCOriginal::PredictiveController>>( settings );
}

std::unique_ptr<LegacyApp> original_app()
{
    return std::make_unique<LegacyAppAdapter<MagAOX::app::loPredCtrlOriginal, false>>();
}
} //namespace libXWCTest::loPredCtrlAccTest
/// \endcond
