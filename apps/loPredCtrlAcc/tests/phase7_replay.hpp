/** \file phase7_replay.hpp
 * \brief Reuse the pure replay interface for an independently compiled committed Phase 7 manager.
 */
#ifndef loPredCtrlAcc_phase7_replay_hpp
#define loPredCtrlAcc_phase7_replay_hpp
#include "phase6_replay.hpp"
namespace libXWCTest::loPredCtrlAccTest
{
/// Construct the actual committed Phase 7 pair behind the existing stream-free oracle interface.
std::unique_ptr<Phase6Replay> phase7_replay( int   modes /**< [in] Controlled modes. */,
                                             bool  qrd /**< [in] QR learner. */,
                                             float gamma        = 1.f /**< [in] Forgetting factor. */,
                                             int   history      = 5 /**< [in] WFS history. */,
                                             int   future       = 3 /**< [in] Horizon. */,
                                             int   accelHistory = 20 /**< [in] Sensor history. */ );
} //namespace libXWCTest::loPredCtrlAccTest
#endif
