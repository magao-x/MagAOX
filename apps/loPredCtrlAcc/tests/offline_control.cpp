/** \file offline_control.cpp
 * \brief Test vocabulary adapter; all aggregation, qualification and bounds use the shared components.
 */
#include "offline_control.hpp"
namespace libXWCTest::loPredCtrlAccTest
{
const char *offline_mode( OfflineMode mode )
{
    return DDSPC::policy_mode( mode );
}
} // namespace libXWCTest::loPredCtrlAccTest
