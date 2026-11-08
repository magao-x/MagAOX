/** \file po4aoRouteIO.hpp
 * \brief Installed HOWFS utility and read-only FPS/process-info boundary.
 */
#ifndef po4aoRouteIO_hpp
#define po4aoRouteIO_hpp
#include "po4aoRouting.hpp"
#include "../nnReconstructor/cacaoWriter.hpp"
#include <atomic>

namespace MagAOX::app
{
/// Non-real-time adapter for the inspected tweeter-vispyr utilities.
class Po4aoRouteIO
{
    std::string m_beforeCalibration; ///< Completion marker captured before applying default.

    cacaoWriter m_writer; ///< Read-only verification of the calibration-driven WFS writer restart.

    std::string m_beforeWriter; ///< Pre-command PID/start-tick identity; metadata changes do not count as restart.

  public:
    std::atomic<bool> m_cancelled{ false }; ///< Shutdown cancellation, checked while waiting for children/readbacks.

    /// Bind installed paths, or private paths in the explicitly fake test build.
    Po4aoRouteIO();

    /// Read the route/calibration and require an operator-open loop before any command.
    Po4aoRouteSnapshot capture();

    /// With NN reserved, capture writer identity, invoke the switch utility and verify the symlink.
    void switchInput( const std::string          &target /**< [in] One of the two supported inputs. */,
                      Po4aoRouteBackend::Deadline deadline /**< [in] Operation deadline. */ );

    /// Apply default and await both its marker and a newly running WFS writer before releasing NN.
    void applyDefault( Po4aoRouteBackend::Deadline deadline /**< [in] Operation deadline. */ );

    /// Verify stable process identity and advancing consumption of the requested inode.
    void verifyConsumption( const std::string          &target /**< [in] Requested input stream. */,
                            Po4aoRouteBackend::Deadline deadline /**< [in] Operation deadline. */ );

    /// Read the actual current input; throws when it is absent or unrecognized.
    std::string route() const;
};
} // namespace MagAOX::app
#endif
