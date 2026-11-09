/** \file po4aoRouting.hpp
 * \brief Whole-app routing contracts, sequencing and isolated installed-utility adapter.
 */
#ifndef po4aoRouting_hpp
#define po4aoRouting_hpp

#include "../nnReconstructor/cacaoWriter.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace MagAOX::app
{
/// Routing identity and ownership captured once for an accepted request.
struct Po4aoRouteSnapshot
{
    std::string route, calibration, writer; ///< Verified pre-request input, calibration and writer.
};

/// Instrument boundary; each operation must finish or throw by the supplied deadline.
/** Callbacks run only in the non-real-time control worker. A command return alone
 * must never satisfy verifyConsumption. The snapshot survives failure for recovery.
 */
struct Po4aoRouteBackend
{
    using Deadline = std::chrono::steady_clock::time_point; ///< Shared operation deadline.
    std::function<Po4aoRouteSnapshot( Deadline )> capture;  ///< Read current routing and ownership before changes.
    std::function<void( const Po4aoRouteSnapshot &, Deadline )> reserveWriter; ///< Inhibit conflicting NN operations.
    std::function<void( Deadline )> freshOutput; ///< Attach and establish fresh PO4AO output before routing to it.
    std::function<void( const std::string &, Deadline )> switchInput; ///< Run the verified input-switch utility.
    std::function<void( Deadline )> applyDefault; ///< Apply default calibration and verify its completion.
    std::function<void( const Po4aoRouteSnapshot &, Deadline )>
        restoreWriter; ///< Restore the reserved writer ownership.
    std::function<void( const std::string &, Deadline )>
                                    verifyConsumption; ///< Observe fresh frames on the requested inode.
    std::function<void( Deadline )> park;              ///< Close admission and wait for actual monitor detachment.
};

/// Execute one serialized request; failure never calls park or attempts a blind rollback.
inline void
routePo4ao( bool                start /**< [in] Explicit whole-app target. */,
            Po4aoRouteBackend  &backend /**< [in] Verified command and readback boundary. */,
            Po4aoRouteSnapshot &snapshot /**< [out] Retained recovery snapshot. */,
            const std::function<void( const std::string & )> &phase /**< [in] Publish operation progress. */,
            Po4aoRouteBackend::Deadline                       deadline /**< [in] End-to-end completion deadline. */ )
{
    phase( "snapshot" );
    snapshot = backend.capture( deadline );
    if( start )
    {
        phase( "fresh_output" );
        backend.freshOutput( deadline );
    }
    phase( "reserving_writer" );
    backend.reserveWriter( snapshot, deadline );
    const std::string target = start ? "po4ao_output" : "aol1_modevalWFS";
    phase( "switching_input" );
    backend.switchInput( target, deadline );
    phase( "applying_default" );
    backend.applyDefault( deadline );
    phase( "restoring_writer" );
    backend.restoreWriter( snapshot, deadline );
    phase( "verifying_consumption" );
    backend.verifyConsumption( target, deadline );
    if( !start )
    {
        phase( "detaching" );
        backend.park( deadline );
    }
    phase( start ? "running" : "stopped" );
}
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
