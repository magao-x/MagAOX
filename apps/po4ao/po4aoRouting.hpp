/** \file po4aoRouting.hpp
 * \brief Whole-app route sequencing, independent of the optional trainer.
 */
#ifndef po4aoRouting_hpp
#define po4aoRouting_hpp

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
} // namespace MagAOX::app
#endif
