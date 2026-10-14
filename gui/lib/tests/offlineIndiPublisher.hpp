/** \file offlineIndiPublisher.hpp
 * \brief Socket-free transport and deterministic clock for shared GUI publisher tests.
 */
#ifndef offlineIndiPublisher_hpp
#define offlineIndiPublisher_hpp

#include <QCoreApplication>
#include <chrono>
#include "../../../INDI/libcommon/IndiClient.hpp"

/// \cond TEST_HARNESS
namespace pcf
{
/// Socket-free replacement for the transport base; publisher logic remains real.
class OfflineIndiClient
{
  public:
    std::vector<IndiProperty> m_queries; ///< Captured discovery requests.

    std::vector<IndiProperty> m_commands; ///< Captured command attempts.

    /// Accept connection metadata without constructing any socket.
    OfflineIndiClient( const std::string &name,     /**< [in] Ignored client name. */
                       const std::string &version,  /**< [in] Ignored client version. */
                       const std::string &protocol, /**< [in] Ignored protocol version. */
                       const std::string &host,     /**< [in] Ignored hostname; no socket is created. */
                       int                port /**< [in] Ignored port. */ );

    /// Allow the publisher's receive-loop update override.
    virtual void update();

    /// Capture a targeted discovery request.
    void sendGetProperties( const IndiProperty &property /**< [in] Query to record. */ );

    /// Capture any command, which discovery must never generate.
    void sendNewProperty( const IndiProperty &property /**< [in] Command to record. */ );

    /// Provide the unused receive-loop entry point without transport.
    void processIndiRequests( bool threaded /**< [in] Unused threading flag. */ );
};

inline OfflineIndiClient::OfflineIndiClient(
    const std::string &, const std::string &, const std::string &, const std::string &, int )
{
}
inline void OfflineIndiClient::update()
{
}
inline void OfflineIndiClient::processIndiRequests( bool )
{
}
inline void OfflineIndiClient::sendGetProperties( const IndiProperty &property )
{
    m_queries.push_back( property );
}
inline void OfflineIndiClient::sendNewProperty( const IndiProperty &property )
{
    m_commands.push_back( property );
}
} // namespace pcf
#define IndiClient OfflineIndiClient
#include "../multiIndiPublisher.hpp"
#undef IndiClient

namespace libXWCTest
{
namespace guiTest
{
using namespace std::chrono;

/// Run the real publisher with a deterministic monotonic clock and fake transport.
class OfflinePublisher : public multiIndiPublisher
{
  public:
    steady_clock::time_point m_now{}; ///< Controlled monotonic time.

    /// Construct the offline transport.
    OfflinePublisher();

    /// Detach live subscribers before publisher destruction.
    ~OfflinePublisher() noexcept;

    /// Advance time and run one production receive-loop update.
    void tick( seconds delta /**< [in] Time to advance. */ );

    /// Expose record count for final-listener and teardown assertions.
    size_t records() const;

  protected:
    /// Supply deterministic time to production scheduling.
    steady_clock::time_point discoveryNow() const override;
};
inline OfflinePublisher::OfflinePublisher() : multiIndiPublisher( "offline", "unused", 0 )
{
}
inline OfflinePublisher::~OfflinePublisher() noexcept
{
    detachAllSubscribers();
}
inline void OfflinePublisher::tick( seconds delta )
{
    m_now += delta;
    update();
}
inline size_t OfflinePublisher::records() const
{
    return m_discovery.size();
}
inline steady_clock::time_point OfflinePublisher::discoveryNow() const
{
    return m_now;
}

} // namespace guiTest
} // namespace libXWCTest
/// \endcond
#endif
