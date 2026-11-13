/** \file adc_transport_fixture.hpp
 * \brief Process-private fake ImageStreamIO transport for actual ADC worker/lifecycle tests.
 */
#ifndef PCADC_TRANSPORT_FIXTURE_HPP
#define PCADC_TRANSPORT_FIXTURE_HPP
#include <ImageStreamIO/ImageStreamIO.h>
#include <memory>
#include <string>

/// \cond ADC_TRANSPORT_FIXTURE
namespace libXWCTest::loPredCtrlAccTest
{

/// Own immutable fake publication storage and a private backing-identity file; never a native shmim.
class AdcTransportFixture
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Sole private fixture; reader must stop before fixture destruction.

  public:
    /// Register a static finite raw pair with recent actual clock metadata.
    explicit AdcTransportFixture(
        const std::string &name /**< [in] Process-private input name. */,
        std::uint32_t      depth = 1 /**< [in] Temporal depth, allocated before the worker starts. */ );

    /// Register immutable caller-owned observation pixels/metadata; caller keeps them alive until readers stop.
    AdcTransportFixture( const std::string &name /**< [in] Private fixture name. */,
                         const IMAGE       &image /**< [in] Immutable borrowed source, never a native image. */ );

    /// Unregister and remove only the fixture's private ordinary file.
    ~AdcTransportFixture();

    /// Simulate unlink without modifying memory observed by the running reader.
    void disconnect();

    /// Recreate the private backing file; payload stays immutable for race-free threaded tests.
    void reconnect();

    /// Atomically replace the private identity file while keeping borrowed payload storage unchanged.
    void replace();

    /// Inspect bookkeeping only after the reader stops.
    pid_t reader() const;
};

} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif // PCADC_TRANSPORT_FIXTURE_HPP
