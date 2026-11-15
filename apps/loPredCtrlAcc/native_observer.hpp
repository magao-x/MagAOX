/** \file native_observer.hpp
 * \brief Existing-only native observation and bounded private recording; no controller or output-stream interface.
 */
#ifndef PCNATIVE_OBSERVER_HPP
#define PCNATIVE_OBSERVER_HPP
#include "native_observation.hpp"
#include <memory>

namespace DDSPC
{
/// Optional observation consumer; permits bounded cached inference but forbids blocking, learning or factorization.
class NativeObservationSink
{
  public:
    /// Destroy the interface without owning any native producer.
    virtual ~NativeObservationSink() = default;

    /// Consume an immutable record; bounded cached inference is allowed, RLS/QR/rank work is not.
    virtual void observe( const NativeObservation &record /**< [in] Original observation. */ ) noexcept = 0;

    /// Offer a complete immutable DM copy; default consumers need metadata only.
    virtual void command( const NativeObservation &record /**< [in] Original request. */,
                          std::span<const float>   values /**< [in] Entire coherent payload. */ ) noexcept;

    /// Offer an actual coherent/idle command poll witness; busy/racing reads cannot qualify request coverage.
    virtual void command_poll( bool         coherent /**< [in] Existing command metadata read was coherent. */,
                               std::int64_t nowNs /**< [in] Actual post-read time. */ ) noexcept;

    /// Offer an actual local request-sampling opportunity; no missed opportunities are synthesized.
    virtual void tick( std::int64_t nowNs /**< [in] Actual observer MONOTONIC time. */ ) noexcept;

    /// Offer an existing coherent version-2 schedule without registering a producer or consuming a semaphore.
    virtual void schedule( const MagAOX::app::synco::Record &record /**< [in] Original scheduler record. */,
                           std::int64_t receivedNs /**< [in] Actual observer receipt. */ ) noexcept;
};

/// Continuous or finite input worker, with optional bounded recording; ADC reader outlives stop()/destruction.
class NativeObserver
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Sole worker, existing mappings, preallocated queue and private files.

  public:
    /// Validate and allocate without opening files, streams, or starting workers.
    explicit NativeObserver(
        const NativeObservationConfig &config /**< [in] Immutable private observation settings. */ );

    /// Stop/join both workers and close only this observer's mappings/files.
    ~NativeObserver();

    /// Start inputs and optional exclusive private recording; never create/update a shared-memory stream.
    void start( AdcReader             &adc /**< [in] Already-started, independently owned read-only ADC reader. */,
                NativeObservationSink *sink = nullptr /**< [in] Optional nonblocking consumer; outlives stop(). */ );

    /// Stop input, drain the bounded recording queue, and join the writer before releasing the ADC reference.
    void stop();

    /// Copy current observation evidence under its short diagnostic lock.
    NativeObservationStatus status() const;

    /// Copy a complete observed command with its metadata; no physical application is asserted.
    bool copy_command( std::size_t        age /**< [in] Zero selects the newest retained command. */,
                       NativeObservation &record /**< [out] Owned command publication metadata. */,
                       std::span<float>   values /**< [out] Caller storage for the entire command. */ ) const;
};
} // namespace DDSPC
#endif
