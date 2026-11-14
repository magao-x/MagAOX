/** \file native_observer.hpp
 * \brief Existing-only native observation and bounded private recording; no controller or output-stream interface.
 */
#ifndef PCNATIVE_OBSERVER_HPP
#define PCNATIVE_OBSERVER_HPP
#include "native_observation.hpp"
#include <memory>

namespace DDSPC
{
/// Optional observation consumer; called only by the observer worker and forbidden from blocking or doing model work.
class NativeObservationSink
{
  public:
    /// Destroy the interface without owning any native producer.
    virtual ~NativeObservationSink() = default;

    /// Copy/enqueue an immutable record or count loss; no acquisition-thread numerical work is permitted.
    virtual void observe( const NativeObservation &record /**< [in] Original observation. */ ) noexcept = 0;
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
