/** \file native_observer.hpp
 * \brief Existing-only native observation and bounded private recording; no controller or output-stream interface.
 */
#ifndef PCNATIVE_OBSERVER_HPP
#define PCNATIVE_OBSERVER_HPP
#include "native_observation.hpp"
#include <memory>

namespace DDSPC
{
/// Finite observation worker and independent local writer; borrowed ADC reader must outlive stop()/destruction.
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

    /// Exclusively create private files and start input observation; never create/update a shared-memory stream.
    void start( AdcReader &adc /**< [in] Already-started, independently owned read-only ADC reader. */ );

    /// Stop input, drain the bounded recording queue, and join the writer before releasing the ADC reference.
    void stop();

    /// Copy current observation evidence under its short diagnostic lock.
    NativeObservationStatus status() const;
};
} // namespace DDSPC
#endif
