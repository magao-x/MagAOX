/** \file adc_reader.hpp
 * \brief Read-only native ADC worker; no DDSPC learner, WFS validity, or command publication interface.
 */
#ifndef PCADC_READER_HPP
#define PCADC_READER_HPP

#include <memory>
#include "adc_input.hpp"

namespace DDSPC
{

/// Existing-only ImageStreamIO worker, with all native handles hidden from application/test headers.
class AdcReader
{
    struct Impl;
    std::unique_ptr<Impl> m_impl; ///< Sole worker, mapping, capture state and diagnostic lock.

  public:
    /// Allocate and validate before starting acquisition; opens no stream.
    explicit AdcReader( const AdcConfig &config /**< [in] Startup-only source/calibration. */ );

    /// Stop/join and release only this reader's mapping and semaphore slot.
    ~AdcReader();

    /// Start the read-only worker; never starts an instrument or creates an image.
    void start();

    /// Stop and join outside acquisition, preserving lifetime counters.
    void stop();

    /// Copy ADC-only diagnostics, independently of model locks.
    AdcStatus status() const;

    /// Copy bounded retained samples oldest first for future alignment, outside acquisition.
    std::vector<AdcSample> samples() const;
};

} // namespace DDSPC
#endif // PCADC_READER_HPP
