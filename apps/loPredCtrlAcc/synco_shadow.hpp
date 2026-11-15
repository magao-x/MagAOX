/** \file synco_shadow.hpp
 * \brief Precomputed shadow payloads for the existing syncopation producer Request interface; no stream writes.
 */
#ifndef PC_SYNCO_SHADOW_HPP
#define PC_SYNCO_SHADOW_HPP
#include "shadow_controller.hpp"
#include "../../libMagAOX/app/dmSyncopationTiming.hpp"
#include <span>
#include <vector>
#include <atomic>
#include <memory>
#include <mutex>

namespace DDSPC
{
struct OnlineShadowStatus;
/// Immutable private payloads prepared by the numerical worker, never an auxiliary shared image.
class SyncoShadowPlans
{
    std::vector<float> m_pixels;        ///< Chronological frame/phase/pixel storage, allocated before preparation.
    std::size_t        m_elements{ 0 }; ///< Complete DM image extent.
    unsigned           m_frames{ 0 };   ///< Prepared WFS command frames, each with four distinct slots.
    std::int64_t       m_epoch{ 0 }, m_generation{ 0 }, m_grid{ -1 },
        m_ready{ -1 };     ///< Exact scheduler and preparation epoch.
    bool m_valid{ false }; ///< False on missing inputs, changed schedule or nonfinite plans.

  public:
    /// Allocate fixed payload storage outside the producer callback.
    SyncoShadowPlans( std::size_t elements /**< [in] Full auxiliary image pixels. */,
                      unsigned    frames /**< [in] Future WFS command frames, at most eight. */ );

    /// Render all newest-first native plan rows into independent chronological slot images before polling.
    bool prepare( const Matrix                     &plan /**< [in] Native newest-frame/newest-slot-first modal plan. */,
                  const Matrix                     &shapes /**< [in] Pixels by modes calibrated auxiliary shapes. */,
                  const MagAOX::app::synco::Record &schedule /**< [in] Original version-2 schedule. */,
                  std::int64_t gridNs /**< [in] Independently associated first command-frame target. */,
                  std::int64_t readyNs /**< [in] Actual completion time; not a forecast deadline. */ );

    /// Copy an already prepared exact target or reject; no Eigen, allocation, wait, semaphore or publication.
    bool command( const MagAOX::app::synco::Request &request /**< [in] Exact request returned by the base poll. */,
                  const MagAOX::app::synco::Record  &schedule /**< [in] Current coherent schedule for revalidation. */,
                  std::int64_t                       nowNs /**< [in] Actual same-host callback time. */,
                  std::span<float>                   pixels /**< [out] Caller-owned complete image. */ ) const;

    /// Inhibit every old payload after input loss, mode changes or scheduler reset.
    void invalidate();
};

/// Private prepared-payload cache shared by numerical preparation and the observation-only callback.
class SyncoShadowProducer
{
    std::array<std::unique_ptr<SyncoShadowPlans>, 2> m_ready, m_previous, m_work; ///< Preserve pending preceding slots.
    std::vector<float>    m_copy;  ///< Sole callback's bounded private image, never an operational mapping.
    std::mutex            m_mutex; ///< Held only for pointer exchange or one prepared-image copy, never model work.
    std::atomic<unsigned> m_models{ 0 }, m_requested{ 0 }; ///< Readiness mask and diagnostic selection only.
    std::atomic<unsigned> m_planMask{ 0 }; ///< Successful preparations since the last corresponding input fault.
    std::array<std::atomic<std::int64_t>, 2> m_lossNs{
        -1, -1 }; ///< Latest acquisition loss, preventing old queued origins rearming.
    std::array<std::int64_t, 2> m_originNs{ -1, -1 }; ///< Origin of each complete prepared cache, under the copy lock.
    std::array<std::int64_t, 2> m_previousOriginNs{ -1, -1 }; ///< Source age of the preceding still-pending cache.
    MagAOX::app::synco::Request m_last; ///< Suppress duplicate polls exactly as the existing producer helper.
    std::atomic<std::uint64_t>  m_prepared{ 0 }, m_commands{ 0 }, m_missed{ 0 }, m_busy{ 0 }; ///< Private outcomes.
    std::atomic<std::int64_t> m_callbackNs{ -1 }, m_peakNs{ -1 }; ///< Copy/validation measurements, not deadline proof.

    /// Copy the newest eligible target, retaining older pending slots; caller owns the short copy lock.
    bool copy( const MagAOX::app::synco::Request &request /**< [in] Exact base request. */,
               const MagAOX::app::synco::Record  &schedule /**< [in] Current coherent schedule. */,
               std::int64_t                       nowNs /**< [in] Actual copy time. */,
               std::span<float>                   pixels /**< [out] Caller-owned complete image. */ );

  public:
    /// Allocate current, preceding and work buffers per model, plus the sole private callback image.
    SyncoShadowProducer( std::size_t elements /**< [in] Complete auxiliary image. */,
                         unsigned    frames /**< [in] Future command frames. */ );

    /// Render ahead on the numerical thread, then exchange ownership without holding the model-work lock.
    bool prepare( ShadowMode                        mode /**< [in] Independent model. */,
                  const Matrix                     &plan /**< [in] Native newest-first plan. */,
                  const Matrix                     &shapes /**< [in] Reviewed full-image shapes. */,
                  const MagAOX::app::synco::Record &schedule /**< [in] Original schedule. */,
                  std::int64_t                      gridNs /**< [in] Reviewed first command target. */,
                  std::int64_t                      nowNs /**< [in] Actual preparation time. */,
                  std::int64_t originNs = -1 /**< [in] Actual source decision time; default uses preparation time. */ );

    /// Set diagnostic selection and readiness without granting physical output permission.
    void models( ShadowMode requested /**< [in] Requested mode. */,
                 bool       standard /**< [in] Standard ready. */,
                 bool       assisted /**< [in] Assisted ready and currently synchronized. */ );

    /// Revoke assistance or both models immediately on acquisition loss, without waiting for the worker.
    void
    inhibit( bool         assistedOnly /**< [in] ADC-only loss preserves the standard diagnostic path. */,
             std::int64_t lossNs = -1 /**< [in] Actual acquisition loss time; old queued origins cannot rearm. */ );

    /// Same Request/payload selection used after a future base poll, currently entirely private and non-actuating.
    bool callback( const MagAOX::app::synco::Record &schedule /**< [in] Current coherent schedule. */,
                   std::int64_t                      nowNs /**< [in] Actual callback time. */ );

    /// Supply pixels for a Request obtained from the existing base; never attach, commit, acknowledge or publish.
    bool command( const MagAOX::app::synco::Request &request /**< [in] Exact base Request. */,
                  const MagAOX::app::synco::Record  &schedule /**< [in] Current scheduler record. */,
                  std::int64_t                       nowNs /**< [in] Actual preparation readout time. */,
                  std::span<float>                   pixels /**< [out] Caller-owned complete private image. */ );

    /// Copy bounded scalar outcomes; never reports physical application or transport acknowledgement.
    void status( OnlineShadowStatus &status /**< [out] Existing compact worker diagnostics. */ ) const;
};
} // namespace DDSPC
#endif
