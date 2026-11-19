/** \file po4ao_utils.hpp
 * \brief History, command safety and bounded callback sampling for PO4AO.
 *
 * \ingroup po4ao_files
 */

#ifndef po4ao_utils_hpp
#define po4ao_utils_hpp

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

/// Take a current-generation model, retiring stale candidates outside the real-time thread.
/** Model must expose the reload request's `generation` member. A busy retirement
 * slot leaves the candidate pending rather than overwriting an owned pointer.
 */
template <typename Model>
inline Model *
takePendingModel( std::atomic<Model *> &pending /**< [in,out] Candidate awaiting activation. */,
                  std::atomic<Model *> &retired /**< [in,out] Model awaiting background destruction. */,
                  std::atomic<int>     &reloadStatus /**< [in,out] Reload state; stale candidates return to idle. */,
                  uint64_t              generation /**< [in] Currently accepted reload generation. */ )
{
    if( retired.load( std::memory_order_acquire ) != nullptr )
    {
        return nullptr;
    }
    Model *candidate = pending.exchange( nullptr, std::memory_order_acq_rel );
    if( candidate != nullptr && candidate->generation != generation )
    {
        retired.store( candidate, std::memory_order_release );
        reloadStatus.store( 0, std::memory_order_release );
        return nullptr;
    }
    return candidate;
}

/// Identify a queued semaphore notification for an already processed source frame.
inline bool sourceCounterIsDuplicate( uint64_t current /**< [in] Current source publication counter. */,
                                      uint64_t previous /**< [in] Previously processed source counter. */,
                                      bool     havePrevious /**< [in] Whether the previous counter is valid. */ )
{
    return havePrevious && current == previous;
}

/// Report whether the current source counter directly follows the previous one.
inline bool sourceCounterIsConsecutive( uint64_t current /**< [in] Current source publication counter. */,
                                        uint64_t previous /**< [in] Previously processed source counter. */,
                                        bool     havePrevious /**< [in] Whether the previous counter is valid. */ )
{
    return !havePrevious || current == previous + 1;
}

/// Collection lifecycle shared by the frame producer and replay publisher.
enum class CollectionPhase : uint64_t
{
    idle        = 0, ///< No collection has started.
    collecting  = 1, ///< Accept samples and explore once controller history is ready.
    publishing  = 2, ///< All commands were emitted; the frozen snapshot is pending.
    ready       = 3, ///< A complete snapshot was published.
    interrupted = 4  ///< This generation cannot produce a valid warmup snapshot.
};

/// Track a fixed-controller collection without owning or clearing controller history.
/** The frame producer alone calls begin, validate, discard and emitted. The publisher may
 * interrupt or acknowledge publication. The generation and phase share one atomic
 * token so a late publisher cannot complete or interrupt a newer collection.
 */
class WarmupCollection
{
  private:
    std::atomic<uint64_t> m_token{ 0 };   ///< Collection generation and phase, published together.
    std::atomic<uint64_t> m_frames{ 0 };  ///< Accepted samples, excluding discarded partial episodes.
    uint64_t              m_target{ 0 };  ///< Requested frame count, owned by the frame producer.
    uint64_t              m_control{ 0 }; ///< Captured control request identity, including enable state.
    uint64_t              m_model{ 0 };   ///< Captured selected policy hash; zero means integrator.
    uint64_t              m_reset{ 0 };   ///< Captured model-reset generation.

  public:
    /// Start one collection, without touching history or model state.
    void begin( uint64_t generation /**< [in] Monotonically increasing collection identity. */,
                uint64_t target /**< [in] Number of accepted samples required. */,
                uint64_t control /**< [in] Current control request identity. */,
                uint64_t model /**< [in] Selected policy hash, or zero for integrator. */,
                uint64_t reset /**< [in] Current model-reset generation. */ );

    /// Return the generation and phase in one atomic snapshot.
    uint64_t token() const;

    /// Decode the phase of a previously read token.
    static CollectionPhase phase( uint64_t token /**< [in] Atomic collection snapshot. */ );

    /// Decode the collection identity of a previously read token.
    static uint64_t generation( uint64_t token /**< [in] Atomic collection snapshot. */ );

    /// Return the accepted sample count, including the current partial episode.
    uint64_t frames() const;

    /// Interrupt a pending collection if its selected controller changed.
    void validate( uint64_t control /**< [in] Current control request identity. */,
                   uint64_t model /**< [in] Selected policy hash, independent of temporary history refill. */,
                   uint64_t reset /**< [in] Current model-reset generation. */ );

    /// Reuse the exploration indices of a discarded partial episode; preserve completed collections.
    void discard( uint64_t frames /**< [in] Accepted samples in the discarded partial episode. */ );

    /// Advance the exploration schedule only after accepting a successfully emitted sample.
    void emitted();

    /// Invalidate a collecting or pending-publication generation; preserve an already completed snapshot.
    void interrupt( uint64_t generation /**< [in] Collection identity to interrupt. */ );

    /// Acknowledge a complete frozen snapshot only for the matching pending generation.
    bool published( uint64_t generation /**< [in] Identity of the successfully published snapshot. */ );
};

inline void
WarmupCollection::begin( uint64_t generation, uint64_t target, uint64_t control, uint64_t model, uint64_t reset )
{
    m_target  = target;
    m_control = control;
    m_model   = model;
    m_reset   = reset;
    m_frames.store( 0, std::memory_order_release );
    m_token.store( ( generation << 3 ) |
                       static_cast<uint64_t>( target ? CollectionPhase::collecting : CollectionPhase::interrupted ),
                   std::memory_order_release );
}

inline uint64_t WarmupCollection::token() const
{
    return m_token.load( std::memory_order_acquire );
}

inline CollectionPhase WarmupCollection::phase( uint64_t token )
{
    return static_cast<CollectionPhase>( token & 7 );
}

inline uint64_t WarmupCollection::generation( uint64_t token )
{
    return token >> 3;
}

inline uint64_t WarmupCollection::frames() const
{
    return m_frames.load( std::memory_order_acquire );
}

inline void WarmupCollection::validate( uint64_t control, uint64_t model, uint64_t reset )
{
    if( control != m_control || model != m_model || reset != m_reset )
    {
        interrupt( generation( token() ) );
    }
}

inline void WarmupCollection::discard( uint64_t frames )
{
    if( phase( token() ) == CollectionPhase::collecting )
    {
        m_frames.fetch_sub( frames, std::memory_order_acq_rel );
    }
}

inline void WarmupCollection::emitted()
{
    uint64_t current = token();
    if( phase( current ) != CollectionPhase::collecting )
    {
        return;
    }
    if( m_frames.fetch_add( 1, std::memory_order_acq_rel ) + 1 >= m_target )
    {
        m_token.compare_exchange_strong( current,
                                         ( generation( current ) << 3 ) |
                                             static_cast<uint64_t>( CollectionPhase::publishing ),
                                         std::memory_order_acq_rel );
    }
}

inline void WarmupCollection::interrupt( uint64_t generation )
{
    uint64_t current = token();
    while( WarmupCollection::generation( current ) == generation &&
           ( phase( current ) == CollectionPhase::collecting || phase( current ) == CollectionPhase::publishing ) )
    {
        if( m_token.compare_exchange_weak( current,
                                           ( generation << 3 ) | static_cast<uint64_t>( CollectionPhase::interrupted ),
                                           std::memory_order_acq_rel ) )
        {
            break;
        }
    }
}

inline bool WarmupCollection::published( uint64_t generation )
{
    uint64_t expected = ( generation << 3 ) | static_cast<uint64_t>( CollectionPhase::publishing );
    return m_token.compare_exchange_strong(
        expected, ( generation << 3 ) | static_cast<uint64_t>( CollectionPhase::ready ), std::memory_order_acq_rel );
}

/// Evaluate independent per-mode linear temporal filters without CUDA.
inline bool applyLinearLocalPolicy( float       *output /**< [out] Controlled modal commands. */,
                                    const float *input /**< [in] Feature-major modal input. */,
                                    const float *weights /**< [in] Feature-major modal weights. */,
                                    size_t       features /**< [in] Temporal feature count. */,
                                    size_t       trainedModes /**< [in] Modal stride in input and weights. */,
                                    size_t       controlledModes /**< [in] Leading modes to evaluate. */ )
{
    if( output == nullptr || input == nullptr || weights == nullptr || features == 0 || controlledModes == 0 ||
        controlledModes > trainedModes )
    {
        return false;
    }

    for( size_t mode = 0; mode < controlledModes; ++mode )
    {
        float command = 0.F;
        for( size_t feature = 0; feature < features; ++feature )
        {
            const size_t index = feature * trainedModes + mode;
            command += input[index] * weights[index];
        }
        output[mode] = command;
    }
    return true;
}

/// Fixed-capacity float history used by one thread at a time.
class CircularBuffer
{
  private:
    size_t                   m_sampleSize{ 0 };  ///< Number of floats in one sample.
    size_t                   m_historySize{ 0 }; ///< Maximum number of samples retained.
    std::unique_ptr<float[]> m_buffer;           ///< Contiguous sample storage.
    size_t                   m_head{ 0 };        ///< Monotonic index of the next write.

  public:
    /// Construct a circular buffer with fixed sample size and capacity.
    CircularBuffer( size_t sampleSize /**< [in] Number of floats in one sample. */,
                    size_t historySize /**< [in] Number of samples retained. */ );

    /// Exercise wraparound and indexing behavior without external dependencies.
    static bool selfTest();

    /// Add one sample, overwriting the oldest sample after the buffer fills.
    void add( const float *sample /**< [in] Contiguous sample data. */ );

    /// Reset the logical contents without reallocating storage.
    void reset();

    /// Return the number of valid samples currently retained.
    size_t size() const;

    /// Return a sample by age, where zero is the newest sample.
    const float *getItem( size_t age /**< [in] Sample age relative to the newest sample. */ ) const;
};

inline CircularBuffer::CircularBuffer( size_t sampleSize, size_t historySize )
    : m_sampleSize( sampleSize ), m_historySize( historySize ),
      m_buffer( std::make_unique<float[]>( m_sampleSize * m_historySize ) )
{
}

inline bool CircularBuffer::selfTest()
{
    CircularBuffer buf( 4, 3 );
    float          f1[4] = { 1.F, 2.F, 3.F, 4.F };
    float          f2[4] = { 10.F, 11.F, 12.F, 13.F };
    float          f3[4] = { 20.F, 21.F, 22.F, 23.F };
    float          f4[4] = { 30.F, 31.F, 32.F, 33.F };

    buf.add( f1 );
    if( buf.size() != 1 || buf.getItem( 0 )[0] != 1.F )
    {
        return false;
    }
    buf.add( f2 );
    buf.add( f3 );
    if( buf.size() != 3 || buf.getItem( 0 )[0] != 20.F || buf.getItem( 2 )[0] != 1.F )
    {
        return false;
    }
    buf.add( f4 );
    return buf.size() == 3 && buf.getItem( 2 )[0] == 10.F && buf.getItem( 3 ) == nullptr;
}

inline void CircularBuffer::add( const float *sample )
{
    if( m_historySize == 0 )
    {
        return;
    }
    memcpy( &m_buffer[( m_head % m_historySize ) * m_sampleSize], sample, sizeof( float ) * m_sampleSize );
    ++m_head;
}

inline void CircularBuffer::reset()
{
    m_head = 0;
}

inline size_t CircularBuffer::size() const
{
    return std::min( m_head, m_historySize );
}

inline const float *CircularBuffer::getItem( size_t age ) const
{
    if( m_historySize == 0 || age >= size() )
    {
        return nullptr;
    }
    const size_t index = ( m_head + m_historySize - 1 - age ) % m_historySize;
    return m_buffer.get() + index * m_sampleSize;
}

/// One preallocated episode passed from the real-time thread to the publisher thread.
struct EpisodeBuffer
{
    std::vector<float> observations;    ///< Time-major modal observations.
    std::vector<float> commands;        ///< Time-major differential commands.
    size_t             capacity{ 0 };   ///< Maximum frames held by this episode slot.
    size_t             modes{ 0 };      ///< Modal values in each frame.
    size_t             count{ 0 };      ///< Valid frames currently stored.
    uint64_t           episode{ 0 };    ///< One-based episode identifier.
    uint64_t           generation{ 0 }; ///< Warm-up collection generation containing this episode.
    uint64_t           modelId{ 0 };    ///< Effective policy at the final emitted frame; zero is integrator.

    /// Allocate storage once before real-time processing starts.
    void configure( size_t frameCapacity /**< [in] Frames per episode. */,
                    size_t modeCount /**< [in] Modal values per frame. */ );

    /// Clear logical contents without releasing storage.
    void reset();

    /// Append one observation-command pair.
    bool add( const float *observation /**< [in] Modal observation. */,
              const float *command /**< [in] Modal differential command. */ );
};

inline void EpisodeBuffer::configure( size_t frameCapacity, size_t modeCount )
{
    capacity = frameCapacity;
    modes    = modeCount;
    observations.resize( capacity * modes );
    commands.resize( capacity * modes );
    reset();
}

inline void EpisodeBuffer::reset()
{
    count      = 0;
    episode    = 0;
    generation = 0;
    modelId    = 0;
}

inline bool EpisodeBuffer::add( const float *observation, const float *command )
{
    if( count >= capacity )
    {
        return false;
    }
    memcpy( observations.data() + count * modes, observation, modes * sizeof( float ) );
    memcpy( commands.data() + count * modes, command, modes * sizeof( float ) );
    ++count;
    return true;
}

/// Assemble the exported modal-policy tensor in its documented feature order.
inline bool assemblePolicyInput( float                *destination /**< [out] Contiguous policy input. */,
                                 size_t                modes /**< [in] Modal values per feature. */,
                                 size_t                historyLength /**< [in] Observation history length. */,
                                 const CircularBuffer &observations /**< [in] Observation history. */,
                                 const CircularBuffer &commands /**< [in] Command history. */ )
{
    if( destination == nullptr || historyLength == 0 || observations.size() < historyLength ||
        commands.size() + 1 < historyLength )
    {
        return false;
    }

    size_t feature = 0;
    memcpy( destination + feature++ * modes, observations.getItem( 0 ), modes * sizeof( float ) );
    for( size_t age = historyLength - 1; age > 0; --age )
    {
        memcpy( destination + feature++ * modes, observations.getItem( age ), modes * sizeof( float ) );
    }
    for( size_t offset = 0; offset + 1 < historyLength; ++offset )
    {
        const size_t age = historyLength - 2 - offset;
        memcpy( destination + feature++ * modes, commands.getItem( age ), modes * sizeof( float ) );
    }
    return true;
}

/// Test an IEEE-754 float for finiteness without relying on fast-math-sensitive classification builtins.
inline bool finiteFloat( float value /**< [in] Value to classify. */ )
{
    static_assert( sizeof( float ) == sizeof( uint32_t ) && std::numeric_limits<float>::is_iec559,
                   "PO4AO requires IEEE-754 binary32 floats" );
    uint32_t                      bits        = 0;
    auto                         *destination = reinterpret_cast<unsigned char *>( &bits );
    const volatile unsigned char *source      = reinterpret_cast<const volatile unsigned char *>( &value );
    for( size_t byte = 0; byte < sizeof( bits ); ++byte )
    {
        destination[byte] = source[byte];
    }
    return ( bits & 0x7F800000U ) != 0x7F800000U;
}

/// Blend a learned policy command with the proportional fallback command.
inline bool blendPolicyCommand( float       *command /**< [in,out] Learned command replaced by the blend. */,
                                const float *observation /**< [in] Current modal observation. */,
                                size_t       controlledModes /**< [in] Leading modes to blend. */,
                                float        integratorGain /**< [in] Proportional fallback gain. */,
                                float        policyBlend /**< [in] Learned-policy fraction in [0, 1]. */ )
{
    if( command == nullptr || observation == nullptr || controlledModes == 0 || !finiteFloat( integratorGain ) ||
        !finiteFloat( policyBlend ) || policyBlend < 0.F || policyBlend > 1.F )
    {
        return false;
    }
    if( policyBlend == 1.F )
    {
        return true;
    }

    const float fallbackBlend = 1.F - policyBlend;
    for( size_t mode = 0; mode < controlledModes; ++mode )
    {
        command[mode] = policyBlend * command[mode] + fallbackBlend * integratorGain * observation[mode];
    }
    return true;
}

/// Reject non-finite commands and enforce the per-mode safety limit.
inline bool limitCommand( float *command /**< [in,out] Differential modal command. */,
                          size_t modes /**< [in] Number of modal values. */,
                          float  absoluteLimit /**< [in] Maximum absolute modal value. */ )
{
    if( command == nullptr || modes == 0 || absoluteLimit <= 0.F )
    {
        return false;
    }

    for( size_t mode = 0; mode < modes; ++mode )
    {
        if( !finiteFloat( command[mode] ) )
        {
            return false;
        }
        command[mode] = std::max( -absoluteLimit, std::min( absoluteLimit, command[mode] ) );
    }
    return true;
}

/// Add one scheduled exploration sample to the base command before applying command limits.
inline void addExploration( float       *command /**< [in,out] Base controller output. */,
                            size_t       modes /**< [in] Controlled modal prefix. */,
                            const float *noise /**< [in] Selected row of the configured noise table. */,
                            uint64_t     frame /**< [in] Zero-based emitted collection frame. */,
                            uint64_t     total /**< [in] Total collection frames. */,
                            float        start /**< [in] Initial exploration scale. */,
                            float        end /**< [in] Final exploration scale. */ )
{
    const float progress = total > 1 ? std::min( 1.F, static_cast<float>( frame ) / ( total - 1 ) ) : 1.F;
    const float sigma    = start + progress * ( end - start );
    for( size_t mode = 0; mode < modes; ++mode )
    {
        command[mode] += sigma * noise[mode];
    }
}

/// Limit the controlled modal prefix and copy all remaining observations unchanged.
inline bool finalizeControlledCommand( float       *command /**< [in,out] Full-sized output command. */,
                                       const float *observation /**< [in] Full-sized current observation. */,
                                       size_t       totalModes /**< [in] Total stream mode count. */,
                                       size_t       controlledModes /**< [in] Leading modes controlled by PO4AO. */,
                                       float        absoluteLimit /**< [in] Maximum absolute controlled-mode value. */ )
{
    if( command == nullptr || observation == nullptr || controlledModes == 0 || controlledModes > totalModes ||
        !limitCommand( command, controlledModes, absoluteLimit ) )
    {
        return false;
    }

    for( size_t mode = controlledModes; mode < totalModes; ++mode )
    {
        if( !finiteFloat( observation[mode] ) )
        {
            return false;
        }
        command[mode] = observation[mode];
    }
    return true;
}

namespace MagAOX::app
{
/// One consumer snapshot; durations are callback microseconds, not end-to-end latency.
struct Po4aoStatistics
{
    uint32_t p50{ 0 }, p95{ 0 }, p99{ 0 }, maximum{ 0 }; ///< Rolling-window duration percentiles.
    uint64_t count{ 0 }, sequence{ 0 },
        lastTimeNs{ 0 };                 ///< Valid samples, producer sequence and monotonic sample time.
    uint64_t model{ 0 };                 ///< Identity of the newest included sample.
    uint32_t backend{ 0 };               ///< Backend of the newest included sample.
    bool mixed{ false }, fresh{ false }; ///< Window spans controller identities, or contains newly published samples.
};

/// Single-producer ring; only the non-real-time consumer allocates and sorts.
class Po4aoSamples
{
    struct Sample
    {
        std::atomic<uint64_t> sequence{ 0 }, model{ 0 },
            timeNs{ 0 };                                   ///< Publication stamp and controller identity/time.
        std::atomic<uint32_t> duration{ 0 }, backend{ 0 }; ///< Callback microseconds and fallback/CPU/TensorRT code.
    };
    std::array<Sample, 1024> m_samples;       ///< Fixed storage; zero marks a slot being replaced.
    std::atomic<uint64_t>    m_sequence{ 0 }; ///< Last completely published sample.

  public:
    /// Return the latest completed callback sequence without copying the timing window.
    uint64_t sequence() const;

    /// Append one callback sample without allocation or locking.
    void record( uint32_t duration /**< [in] Callback duration in microseconds. */,
                 uint64_t model /**< [in] Effective model ID. */,
                 uint32_t backend /**< [in] 0 fallback, 1 CPU, 2 TensorRT. */,
                 uint64_t timeNs /**< [in] Monotonic callback completion time. */ );

    /// Read a consistent subset of the rolling window, flagging mixed controller identities.
    Po4aoStatistics snapshot( uint64_t previous /**< [in] Sequence in the preceding telemetry record. */ ) const;
};

inline uint64_t Po4aoSamples::sequence() const
{
    return m_sequence.load();
}

inline void Po4aoSamples::record( uint32_t duration, uint64_t model, uint32_t backend, uint64_t timeNs )
{
    const uint64_t next   = m_sequence.load() + 1;
    auto          &sample = m_samples[( next - 1 ) % m_samples.size()];
    // Sequentially consistent atomic fields keep wraparound reads free of data races.
    sample.sequence.store( 0 );
    sample.duration.store( duration );
    sample.model.store( model );
    sample.backend.store( backend );
    sample.timeNs.store( timeNs );
    sample.sequence.store( next );
    m_sequence.store( next );
}

inline Po4aoStatistics Po4aoSamples::snapshot( uint64_t previous ) const
{
    Po4aoStatistics result;
    result.sequence = m_sequence.load();
    result.fresh    = result.sequence != previous;
    std::vector<uint32_t> values;
    values.reserve( m_samples.size() );
    const uint64_t first = result.sequence > m_samples.size() ? result.sequence - m_samples.size() + 1 : 1;
    for( uint64_t sequence = first; sequence <= result.sequence; ++sequence )
    {
        const auto &sample = m_samples[( sequence - 1 ) % m_samples.size()];
        if( sample.sequence.load() != sequence )
        {
            continue;
        }
        const auto duration = sample.duration.load();
        const auto identity = sample.model.load();
        const auto kind     = sample.backend.load();
        const auto time     = sample.timeNs.load();
        if( sample.sequence.load() != sequence )
        {
            continue;
        }
        result.mixed |= !values.empty() && ( identity != result.model || kind != result.backend );
        result.model   = identity;
        result.backend = kind;
        values.push_back( duration ); // A sub-microsecond callback is a valid zero sample.
        result.lastTimeNs = std::max( result.lastTimeNs, time );
    }
    std::sort( values.begin(), values.end() );
    result.count = values.size();
    if( !values.empty() )
    {
        result.p50     = values[( values.size() * 50 + 99 ) / 100 - 1];
        result.p95     = values[( values.size() * 95 + 99 ) / 100 - 1];
        result.p99     = values[( values.size() * 99 + 99 ) / 100 - 1];
        result.maximum = values.back();
    }
    return result;
}
} // namespace MagAOX::app

#endif // po4ao_utils_hpp
