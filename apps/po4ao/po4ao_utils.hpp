/** \file po4ao_utils.hpp
 * \brief Allocation-free history and command-safety utilities for PO4AO.
 *
 * \ingroup app_files
 */

#ifndef po4ao_utils_hpp
#define po4ao_utils_hpp

#include <algorithm>
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
inline Model *takePendingModel( std::atomic<Model *> &pending /**< [in,out] Candidate awaiting activation. */,
                               std::atomic<Model *> &retired /**< [in,out] Model awaiting background destruction. */,
                               std::atomic<int> &reloadStatus /**< [in,out] Reload state; stale candidates return to idle. */,
                               uint64_t generation /**< [in] Currently accepted reload generation. */ )
{
    if( retired.load( std::memory_order_acquire ) != nullptr )
        return nullptr;
    Model *candidate = pending.exchange( nullptr, std::memory_order_acq_rel );
    if( candidate != nullptr && candidate->generation != generation )
    {
        retired.store( candidate, std::memory_order_release );
        reloadStatus.store( 0, std::memory_order_release );
        return nullptr;
    }
    return candidate;
}

/// Report whether the current source counter directly follows the previous one.
inline bool sourceCounterIsConsecutive( uint64_t current /**< [in] Current source publication counter. */,
                                        uint64_t previous /**< [in] Previously processed source counter. */,
                                        bool     havePrevious /**< [in] Whether the previous counter is valid. */ )
{
    return !havePrevious || current == previous + 1;
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
        return false;

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
                    size_t historySize /**< [in] Number of samples retained. */ )
        : m_sampleSize( sampleSize ), m_historySize( historySize ),
          m_buffer( std::make_unique<float[]>( m_sampleSize * m_historySize ) )
    {
    }

    /// Exercise wraparound and indexing behavior without external dependencies.
    static bool selfTest()
    {
        CircularBuffer buf( 4, 3 );
        float          f1[4] = { 1.F, 2.F, 3.F, 4.F };
        float          f2[4] = { 10.F, 11.F, 12.F, 13.F };
        float          f3[4] = { 20.F, 21.F, 22.F, 23.F };
        float          f4[4] = { 30.F, 31.F, 32.F, 33.F };

        buf.add( f1 );
        if( buf.size() != 1 || buf.getItem( 0 )[0] != 1.F )
            return false;
        buf.add( f2 );
        buf.add( f3 );
        if( buf.size() != 3 || buf.getItem( 0 )[0] != 20.F || buf.getItem( 2 )[0] != 1.F )
            return false;
        buf.add( f4 );
        return buf.size() == 3 && buf.getItem( 2 )[0] == 10.F && buf.getItem( 3 ) == nullptr;
    }

    /// Add one sample, overwriting the oldest sample after the buffer fills.
    void add( const float *sample /**< [in] Contiguous sample data. */ )
    {
        if( m_historySize == 0 )
            return;
        memcpy( &m_buffer[( m_head % m_historySize ) * m_sampleSize], sample, sizeof( float ) * m_sampleSize );
        ++m_head;
    }

    /// Reset the logical contents without reallocating storage.
    void reset()
    {
        m_head = 0;
    }

    /// Return the number of valid samples currently retained.
    size_t size() const
    {
        return std::min( m_head, m_historySize );
    }

    /// Return a sample by age, where zero is the newest sample.
    const float *getItem( size_t age /**< [in] Sample age relative to the newest sample. */ ) const
    {
        if( m_historySize == 0 || age >= size() )
            return nullptr;
        const size_t index = ( m_head + m_historySize - 1 - age ) % m_historySize;
        return m_buffer.get() + index * m_sampleSize;
    }
};

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

    /// Allocate storage once before real-time processing starts.
    void configure( size_t frameCapacity /**< [in] Frames per episode. */,
                    size_t modeCount /**< [in] Modal values per frame. */ )
    {
        capacity = frameCapacity;
        modes    = modeCount;
        observations.resize( capacity * modes );
        commands.resize( capacity * modes );
        reset();
    }

    /// Clear logical contents without releasing storage.
    void reset()
    {
        count      = 0;
        episode    = 0;
        generation = 0;
    }

    /// Append one observation-command pair.
    bool add( const float *observation /**< [in] Modal observation. */,
              const float *command /**< [in] Modal differential command. */ )
    {
        if( count >= capacity )
            return false;
        memcpy( observations.data() + count * modes, observation, modes * sizeof( float ) );
        memcpy( commands.data() + count * modes, command, modes * sizeof( float ) );
        ++count;
        return true;
    }
};

/// Assemble the exported modal-policy tensor in its documented feature order.
inline bool assemblePolicyInput( float                *destination /**< [out] Contiguous policy input. */,
                                 size_t                modes /**< [in] Modal values per feature. */,
                                 size_t                historyLength /**< [in] Observation history length. */,
                                 const CircularBuffer &observations /**< [in] Observation history. */,
                                 const CircularBuffer &commands /**< [in] Command history. */ )
{
    if( destination == nullptr || historyLength == 0 || observations.size() < historyLength ||
        commands.size() + 1 < historyLength )
        return false;

    size_t feature = 0;
    memcpy( destination + feature++ * modes, observations.getItem( 0 ), modes * sizeof( float ) );
    for( size_t age = historyLength - 1; age > 0; --age )
        memcpy( destination + feature++ * modes, observations.getItem( age ), modes * sizeof( float ) );
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
        destination[byte] = source[byte];
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
        return false;
    if( policyBlend == 1.F )
        return true;

    const float fallbackBlend = 1.F - policyBlend;
    for( size_t mode = 0; mode < controlledModes; ++mode )
        command[mode] = policyBlend * command[mode] + fallbackBlend * integratorGain * observation[mode];
    return true;
}

/// Reject non-finite commands and enforce the per-mode safety limit.
inline bool limitCommand( float *command /**< [in,out] Differential modal command. */,
                          size_t modes /**< [in] Number of modal values. */,
                          float  absoluteLimit /**< [in] Maximum absolute modal value. */ )
{
    if( command == nullptr || modes == 0 || absoluteLimit <= 0.F )
        return false;

    for( size_t mode = 0; mode < modes; ++mode )
    {
        if( !finiteFloat( command[mode] ) )
            return false;
        command[mode] = std::max( -absoluteLimit, std::min( absoluteLimit, command[mode] ) );
    }
    return true;
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
        return false;

    for( size_t mode = controlledModes; mode < totalModes; ++mode )
    {
        if( !finiteFloat( observation[mode] ) )
            return false;
        command[mode] = observation[mode];
    }
    return true;
}

#endif // po4ao_utils_hpp
