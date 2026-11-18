/** \file telem_gpu.hpp
 * \brief Indexed GPU diagnostics with explicit per-field validity and legacy identity decoding.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_gpu_hpp
#define logger_types_telem_gpu_hpp

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "generated/telem_gpu_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Record a GPU query and the current validity of each indexed device's measurements.
/** \ingroup logger_types */
struct telem_gpu : public flatbuffer_log
{
    /// Logger dispatch code.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_GPU;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Last emission time maintained by the telemetry scheduler.
    static timespec lastRecord;

    /// No reported NVIDIA index, including records written before indices were recorded.
    static constexpr uint32_t UnknownIndex = std::numeric_limits<uint32_t>::max();

    /// Stable validity bits for the eight measurement fields.
    enum validBits : uint32_t
    {
        Temperature   = 1U << 0, ///< Temperature is available.
        Load          = 1U << 1, ///< GPU kernel-active fraction is available.
        MemoryLoad    = 1U << 2, ///< Memory read/write-active fraction is available.
        MemoryUsed    = 1U << 3, ///< Occupied device memory is available.
        MemoryTotal   = 1U << 4, ///< Total device memory is available.
        ClockGraphics = 1U << 5, ///< Graphics clock is available.
        ClockSM       = 1U << 6, ///< SM clock is available.
        ClockMemory   = 1U << 7, ///< Memory clock is available.
        AllValid      = 255U     ///< All measurement fields are available.
    };

    /// One device's measurements; numeric values are meaningful only when their validity bit is set.
    struct gpuT
    {
        std::string m_uuid;   ///< Stable NVIDIA GPU UUID, retained in memory and decoded from legacy records only.
        std::string m_pci;    ///< PCI bus address, retained in memory and decoded from legacy records only.
        std::string m_status; ///< Acquisition state: ok, partial, or a collector-supplied failure state.

        double m_temperature{ 0 };   ///< GPU temperature in Celsius.
        double m_load{ 0 };          ///< Fraction of sampling time executing kernels, 0..1.
        double m_memoryLoad{ 0 };    ///< Fraction of sampling time reading/writing memory, 0..1.
        double m_memoryUsed{ 0 };    ///< Occupied device memory in MiB.
        double m_memoryTotal{ 0 };   ///< Total device memory in MiB.
        double m_clockGraphics{ 0 }; ///< Graphics clock in MHz.
        double m_clockSM{ 0 };       ///< SM clock in MHz.
        double m_clockMemory{ 0 };   ///< Memory clock in MHz.

        uint32_t m_valid{ 0 };      ///< Bit mask identifying currently available measurement fields.
        double   m_sampleTime{ 0 }; ///< Unix UTC seconds of the last successful query for this device.

        uint32_t m_index{ UnknownIndex }; ///< Associated NVIDIA index, or UnknownIndex; UUID is the stable identity.

        /// Compare measurement state while ignoring the acquisition time for change-triggered recording.
        bool operator==( const gpuT &other /**< [in] device state to compare */ ) const;
    };

    /// Serialized query state and indexed measurements, excluding static device identity strings.
    struct messageT : public fbMessage
    {
        /// Serialize device indices, measurements, invalidation state, and last successful sample times.
        /** UUID and PCI strings are omitted; discovery and mapping changes belong in separate diagnostic logs. */
        messageT( const std::string       &status /**< [in] overall query state */,
                  const std::vector<gpuT> &gpus /**< [in] identified device measurements */ );
    };

    /// Verify a serialized GPU telemetry payload and its validity masks.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] serialized payload length */ );

    /// Format device indices, units, validity, sample times, and any legacy identity strings.
    static std::string msgString( void             *buffer /**< [in] serialized payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );

    /// Decode all measurements and timestamps; UUID/PCI strings are empty unless present in legacy records.
    static std::vector<gpuT> gpus( void *buffer /**< [in] serialized payload */ );

    /// Get the overall acquisition state.
    static std::string status( void *buffer /**< [in] serialized payload */ );

    /// Get metadata accessors; numeric vectors share ordering with device indices and validity masks.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );

  private:
    /// Extract one numeric field in device order; callers must also inspect the validity vector.
    template <typename valueT, auto member>
    static std::vector<valueT> numbers( void *buffer /**< [in] serialized payload */ );

    /// Extract one string field in device order.
    template <auto member>
    static std::vector<std::string> strings( void *buffer /**< [in] serialized payload */ );
};

inline bool telem_gpu::gpuT::operator==( const gpuT &other ) const
{
    return std::tie( m_uuid,
                     m_pci,
                     m_status,
                     m_temperature,
                     m_load,
                     m_memoryLoad,
                     m_memoryUsed,
                     m_memoryTotal,
                     m_clockGraphics,
                     m_clockSM,
                     m_clockMemory,
                     m_valid,
                     m_index ) == std::tie( other.m_uuid,
                                            other.m_pci,
                                            other.m_status,
                                            other.m_temperature,
                                            other.m_load,
                                            other.m_memoryLoad,
                                            other.m_memoryUsed,
                                            other.m_memoryTotal,
                                            other.m_clockGraphics,
                                            other.m_clockSM,
                                            other.m_clockMemory,
                                            other.m_valid,
                                            other.m_index );
}

inline telem_gpu::messageT::messageT( const std::string &status, const std::vector<gpuT> &gpus )
{
    std::vector<flatbuffers::Offset<Gpu_sample_fb>> devices;
    devices.reserve( gpus.size() );

    for( const auto &gpu : gpus )
    {
        auto state = builder.CreateString( gpu.m_status );
        devices.push_back( CreateGpu_sample_fb( builder,
                                                0,
                                                0,
                                                state,
                                                gpu.m_temperature,
                                                gpu.m_load,
                                                gpu.m_memoryLoad,
                                                gpu.m_memoryUsed,
                                                gpu.m_memoryTotal,
                                                gpu.m_clockGraphics,
                                                gpu.m_clockSM,
                                                gpu.m_clockMemory,
                                                gpu.m_valid,
                                                gpu.m_sampleTime,
                                                gpu.m_index ) );
    }

    auto state   = builder.CreateString( status );
    auto samples = builder.CreateVector( devices );
    builder.Finish( CreateTelem_gpu_fb( builder, state, samples ) );
}

inline bool telem_gpu::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );

    if( !VerifyTelem_gpu_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *devices = GetTelem_gpu_fb( payload )->gpus();

    if( devices )
    {
        for( const auto *gpu : *devices )
        {
            if( gpu->valid() & ~AllValid )
            {
                return false;
            }
        }
    }

    return true;
}

inline std::string telem_gpu::status( void *buffer )
{
    return GetTelem_gpu_fb( buffer )->status()->str();
}

inline std::vector<telem_gpu::gpuT> telem_gpu::gpus( void *buffer )
{
    std::vector<gpuT> result;
    const auto       *devices = GetTelem_gpu_fb( buffer )->gpus();

    if( devices )
    {
        result.reserve( devices->size() );

        for( const auto *gpu : *devices )
        {
            result.push_back( { gpu->uuid() ? gpu->uuid()->str() : std::string{},
                                gpu->pci() ? gpu->pci()->str() : std::string{},
                                gpu->status()->str(),
                                gpu->temperature(),
                                gpu->load(),
                                gpu->memory_load(),
                                gpu->memory_used(),
                                gpu->memory_total(),
                                gpu->clock_graphics(),
                                gpu->clock_sm(),
                                gpu->clock_memory(),
                                gpu->valid(),
                                gpu->sample_time(),
                                gpu->index() } );
        }
    }

    return result;
}

inline std::string telem_gpu::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    std::ostringstream output;
    output << "[gpu] " << status( buffer );

    for( const auto &gpu : gpus( buffer ) )
    {
        output << " [index=" << ( gpu.m_index == UnknownIndex ? "unknown" : std::to_string( gpu.m_index ) );

        if( !gpu.m_uuid.empty() )
        {
            output << " uuid=" << gpu.m_uuid;
        }

        if( !gpu.m_pci.empty() )
        {
            output << " pci=" << gpu.m_pci;
        }

        output << " " << gpu.m_status << " valid=" << gpu.m_valid << " sample_time=" << std::setprecision( 17 )
               << gpu.m_sampleTime << " temperature=" << gpu.m_temperature << " C load=" << gpu.m_load
               << " memory_load=" << gpu.m_memoryLoad << " memory=" << gpu.m_memoryUsed << "/" << gpu.m_memoryTotal
               << " MiB clocks=" << gpu.m_clockGraphics << "/" << gpu.m_clockSM << "/" << gpu.m_clockMemory << " MHz]";
    }

    return output.str();
}

template <typename valueT, auto member>
inline std::vector<valueT> telem_gpu::numbers( void *buffer )
{
    std::vector<valueT> result;
    const auto         *devices = GetTelem_gpu_fb( buffer )->gpus();

    if( devices )
    {
        for( const auto *gpu : *devices )
        {
            result.push_back( ( gpu->*member )() );
        }
    }

    return result;
}

template <auto member>
inline std::vector<std::string> telem_gpu::strings( void *buffer )
{
    std::vector<std::string> result;
    const auto              *devices = GetTelem_gpu_fb( buffer )->gpus();

    if( devices )
    {
        for( const auto *gpu : *devices )
        {
            const auto *value = ( gpu->*member )();
            result.push_back( value ? value->str() : std::string{} );
        }
    }

    return result;
}

inline logMetaDetail telem_gpu::getAccessor( const std::string &member )
{
    if( member == "status" )
    {
        return { "GPU STATUS",
                 "GPU query acquisition state",
                 logMeta::String,
                 logMeta::State,
                 reinterpret_cast<void *>( &status ),
                 true };
    }

    if( member == "uuid" )
    {
        return { "GPU UUID",
                 "legacy device UUIDs in measurement order; empty in indexed records",
                 logMeta::Vector_String,
                 logMeta::State,
                 reinterpret_cast<void *>( &strings<&Gpu_sample_fb::uuid> ),
                 true };
    }

    if( member == "pci" )
    {
        return { "GPU PCI",
                 "legacy PCI addresses in measurement order; empty in indexed records",
                 logMeta::Vector_String,
                 logMeta::State,
                 reinterpret_cast<void *>( &strings<&Gpu_sample_fb::pci> ),
                 true };
    }

    if( member == "index" )
    {
        return { "GPU INDEX",
                 "NVIDIA indices in measurement order; 4294967295 means unknown",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<unsigned, &Gpu_sample_fb::index> ),
                 true };
    }

    if( member == "device_status" )
    {
        return { "GPU DEVICE STATUS",
                 "device acquisition states",
                 logMeta::Vector_String,
                 logMeta::State,
                 reinterpret_cast<void *>( &strings<&Gpu_sample_fb::status> ),
                 true };
    }

    if( member == "valid" )
    {
        return { "GPU VALID",
                 "measurement validity masks in device order",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<unsigned, &Gpu_sample_fb::valid> ),
                 true };
    }

    if( member == "sample_time" )
    {
        return { "GPU SAMPLE TIME",
                 "last successful device query [Unix UTC s]",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::sample_time> ),
                 true };
    }

    if( member == "temperature" )
    {
        return { "GPU TEMPERATURE",
                 "GPU temperatures [C]; requires validity bit 0",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::temperature> ),
                 true };
    }

    if( member == "load" )
    {
        return { "GPU LOAD",
                 "kernel-active fractions; requires validity bit 1",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::load> ),
                 true };
    }

    if( member == "memory_load" )
    {
        return { "GPU MEMORY LOAD",
                 "memory read/write-active fractions; requires validity bit 2",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::memory_load> ),
                 true };
    }

    if( member == "memory_used" )
    {
        return { "GPU MEMORY USED",
                 "occupied memory [MiB]; requires validity bit 3",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::memory_used> ),
                 true };
    }

    if( member == "memory_total" )
    {
        return { "GPU MEMORY TOTAL",
                 "total memory [MiB]; requires validity bit 4",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::memory_total> ),
                 true };
    }

    if( member == "clock_graphics" )
    {
        return { "GPU CLOCK GRAPHICS",
                 "graphics clocks [MHz]; requires validity bit 5",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::clock_graphics> ),
                 true };
    }

    if( member == "clock_sm" )
    {
        return { "GPU CLOCK SM",
                 "SM clocks [MHz]; requires validity bit 6",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::clock_sm> ),
                 true };
    }

    if( member == "clock_memory" )
    {
        return { "GPU CLOCK MEMORY",
                 "memory clocks [MHz]; requires validity bit 7",
                 logMeta::Vector_Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<double, &Gpu_sample_fb::clock_memory> ),
                 true };
    }

    return {};
}

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_gpu_hpp
