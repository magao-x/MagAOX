/** \file cacaoWriter.hpp
 * \brief Bounded CACAO writer requests and observed FPS process identity.
 */
#ifndef nnReconstructor_cacaoWriter_hpp
#define nnReconstructor_cacaoWriter_hpp

#include <cstdint>
#include <string>

namespace MagAOX::app
{

/// Use milk's installed FPS metadata layout; never infer completion from FIFO delivery.
class cacaoWriter
{
  public:
    std::string m_name{ "wfs2cmodeval-1" }; ///< Configured FPS writer name.

    std::string m_fifo{ "/milk/shm/tweeter-vispyr_fpsCTRL.fifo" }; ///< Existing controller FIFO.

    std::string m_processMatch{ "wfs2cmodeval" }; ///< Expected RUN process command-line substring.

    /// Bind standard instrument paths or isolated fixture directories.
    explicit cacaoWriter( const std::string &shmDirectory  = "/milk/shm", /**< [in] FPS shared-memory directory. */
                          const std::string &procDirectory = "/proc" /**< [in] Process information directory. */ );

    /// Return verified RUN state, throwing when the metadata or process identity is unavailable.
    bool running();

    /// Deliver a single nonblocking request; the caller must observe completion separately.
    void command( bool enabled /**< [in] Start rather than stop the configured writer. */ );

  private:
    std::string m_shmDirectory; ///< Directory containing FPS metadata.

    std::string m_procDirectory; ///< Directory containing process stat/cmdline records.

    std::string m_identity; ///< FPS file, RUN PID and start timestamp from the previous observation.

    std::string m_startTicks; ///< Linux process start time paired with that FPS identity.
};

} // namespace MagAOX::app
#endif
