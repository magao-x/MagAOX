/** \file syncoPublicationClock.hpp
 * \brief Compatibility name for the shared publication-clock mapping helper.
 */
#ifndef dmSyncopation_syncoPublicationClock_hpp
#define dmSyncopation_syncoPublicationClock_hpp

#include "../../libMagAOX/common/publicationClock.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{
/// Preserve the synco API while sharing clock qualification with other diagnostics.
using PublicationClock = MagAOX::timing::PublicationClock;
} // namespace synco
} // namespace app
} // namespace MagAOX
#endif // dmSyncopation_syncoPublicationClock_hpp
