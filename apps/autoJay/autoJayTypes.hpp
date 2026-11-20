/** \file autoJayTypes.hpp
 * \brief Shared data for the dual-DM alignment sensor.
 * \ingroup autoJay_files
 */
#ifndef autoJayTypes_hpp
#define autoJayTypes_hpp

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <mx/improc/eigenImage.hpp>
#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/common/roiGeometry.hpp"

namespace MagAOX
{
namespace app
{
namespace autoJayDetail
{

/// Images use rows for the detector x coordinate, following MagAO-X streams.
using imageT = mx::improc::eigenImage<float>;

/// Applied geometry shared by all images in one acquisition.
struct geometry
{
    roi::region region;                           ///< Applied native ROI and binning.
    roi::units  units{ roi::units::unsupported }; ///< Native camera coordinate convention.
    std::string orientation;                      ///< Configured detector-to-stream orientation identity.
};

/// Averages and uncertainty from distinct source frames.
struct frameStack
{
    imageT   mean;              ///< Mean of accepted frames.
    imageT   variance;          ///< Estimated variance of the mean, not spatial image variance.
    uint64_t count{ 0 };        ///< Number of distinct accepted source frames.
    uint64_t firstCounter{ 0 }; ///< First source counter in the stack.
    uint64_t lastCounter{ 0 };  ///< Last source counter in the stack.
};

/// A required external property value; type is number, text, or switch.
struct setupRule
{
    std::string device;         ///< Device publishing the required property.
    std::string property;       ///< Property to observe without commanding it.
    std::string element;        ///< Element whose value must match.
    std::string value;          ///< Expected value in canonical text form.
    int         type{ 0 };      ///< 0: number, 1: text, 2: switch (0 or 1).
    double      tolerance{ 0 }; ///< Absolute numeric tolerance; unused for exact text/switch comparisons.
};

/// Pattern and recorded response for one owned additive DM channel.
struct dmCalibration
{
    std::string name;           ///< Physical DM identity.
    std::string channel;        ///< Dedicated additive command channel.
    std::string trigger;        ///< Combined DM stream used to verify writes; empty for a relayed tweeter.
    imageT      pattern;        ///< Full DM-sized command map before amplitude scaling.
    imageT      reference;      ///< Illuminated-baseline-subtracted aligned response.
    float       amplitude{ 0 }; ///< Command-map scaling in DM command units.
};

/// Complete immutable calibration; reference capture defines zero differential.
struct calibration
{
    std::string                  id;                ///< Published timestamped bundle identifier.
    std::string                  profile;           ///< Explicit optical/camera calibration profile.
    std::string                  camera;            ///< Camera INDI device identity.
    std::string                  stream;            ///< Camera shared-memory stream identity.
    geometry                     cameraGeometry;    ///< Common geometry of the two references and baseline.
    std::array<dmCalibration, 2> dm;                ///< NCPC first, tweeter second.
    imageT                       baseline;          ///< Illuminated unpoked reference-capture mean.
    std::vector<setupRule>       setup;             ///< Optical, camera, and loop conditions during capture.
    unsigned                     frames{ 5 };       ///< Distinct frames averaged in each phase.
    unsigned                     settleFrames{ 2 }; ///< Distinct frames discarded after a settled DM update.
    double                       dmSleep{ 0.01 };   ///< Settling seconds after a combined update or relayed write.
    double                       phaseTimeout{ 5 }; ///< Maximum wait per acquisition phase in seconds.
};

} // namespace autoJayDetail
} // namespace app
} // namespace MagAOX

#endif // autoJayTypes_hpp
