/** \file edtCamera_test.hpp
 * \brief Test harness for the MagAOX::app::dev::edtCamera device mixin.
 *
 * One harness class template, edtCameraHarness<traitsT>, drives the real edtCamera code
 * against the fake EDT PDV SDK defined in edtCamera_test.cpp. traitsT is a plain struct
 * that supplies the config name, the c_edtCamera_relativeConfigPath flag, and the default
 * EDT config file that addMode() seeds. Two configurations are used:
 *
 * - edtCameraTestApp uses a relative EDT config path, resolved under the config directory.
 * - edtCameraTestAppAbs uses an absolute EDT config path.
 *
 * The harness supplies the mode map and mode name members that the mixin reads through
 * derived(). Its lifecycle methods forward to the mixin so tests can call them directly.
 * The common parts of every dev:: harness, such as the FIFO-less indiDriver and the
 * registration fault injection, come from appHarnessBase in testHarnessCommon.hpp.
 *
 * The harness sets its config directory to /tmp so relative EDT config paths resolve there.
 *
 * \ingroup testing
 */

#include "../../../../tests/catch2/catch.hpp"

#include <string>

// Make protected members public for this whole translation unit. The tests can then
// read and set MagAOXApp and edtCamera internals directly instead of adding accessor wrappers.
#define protected public
#include "../../MagAOXApp.hpp"
#include "../edtCamera.hpp"
#undef protected

#include "testHarnessCommon.hpp"

// LCOV_EXCL_START

namespace edtCamera_tests
{

/// Traits for the harness with a relative EDT config path.
struct edtCameraRelativeTraits
{
    static constexpr const char *configName         = "edtCameraTestApp";
    static constexpr bool        relativeConfigPath = true;
    static constexpr const char *defaultConfigFile  = "stub.cfg";
};

/// Traits for the harness with an absolute EDT config path.
struct edtCameraAbsoluteTraits
{
    static constexpr const char *configName         = "edtCameraTestAppAbs";
    static constexpr bool        relativeConfigPath = false;
    static constexpr const char *defaultConfigFile  = "/tmp/stub_abs.cfg";
};

/// The edtCamera test harness. traitsT selects the config path style. See the file comment.
template <class traitsT>
struct edtCameraHarness : public MagAOX::app::dev::testHarness::appHarnessBase,
                          public MagAOX::app::dev::edtCamera<edtCameraHarness<traitsT>>
{
    typedef MagAOX::app::dev::testHarness::appHarnessBase           baseT;
    typedef MagAOX::app::dev::edtCamera<edtCameraHarness<traitsT>> edtCameraT;

    friend edtCameraT;

    static constexpr bool c_edtCamera_relativeConfigPath = traitsT::relativeConfigPath;

    // Members that the mixin reads and writes through derived().
    MagAOX::app::dev::cameraConfigMap m_cameraModes;
    std::string                       m_startupMode;
    std::string                       m_modeName;
    std::string                       m_nextMode;

    edtCameraHarness() : baseT( traitsT::configName )
    {
        m_configDir = "/tmp";
    }

    // Disambiguating wrappers. MagAOXApp and edtCamera both declare these. Each one calls
    // the edtCamera version.
    int setupConfig( mx::app::appConfigurator &config )
    {
        edtCameraT::setupConfig( config );
        return 0;
    }

    int loadConfig( mx::app::appConfigurator &config )
    {
        edtCameraT::loadConfig( config );
        return 0;
    }

    int appStartup() override
    {
        return edtCameraT::appStartup();
    }

    int appLogic() override
    {
        return edtCameraT::appLogic();
    }

    int appShutdown() override
    {
        return edtCameraT::appShutdown();
    }

    int onPowerOff() override
    {
        return edtCameraT::onPowerOff();
    }

    int whilePowerOff() override
    {
        return edtCameraT::whilePowerOff();
    }

    /// Seed one minimal camera mode entry. The default config file comes from traitsT.
    void addMode( const std::string &name, const std::string &configFile = traitsT::defaultConfigFile )
    {
        MagAOX::app::dev::cameraConfig cc;
        cc.m_configFile     = configFile;
        m_cameraModes[name] = cc;
    }
};

/// Harness with a relative EDT config path.
using edtCameraTestApp = edtCameraHarness<edtCameraRelativeTraits>;

/// Harness with an absolute EDT config path.
using edtCameraTestAppAbs = edtCameraHarness<edtCameraAbsoluteTraits>;

} // namespace edtCamera_tests

// LCOV_EXCL_STOP
