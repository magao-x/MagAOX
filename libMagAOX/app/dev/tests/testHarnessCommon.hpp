/** \file testHarnessCommon.hpp
 * \brief Shared helper for the dev:: mixin and MagAOXApp Catch2 test harnesses.
 *
 * Several test harnesses under libMagAOX/app/dev/tests/ and libMagAOX/app/tests/
 * each built the same throwaway indiDriver by hand. They did this only to make
 * their m_indiDriver pointer non-null. See makeFifolessIndiDriver() below for why.
 * This header holds that one piece in a single documented definition instead of
 * many copies.
 *
 * Callers must include MagAOXApp.hpp before this header. MagAOXApp.hpp pulls in
 * indiDriver.hpp. This header does not include MagAOXApp.hpp itself because the
 * relative include path differs between libMagAOX/app/dev/tests/ and
 * libMagAOX/app/tests/.
 *
 * \ingroup testing
 */
#ifndef app_dev_tests_testHarnessCommon_hpp
#define app_dev_tests_testHarnessCommon_hpp

namespace MagAOX
{
namespace app
{
namespace dev
{
namespace testHarness
{

/// Construct a real indiDriver with no FIFO, bound to `app` and `configName`.
/** This exists only so a test harness's `m_indiDriver` is non-null. Many INDI
 * update code paths start with an early `if(!m_indiDriver) return 0;` guard.
 * Examples are onPowerOff(), whilePowerOff(), updateINDI(), and the various
 * newCallBack_* and setCallBack_* handlers. With a non-null driver those
 * functions run their full bodies instead of returning at the guard.
 *
 * The test never calls execute() or activate() on the returned driver, so INDI
 * response mode is never enabled. As a result sendSetProperty(),
 * sendDefProperty(), and similar calls are harmless no-ops. No live INDI server
 * is needed.
 *
 * The caller owns the returned pointer and should assign it to m_indiDriver.
 * A harness that constructs more than one of these over its lifetime must
 * delete the previous value first to avoid a leak.
 *
 * \returns a newly allocated indiDriver<AppT>
 */
template <typename AppT>
MagAOX::app::indiDriver<AppT> *makeFifolessIndiDriver( AppT *app /**< [in] the app the driver is bound to */,
                                                        const std::string &configName /**< [in] the device/config name to bind the driver to */
                                                      )
{
    return new MagAOX::app::indiDriver<AppT>( app, configName, "0", "0" );
}

/// Common base for the dev:: mixin test harnesses.
/** A harness derives from this instead of from MagAOXApp<false> directly. It supplies the
 * pieces every harness used to copy by hand:
 *
 * - a constructor that takes the config name,
 * - setupRealDriver(), which installs a FIFO-less indiDriver so the null-driver guards pass,
 * - INDI registration fault injection through m_regFailAt,
 * - public access to the MagAOXApp power management state.
 *
 * The lifecycle functions (setupConfig, loadConfig, appStartup, ...) are not forwarded here.
 * Each mixin harness forwards them itself, because which mixins take part differs per
 * harness and MagAOXApp declares them pure virtual.
 *
 * \ingroup testing
 */
template <bool useINDI = false>
struct appHarnessBaseT : public MagAOX::app::MagAOXApp<useINDI>
{
    typedef MagAOX::app::MagAOXApp<useINDI> appT;

    using appT::m_configName;
    using appT::m_indiDriver;
    using appT::m_powerMgtEnabled;
    using appT::m_powerOnWait;
    using appT::m_powerOnCounter;

    /// Registration fault injection. Every registerIndiProperty* call counts up m_regCallCount.
    /// The call whose count equals m_regFailAt returns -1. The default of -1 never fails.
    int m_regCallCount{ 0 };
    int m_regFailAt{ -1 };

    // The counting overrides below would otherwise hide every other MagAOXApp overload of
    // these names. These using declarations keep the other overloads visible. A derived
    // function with the same signature still takes precedence over the one it brings in.
    using appT::registerIndiPropertyNew;
    using appT::registerIndiPropertyReadOnly;
    using appT::registerIndiPropertySet;

    explicit appHarnessBaseT( const std::string &configName ) : appT( "", false )
    {
        m_configName = configName;
    }

    /// Install a FIFO-less indiDriver so code guarded by a null-driver check runs.
    void setupRealDriver()
    {
        delete m_indiDriver;
        m_indiDriver = makeFifolessIndiDriver<appT>( this, m_configName );
    }

    int registerIndiPropertyNew( pcf::IndiProperty &prop, int ( *cb )( void *, const pcf::IndiProperty & ) )
    {
        if( ++m_regCallCount == m_regFailAt )
        {
            return -1;
        }
        return appT::registerIndiPropertyNew( prop, cb );
    }

    int registerIndiPropertyReadOnly( pcf::IndiProperty &prop )
    {
        if( ++m_regCallCount == m_regFailAt )
        {
            return -1;
        }
        return appT::registerIndiPropertyReadOnly( prop );
    }

    int registerIndiPropertySet( pcf::IndiProperty         &prop,
                                 const std::string         &devName,
                                 const std::string         &propName,
                                 int ( *cb )( void *, const pcf::IndiProperty & ) )
    {
        if( ++m_regCallCount == m_regFailAt )
        {
            return -1;
        }
        return appT::registerIndiPropertySet( prop, devName, propName, cb );
    }
};

/// The common case, a harness without INDI.
using appHarnessBase = appHarnessBaseT<false>;

} // namespace testHarness
} // namespace dev
} // namespace app
} // namespace MagAOX

#endif // app_dev_tests_testHarnessCommon_hpp
