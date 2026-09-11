/** \file indiDriver_test.hpp
  * \brief Test harness types for the MagAOX::app::indiDriver Catch2 tests.
  *
  * Provides a stand-in parent that replaces MagAOXApp, and a subclass of indiDriver that
  * exposes its protected state to the tests. This header expects indiDriver.hpp and the
  * pcf INDI headers to be included before it.
  *
  * \ingroup indiDriver_tests
  */

#ifndef app_tests_indiDriver_test_hpp
#define app_tests_indiDriver_test_hpp

namespace indiDriver_tests
{

/// Minimal stand-in for a MagAOXApp-like parent.
/** Satisfies the interface required by MagAOX::app::indiDriver<parentT> without pulling in
 * all of MagAOXApp. It provides only the FIFO name accessors, configName(), the
 * handleXXXProperty callbacks, and a static log<>() template matching the one used in
 * indiDriver.hpp. Each callback increments a counter so tests can check that it was called.
 * The log<>() stub discards the message and returns 0.
 */
struct indiDriverTestParent
{
    std::string m_driverInName;
    std::string m_driverOutName;
    std::string m_driverCtrlName;
    std::string m_configName{ "idtest" };

    std::string driverInName() { return m_driverInName; }
    std::string driverOutName() { return m_driverOutName; }
    std::string driverCtrlName() { return m_driverCtrlName; }
    std::string configName() { return m_configName; }

    int defCount{ 0 };
    int getPropCount{ 0 };
    int newPropCount{ 0 };
    int setPropCount{ 0 };

    void handleDefProperty( const pcf::IndiProperty & ) { ++defCount; }
    void handleGetProperties( const pcf::IndiProperty & ) { ++getPropCount; }
    void handleNewProperty( const pcf::IndiProperty & ) { ++newPropCount; }
    void handleSetProperty( const pcf::IndiProperty & ) { ++setPropCount; }

    template <typename logT>
    static int log( const typename logT::messageT &msg )
    {
        static_cast<void>( msg );
        return 0;
    }
};

/// Test-exposed subclass which reaches the protected members of indiDriver.
/// Tests use it to inspect or force the internal state of the driver. Examples are clearing
/// m_parent, or forcing the outgoing client into a disconnected state. This avoids adding
/// test-only surface to the production class.
/// driverT selects which copy of indiDriver to derive from. The default is the production
/// copy. A fault injection test passes the indiDriver from one of the XWCTEST_ namespaces
/// that indiDriver_test.cpp compiles with a single XWCTEST_INDIDRIVER_ name defined.
template <class parentT, template <class> class driverT = MagAOX::app::indiDriver>
struct indiDriverExposed : public driverT<parentT>
{
    indiDriverExposed( parentT *parent, const std::string &name, const std::string &ver, const std::string &proto )
        : driverT<parentT>( parent, name, ver, proto )
    {
    }

    /// Null the parent pointer so the handler no-op paths can be tested.
    void clearParent()
    {
        this->m_parent = nullptr;
    }

    /// Set the address and port that sendNewProperty will connect to.
    void setServer( const std::string &ip, int port )
    {
        this->m_serverIPAddress = ip;
        this->m_serverPort      = port;
    }

    /// Report whether an outgoing client object currently exists.
    bool hasOutGoing()
    {
        return this->m_outGoing != nullptr;
    }

    /// Tell the outgoing client to quit so the next send must reconnect.
    void forceOutGoingQuit()
    {
        if( this->m_outGoing )
        {
            this->m_outGoing->quitProcess();
        }
    }
};

} // namespace indiDriver_tests

#endif // app_tests_indiDriver_test_hpp
