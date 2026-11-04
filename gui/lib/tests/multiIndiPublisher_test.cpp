/** \file multiIndiPublisher_test.cpp
 * \brief Offline shared-publisher discovery and subscription lifecycle tests.
 * \ingroup gui_publisher_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../tests/testXWC.hpp"

#include "offlineIndiPublisher.hpp"

namespace libXWCTest
{
namespace guiPublisherTest
{
/// \cond TEST_HARNESS
using Publisher = guiTest::OfflinePublisher;
using namespace std::chrono;

/// Capture delivery to an ordinary subscriber without Qt scheduling.
class Subscriber : public multiIndiSubscriber
{
  public:
    int m_definitions{ 0 }; ///< Received definitions.

    int m_updates{ 0 }; ///< Received updates.

    /// Count definitions delivered by the real publisher.
    void handleDefProperty( const pcf::IndiProperty &property /**< [in] Definition. */ ) override;

    /// Count updates delivered by the real publisher.
    void handleSetProperty( const pcf::IndiProperty &property /**< [in] Update. */ ) override;
};
void Subscriber::handleDefProperty( const pcf::IndiProperty & )
{
    ++m_definitions;
}
void Subscriber::handleSetProperty( const pcf::IndiProperty & )
{
    ++m_updates;
}

/// Create a named observation or deletion.
pcf::IndiProperty property( const std::string &device, /**< [in] Device name. */
                            const std::string &name /**< [in] Property name, empty for device deletion. */ );
pcf::IndiProperty property( const std::string &device, const std::string &name )
{
    pcf::IndiProperty result;
    result.setDevice( device );
    result.setName( name );
    return result;
}
/// Exercise QObject subscription and callback queuing without a widget or socket.
class QtSubscriber : public QObject, public Subscriber
{
  public:
    /// Register one property from the queued subscribe callback.
    void subscribe() override;
};
void QtSubscriber::subscribe()
{
    m_parent->addSubscriberProperty( this, "qt-device", "value" );
}
/// \endcond

/** \brief Unresolved properties back off exactly and remain capped without generating commands.
 * \ingroup gui_publisher_unit_test
 */
TEST_CASE( "GUI discovery uses capped exponential backoff", "[multiIndiPublisher]" )
{
    // clang-format off
#ifdef MULTIINDI_TEST_DOXYGEN_REF
    multiIndiPublisher::addSubscriberProperty;
    multiIndiPublisher::update;
    multiIndiPublisher::noteDiscoveryRequest;
#endif
    // clang-format on
    Publisher  publisher;
    Subscriber subscriber;
    publisher.addSubscriberProperty( &subscriber, "late", "value" );
    REQUIRE( publisher.m_queries.size() == 1 );
    for( const int interval : { 1, 2, 4, 8, 16, 32, 60, 60, 60 } )
    {
        auto before = publisher.m_queries.size();
        publisher.tick( seconds( interval - 1 ) );
        REQUIRE( publisher.m_queries.size() == before );
        publisher.tick( seconds( 1 ) );
        REQUIRE( publisher.m_queries.size() == before + 1 );
        REQUIRE( publisher.m_queries.back().getDevice() == "late" );
        REQUIRE( publisher.m_queries.back().getName() == "value" );
        REQUIRE( publisher.m_queries.back().getElements().empty() );
    }
    REQUIRE( publisher.m_commands.empty() );
}

/** \brief Def and Set stop retries immediately, including a producer appearing after the cap.
 * \ingroup gui_publisher_unit_test
 */
TEST_CASE( "Late GUI properties resolve from definitions or updates", "[multiIndiPublisher]" )
{
    // clang-format off
#ifdef MULTIINDI_TEST_DOXYGEN_REF
    multiIndiPublisher::handleDefProperty;
    multiIndiPublisher::handleSetProperty;
    multiIndiPublisher::update;
#endif
    // clang-format on
    for( bool definition : { false, true } )
    {
        Publisher  publisher;
        Subscriber subscriber;
        publisher.addSubscriberProperty( &subscriber, "late", "value" );
        for( int i = 0; i < 10; ++i )
            publisher.tick( seconds( 60 ) );
        auto observation = property( "late", "value" );
        if( definition )
            publisher.handleDefProperty( observation );
        else
            publisher.handleSetProperty( observation );
        const auto received = publisher.m_queries.size();
        publisher.tick( seconds( 600 ) );
        REQUIRE( publisher.m_queries.size() == received );
        REQUIRE( subscriber.m_definitions == ( definition ? 1 : 0 ) );
        REQUIRE( subscriber.m_updates == ( definition ? 0 : 1 ) );
        REQUIRE( publisher.m_commands.empty() );
    }
}

/** \brief Repeated and shared registrations coalesce, while a late listener still requests current data.
 * \ingroup gui_publisher_unit_test
 */
TEST_CASE( "GUI discovery shares records across listeners", "[multiIndiPublisher]" )
{
    // clang-format off
#ifdef MULTIINDI_TEST_DOXYGEN_REF
    multiIndiPublisher::addSubscriberProperty;
    multiIndiPublisher::unsubscribe;
    multiIndiPublisher::pruneDiscovery;
#endif
    // clang-format on
    Publisher  publisher;
    Subscriber first, second, late;
    publisher.addSubscriberProperty( &first, "device", "value" );
    publisher.addSubscriberProperty( &first, "device", "value" );
    publisher.addSubscriberProperty( &second, "device", "value" );
    REQUIRE( publisher.m_queries.size() == 1 );
    REQUIRE( publisher.records() == 1 );
    publisher.handleSetProperty( property( "device", "value" ) );
    REQUIRE( first.m_updates == 1 );
    REQUIRE( second.m_updates == 1 );
    publisher.addSubscriberProperty( &late, "device", "value" );
    REQUIRE( publisher.m_queries.size() == 2 );
    publisher.unsubscribe( &first );
    publisher.unsubscribe( &second, "device", "value" );
    publisher.tick( seconds( 1 ) );
    REQUIRE( publisher.m_queries.size() == 3 );
    publisher.unsubscribe( &late );
    REQUIRE( publisher.records() == 0 );
    publisher.tick( seconds( 100 ) );
    REQUIRE( publisher.m_queries.size() == 3 );
}

/** \brief Deletion restarts only the affected property's or device's discovery schedule.
 * \ingroup gui_publisher_unit_test
 */
TEST_CASE( "GUI discovery rearms scoped deletions", "[multiIndiPublisher]" )
{
    // clang-format off
#ifdef MULTIINDI_TEST_DOXYGEN_REF
    multiIndiPublisher::handleDelProperty;
    multiIndiPublisher::update;
#endif
    // clang-format on
    for( const auto *deleted : { "", "fsm", "position", "unknown" } )
    {
        Publisher  publisher;
        Subscriber subscriber;
        for( const auto *device : { "lost", "healthy" } )
            for( const auto *name : { "fsm", "position" } )
            {
                publisher.addSubscriberProperty( &subscriber, device, name );
                publisher.handleDefProperty( property( device, name ) );
            }
        publisher.m_queries.clear();
        publisher.handleDelProperty( property( "lost", deleted ) );
        publisher.tick( seconds( 0 ) );
        size_t expected = std::string( deleted ) == "unknown" ? 0 : std::string( deleted ) == "position" ? 1 : 2;
        REQUIRE( publisher.m_queries.size() == expected );
        for( const auto &query : publisher.m_queries )
            REQUIRE( query.getDevice() == "lost" );
        publisher.tick( seconds( 1 ) );
        REQUIRE( publisher.m_queries.size() == expected * 2 );
        publisher.handleDelProperty( property( "unrelated", "" ) );
        publisher.tick( seconds( 0 ) );
        REQUIRE( publisher.m_queries.size() == expected * 2 );
    }
}

/** \brief Teardown removes retries and a new connection starts with immediate targeted discovery.
 * \ingroup gui_publisher_unit_test
 */
TEST_CASE( "GUI discovery clears teardown state", "[multiIndiPublisher]" )
{
    // clang-format off
#ifdef MULTIINDI_TEST_DOXYGEN_REF
    multiIndiPublisher::detachAllSubscribers;
    multiIndiPublisher::addSubscriberProperty;
    multiIndiPublisher::update;
#endif
    // clang-format on
    Publisher  publisher;
    Subscriber subscriber;
    publisher.addSubscriberProperty( &subscriber, "device", "value" );
    publisher.detachAllSubscribers();
    REQUIRE( publisher.records() == 0 );
    publisher.tick( seconds( 600 ) );
    REQUIRE( publisher.m_queries.size() == 1 );
    Publisher reconnected;
    reconnected.addSubscriberProperty( &subscriber, "device", "value" );
    REQUIRE( reconnected.m_queries.size() == 1 );
    reconnected.tick( seconds( 1 ) );
    REQUIRE( reconnected.m_queries.size() == 2 );
    REQUIRE( reconnected.m_commands.empty() );
}

/** \brief Queued Qt subscription and delivery remain functional and teardown cancels discovery.
 * \ingroup gui_publisher_unit_test
 */
TEST_CASE( "GUI discovery preserves Qt callback dispatch", "[multiIndiPublisher]" )
{
    // clang-format off
#ifdef MULTIINDI_TEST_DOXYGEN_REF
    multiIndiPublisher::addSubscriber;
    multiIndiPublisher::handleDefProperty;
    multiIndiPublisher::handleSetProperty;
    multiIndiPublisher::unsubscribe;
#endif
    // clang-format on
    Publisher publisher;
    auto     *subscriber = new QtSubscriber;
    publisher.addSubscriber( subscriber );
    REQUIRE( publisher.m_queries.empty() );
    QCoreApplication::processEvents();
    REQUIRE( publisher.m_queries.size() == 1 );
    publisher.handleDefProperty( property( "qt-device", "value" ) );
    publisher.handleSetProperty( property( "qt-device", "value" ) );
    REQUIRE( subscriber->m_definitions == 0 );
    REQUIRE( subscriber->m_updates == 0 );
    QCoreApplication::processEvents();
    REQUIRE( subscriber->m_definitions == 1 );
    REQUIRE( subscriber->m_updates == 1 );
    publisher.tick( seconds( 600 ) );
    REQUIRE( publisher.m_queries.size() == 1 );
    publisher.handleDelProperty( property( "qt-device", "" ) );
    delete subscriber; // QObject cancels queued delivery; subscriber removes retry ownership.
    QCoreApplication::processEvents();
    publisher.tick( seconds( 1 ) );
    REQUIRE( publisher.records() == 0 );
    REQUIRE( publisher.m_queries.size() == 1 );
    REQUIRE( publisher.m_commands.empty() );
}

} // namespace guiPublisherTest
} // namespace libXWCTest

/// Run tests with a Qt event loop and no INDI transport.
int main( int    argc, /**< [in] Argument count. */
          char **argv /**< [in] Arguments consumed by Qt and Catch. */ )
{
    QCoreApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
