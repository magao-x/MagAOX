/** \file value_display_test.cpp
 * \brief Offline value display formatting, availability, layout, and plugin lifecycle tests.
 * \ingroup valueDisplay_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include <QFileInfo>
#include <QPixmap>
#include <QPluginLoader>
#include <QPointer>
#include <QTemporaryFile>

#include <sstream>

#include "../indiDictionary/indiDictionary.hpp"
#include "../valueDisplay/valueDisplay.hpp"

namespace libXWCTest
{
namespace valueDisplayTest
{
/// \cond TEST_HARNESS

/// INDI parser with a simulated connection and no client, timer, socket, or outgoing command path.
class Provider : public indiDictionary
{
  protected:
    /// Availability reported to the real parser, independent of any network connection.
    bool m_connected{ true };

    /// Return the simulated connection state.
    bool commandConnected() const override;

  public:
    /// Bind the parser directly to local storage without starting the dictionary plugin.
    explicit Provider( dictionaryT &dictionary /**< [in,out] In-memory dictionary that outlives this provider. */ );

    /// Deliver a definition, partial update, or deletion through the real parser.
    void receive( const pcf::IndiProperty &property,          /**< [in] Complete simulated INDI message. */
                  bool                     definition = true, /**< [in] Whether this is a replacement definition. */
                  bool                     removed    = false /**< [in] Whether this deletes a property or device. */
    );

    /// Simulate loss of the transport and use the real invalidation path.
    void disconnectTransport();

    /// Allow subsequent definitions after the simulated disconnection.
    void reconnectTransport();
};

Provider::Provider( dictionaryT &dictionary )
{
    m_dict = &dictionary;
}

bool Provider::commandConnected() const
{
    return m_connected;
}

void Provider::receive( const pcf::IndiProperty &property, bool definition, bool removed )
{
    receiveProperty( property, m_generation, definition, removed );
}

void Provider::disconnectTransport()
{
    m_connected = false;
    invalidateConnection();
}

void Provider::reconnectTransport()
{
    m_connected = true;
}

/// Capture ordinary plugin diagnostics while restoring the stream even if a test fails.
class Diagnostics
{
  protected:
    /// Accumulated output from the tested operation.
    std::ostringstream m_output;

    /// Non-owning original stream buffer to restore on destruction.
    std::streambuf *m_previous{ nullptr };

  public:
    /// Begin capturing stderr stream diagnostics.
    Diagnostics();

    /// Restore the original stream buffer.
    ~Diagnostics();

    /// Return the captured diagnostics.
    std::string text() const;
};

Diagnostics::Diagnostics()
{
    m_previous = std::cerr.rdbuf( m_output.rdbuf() );
}

Diagnostics::~Diagnostics()
{
    std::cerr.rdbuf( m_previous );
}

std::string Diagnostics::text() const
{
    return m_output.str();
}

/// Configure an overlay from a temporary file using the real unused-key parser.
int attach( rtimvOverlayInterface &overlay, /**< [in,out] Overlay to configure. */
            rtimvGraphicsView     &view,    /**< [in,out] Offline graphics view. */
            dictionaryT      *dictionary,   /**< [in,out] Local dictionary, or null to simulate a missing provider. */
            const QByteArray &settings      /**< [in] Entries in the valueDisplay configuration section. */
)
{
    QTemporaryFile file;
    REQUIRE( file.open() );
    const QByteArray contents = "[valueDisplay]\n" + settings;
    REQUIRE( file.write( contents ) == contents.size() );
    REQUIRE( file.flush() );
    mx::app::appConfigurator config;
    REQUIRE( config.readConfig( file.fileName().toStdString() ) == 0 );

    rtimvOverlayAccess access;
    access.m_graphicsView = &view;
    access.m_dictionary   = dictionary;
    return overlay.attachOverlay( access, config );
}

/// Numeric message built without a server or serialized INDI transport.
pcf::IndiProperty numbers( const std::string                   &device,   /**< [in] Source device. */
                           const std::string                   &property, /**< [in] Source property. */
                           const std::map<std::string, double> &values    /**< [in] Named elements in this message. */
)
{
    pcf::IndiProperty result( pcf::IndiProperty::Number, device, property );
    result.setPerm( pcf::IndiProperty::ReadOnly );
    result.setState( pcf::IndiProperty::Ok );

    for( const auto &[name, value] : values )
    {
        result.add( pcf::IndiElement( name ) );
        result[name] = value;
    }

    return result;
}

/// View and provider fixture; destruction order keeps every borrowed object alive through plugin teardown.
struct Fixture
{
    /// Storage shared by the provider and overlay.
    dictionaryT m_dictionary;

    /// Local scene used only for screen-coordinate layout checks.
    QGraphicsScene m_scene;

    /// Real view containing native overlay widgets.
    rtimvGraphicsView m_view;

    /// Fake connection invoking the production INDI parser.
    Provider m_provider{ m_dictionary };

    /// Real overlay using the offline view and dictionary.
    valueDisplay m_overlay;

    /// Show a representative view without any image or instrument connection.
    Fixture();

    /// Configure the real overlay and optionally withhold the dictionary.
    int configure( const QByteArray &settings,             /**< [in] Value display settings. */
                   bool              withDictionary = true /**< [in] Whether to expose the local dictionary. */
    );

    /// Return the plugin's private widget, failing if attachment did not create it.
    QTextEdit *widget();

    /// Refresh the overlay and return its visible text.
    QString text();

    /// Store an arbitrary terminated payload to exercise the numeric reader.
    void value( const std::string &key, /**< [in] Dictionary key. */
                const std::string &text /**< [in] Raw value, possibly containing embedded nulls. */
    );
};

Fixture::Fixture()
{
    m_view.setScene( &m_scene );
    m_view.setSceneRect( 0, 0, 512, 512 );
    m_view.setBackgroundBrush( QColor( 38, 42, 48 ) );
    m_view.resize( 640, 480 );
    m_view.show();
    QCoreApplication::processEvents();
}

int Fixture::configure( const QByteArray &settings, bool withDictionary )
{
    return attach( m_overlay, m_view, withDictionary ? &m_dictionary : nullptr, settings );
}

QTextEdit *Fixture::widget()
{
    auto *result = m_view.findChild<QTextEdit *>( "valueDisplayText" );
    REQUIRE( result != nullptr );
    return result;
}

QString Fixture::text()
{
    REQUIRE( m_overlay.updateOverlay() == 0 );
    return widget()->toPlainText();
}

void Fixture::value( const std::string &key, const std::string &text )
{
    m_dictionary[key].setBlob( text.c_str(), text.size() + 1 );
}

/// \endcond

/** \brief Configuration supports independent slots, default labels, and literal custom labels.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display configures only requested rows", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::attachOverlay; valueDisplay::updateOverlay; valueDisplay::overlayEnabled;
#endif
    // clang-format on
    Fixture f;

    SECTION( "no keys leaves the plugin inactive" )
    {
        CHECK( f.configure( "" ) > 0 );
        CHECK_FALSE( f.m_overlay.overlayEnabled() );
        CHECK( f.m_dictionary.empty() );
        CHECK( f.m_view.findChild<QTextEdit *>( "valueDisplayText" ) == nullptr );
    }

    SECTION( "either slot can be configured independently" )
    {
        for( const char *key : { "key1", "key2" } )
        {
            Fixture single;
            REQUIRE( single.configure( QByteArray( key ) + "=fit.values.x\n" ) == 0 );
            CHECK( single.m_overlay.overlayEnabled() );
            CHECK( single.m_dictionary.size() == 1 );
            CHECK( single.text() == "x: " );
        }
    }

    SECTION( "different properties and devices have independent labels" )
    {
        REQUIRE( f.configure( "key1=camflowfs-fit.values.x_rms\nlabel1=<b>X rms</b>\nprecision1=2\n"
                              "key2=camwfs-fit.average.dy\n" ) == 0 );
        f.value( "camflowfs-fit.values.x_rms", "0.05" );
        f.value( "camwfs-fit.average.dy", "-1.89847564697266" );
        CHECK( f.text() == "<b>X rms</b>: 0.05\ndy: -1.898" );
        CHECK( f.m_dictionary.size() == 2 );
    }

    SECTION( "shared keys retain cached values and can have two labels" )
    {
        f.value( "fit.values.x", "7.25" );
        REQUIRE( f.configure( "key1=fit.values.x\nkey2=fit.values.x\nlabel2=duplicate\n" ) == 0 );
        CHECK( f.text() == "x: 7.250\nduplicate: 7.250" );
        CHECK( f.m_dictionary.size() == 1 );
    }

    SECTION( "missing provider keeps configured labels blank" )
    {
        REQUIRE( f.configure( "key1=fit.values.x\nkey2=fit.values.y\n", false ) == 0 );
        CHECK( f.text() == "x: \ny: " );
    }
}

/** \brief Invalid configuration fails before registering any dictionary keys.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display rejects malformed keys and precision", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::attachOverlay;
#endif
    // clang-format on
    for( const char *key : { "device",
                             "device.property",
                             ".property.element",
                             "device..element",
                             "device.property.",
                             "device.property.element.extra" } )
    {
        CAPTURE( key );
        Fixture     f;
        Diagnostics diagnostics;
        CHECK( f.configure( QByteArray( "key1=valid.property.element\nkey2=" ) + key + "\n" ) < 0 );
        CHECK_FALSE( f.m_overlay.overlayEnabled() );
        CHECK( f.m_dictionary.empty() );
    }

    for( const char *precision : { "-1", "16", "abc", "1.5", "3junk" } )
    {
        CAPTURE( precision );
        Fixture     f;
        Diagnostics diagnostics;
        CHECK( f.configure( QByteArray( "key1=fit.values.x\nprecision1=" ) + precision + "\n" ) < 0 );
        CHECK( f.m_dictionary.empty() );
    }
}

/** \brief Numeric formatting preserves signs and fixed precision without switching to exponential notation.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display formats finite numbers at fixed precision", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::attachOverlay; valueDisplay::updateOverlay;
#endif
    // clang-format on
    Fixture f;
    REQUIRE( f.configure( "key1=fit.average.dx\nkey2=fit.average.dy\n" ) == 0 );
    f.value( "fit.average.dx", "-0.482002258300781" );
    f.value( "fit.average.dy", "-1.89847564697266" );
    CHECK( f.text() == "dx: -0.482\ndy: -1.898" );
    CHECK( f.widget()->currentFont().pixelSize() == static_cast<int>( f.m_view.statusTextFontSize() ) );
    CHECK( f.widget()->textColor() == QColor( f.m_view.statusTextFontColor() ) );
    CHECK( f.widget()->textBackgroundColor() == QColor( 0, 0, 0, 150 ) );

    const std::vector<std::pair<std::string, QString>> cases{ { "0", "0.000" },
                                                              { "1.2344", "1.234" },
                                                              { "1.2346", "1.235" },
                                                              { "-1.2346", "-1.235" },
                                                              { "1.25e2", "125.000" },
                                                              { "1e-8", "0.000" },
                                                              { "10000000", "10000000.000" } };

    for( const auto &[input, expected] : cases )
    {
        CAPTURE( input );
        f.value( "fit.average.dx", input );
        CHECK( f.text().section( '\n', 0, 0 ) == "dx: " + expected );
    }

    f.value( "fit.average.dx", "1e308" );
    const QString large = f.text().section( '\n', 0, 0 );
    CHECK( large.size() > 300 );
    CHECK_FALSE( large.contains( 'e', Qt::CaseInsensitive ) );
    CHECK( large.endsWith( ".000" ) );
    CHECK( f.widget()->geometry().right() < f.m_view.viewport()->width() );
    CHECK( f.widget()->currentFont().pixelSize() == static_cast<int>( f.m_view.statusTextFontSize() ) );
    CHECK( f.widget()->textColor() == QColor( f.m_view.statusTextFontColor() ) );
    CHECK( f.widget()->textBackgroundColor() == QColor( 0, 0, 0, 150 ) );

    Fixture rms;
    REQUIRE( rms.configure( "key1=fit.values.x_rms\nprecision1=2\nkey2=fit.values.y_rms\nprecision2=2\n" ) == 0 );
    rms.value( "fit.values.x_rms", "0.05" );
    rms.value( "fit.values.y_rms", "0.01" );
    CHECK( rms.text() == "x_rms: 0.05\ny_rms: 0.01" );

    Fixture extremes;
    REQUIRE( extremes.configure( "key1=fit.values.x\nprecision1=0\nkey2=fit.values.y\nprecision2=15\n" ) == 0 );
    extremes.value( "fit.values.x", "2.6" );
    extremes.value( "fit.values.y", "0.125" );
    CHECK( extremes.text() == "x: 3\ny: 0.125000000000000" );
}

/** \brief Missing or invalid payloads produce blank value portions without warnings or apparent numeric zero.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display blanks unavailable numeric data quietly", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::updateOverlay;
#endif
    // clang-format on
    Fixture f;
    REQUIRE( f.configure( "key1=fit.values.x\n" ) == 0 );
    Diagnostics diagnostics;
    CHECK( f.text() == "x: " );

    for( const std::string &invalid : std::vector<std::string>{ "",
                                                                "junk",
                                                                "12junk",
                                                                "NaN",
                                                                "nan",
                                                                "inf",
                                                                "-inf",
                                                                "1e999",
                                                                std::string( "1\0junk", 6 ),
                                                                "1." + std::string( 509, '0' ),
                                                                "1." + std::string( 8192, '0' ) } )
    {
        CAPTURE( invalid.size() );
        f.value( "fit.values.x", invalid );
        CHECK( f.text() == "x: " );
    }

    f.m_dictionary["fit.values.x"].setBlob( "1.25", 4 );
    CHECK( f.text() == "x: " );
    f.value( "fit.values.x", "5" );
    CHECK( f.text() == "x: 5.000" );
    f.m_dictionary["fit.values.x"].setBlob( nullptr, 0 );
    CHECK( f.text() == "x: " );
    f.m_dictionary.erase( "fit.values.x" );
    CHECK( f.text() == "x: " );
    CHECK( f.m_dictionary.empty() );
    CHECK( diagnostics.text().empty() );
}

/** \brief Old cached values remain visible until explicit provider invalidation, including across partial updates.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display follows publication and invalidation without age expiry", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::attachOverlay; valueDisplay::updateOverlay;
    indiDictionary::receiveProperty; indiDictionary::invalidateConnection;
#endif
    // clang-format on
    Fixture       f;
    rtimvDictBlob source;
    source.setBlob( "2.5", 4 );
    f.m_dictionary.try_emplace( "fit.values.x", source );
    char     copied[16];
    timespec timestamp{};
    REQUIRE( f.m_dictionary["fit.values.x"].getBlob( copied, sizeof( copied ), &timestamp ) == 4 );
    REQUIRE( timestamp.tv_sec == 0 );
    REQUIRE( f.configure( "key1=fit.values.x\nkey2=fit.values.y\n" ) == 0 );

    for( int n = 0; n < 8; ++n )
    {
        CHECK( f.text() == "x: 2.500\ny: " );
    }

    f.m_provider.receive( numbers( "fit", "values", { { "x", 1 }, { "y", 2 } } ) );
    f.m_provider.receive( numbers( "fit", "values", { { "x", 3 } } ), false );
    CHECK( f.text() == "x: 3.000\ny: 2.000" );

    SECTION( "property deletion blanks both slots and permits rediscovery" )
    {
        f.m_provider.receive( numbers( "fit", "values", {} ), false, true );
    }

    SECTION( "device deletion blanks both slots and permits rediscovery" )
    {
        f.m_provider.receive( numbers( "fit", "", {} ), false, true );
    }

    SECTION( "disconnect blanks both slots and permits a new connection generation" )
    {
        f.m_provider.disconnectTransport();
        f.m_provider.reconnectTransport();
    }

    CHECK( f.text() == "x: \ny: " );
    CHECK( f.m_dictionary.count( "fit.values.x" ) == 1 );
    CHECK( f.m_dictionary.count( "fit.values.y" ) == 1 );
    f.m_provider.receive( numbers( "fit", "values", { { "x", 4 }, { "y", 5 } } ) );
    CHECK( f.text() == "x: 4.000\ny: 5.000" );
}

/** \brief Replacement definitions invalidate omitted elements while partial updates and unrelated properties persist.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "INDI replacement definitions clear removed value blobs", "[valueDisplay][indiDictionary]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    indiDictionary::receiveProperty; valueDisplay::updateOverlay;
#endif
    // clang-format on
    Fixture f;
    REQUIRE( f.configure( "key1=fit.values.x\nkey2=fit.values.y\n" ) == 0 );
    f.m_provider.receive( numbers( "fit", "values", { { "x", 1 }, { "y", 2 } } ) );
    f.m_provider.receive( numbers( "other", "values", { { "y", 7 } } ) );
    f.m_provider.receive( numbers( "fit", "values_extra", { { "y", 8 } } ) );
    f.m_provider.receive( numbers( "fit", "values", { { "x", 3 } } ), false );
    CHECK( f.text() == "x: 3.000\ny: 2.000" );
    f.m_provider.receive( numbers( "fit", "values", { { "x", 4 } } ) );
    CHECK( f.text() == "x: 4.000\ny: " );
    REQUIRE( f.m_dictionary.count( "fit.values.y" ) == 1 );
    CHECK( f.m_dictionary["fit.values.y"].getBlobSize() == 0 );
    CHECK( f.m_dictionary["other.values.y"].getBlobSize() > 0 );
    CHECK( f.m_dictionary["fit.values_extra.y"].getBlobSize() > 0 );
    CHECK_FALSE( f.m_provider.propertySnapshot( "fit", "values" ).m_elements.contains( "y" ) );

    f.m_provider.receive( numbers( "fit", "values", { { "x", 5 }, { "y", 6 } } ) );
    CHECK( f.text() == "x: 5.000\ny: 6.000" );
    f.m_provider.receive( numbers( "tcsi", "teldata", { { "pa", 12 }, { "el", 45 } } ) );
    REQUIRE( f.m_dictionary["rtimv.north.angle"].getBlobSize() == sizeof( float ) );
    f.m_provider.receive( numbers( "tcsi", "teldata", { { "el", 46 } } ), false );
    CHECK( f.m_dictionary["rtimv.north.angle"].getBlobSize() == sizeof( float ) );
    f.m_provider.receive( numbers( "tcsi", "teldata", { { "el", 47 } } ) );
    CHECK( f.m_dictionary["tcsi.teldata.pa"].getBlobSize() == 0 );
    CHECK( f.m_dictionary["rtimv.north.angle"].getBlobSize() == 0 );
}

/** \brief Screen-anchored values avoid native statistics and heading space, clip safely, and remain below help.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display follows native statistics geometry", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::updateOverlay; valueDisplay::eventFilter;
#endif
    // clang-format on
    Fixture f;
    REQUIRE( f.configure( "key1=fit.values.x_rms\nkey2=fit.values.y_rms\n" ) == 0 );
    f.value( "fit.values.x_rms", "0.05" );
    f.value( "fit.values.y_rms", "0.01" );
    REQUIRE( f.text() == "x_rms: 0.050\ny_rms: 0.010" );
    QTextEdit  *values  = f.widget();
    QTextEdit  *stats   = f.m_view.globalStatsText();
    const QRect initial = values->geometry();
    CHECK( values->isVisible() );
    CHECK( initial.top() > f.m_view.saveBox()->geometry().bottom() );
    CHECK( values->testAttribute( Qt::WA_TransparentForMouseEvents ) );

    stats->setPlainText( "\nmin: 0.00\nmax: 1.00\nmean: 0.50\nmedian: 0.50" );
    stats->setGeometry( 6, 6, 180, 130 );
    stats->show();
    QCoreApplication::processEvents();
    CHECK_FALSE( values->geometry().intersects( stats->geometry() ) );
    CHECK( values->y() > stats->geometry().bottom() );
    CHECK( values->y() > initial.y() );

    stats->move( 6, 40 );
    stats->resize( 200, 150 );
    QCoreApplication::processEvents();
    CHECK( values->y() > stats->geometry().bottom() );
    CHECK_FALSE( values->geometry().intersects( stats->geometry() ) );
    const QRect beforeZoom = values->geometry();
    f.m_view.scale( 2, 2 );
    f.m_view.centerOn( 128, 128 );
    f.m_overlay.updateOverlay();
    CHECK( values->geometry() == beforeZoom );

    const QString renderPath = qEnvironmentVariable( "VALUE_DISPLAY_RENDER_PATH" );

    if( !renderPath.isEmpty() )
    {
        REQUIRE( f.m_view.grab().save( renderPath ) );
    }

    stats->hide();
    QCoreApplication::processEvents();
    CHECK( values->geometry() == initial );
    stats->show();
    QCoreApplication::processEvents();
    CHECK( values->y() > stats->geometry().bottom() );
    f.m_view.helpText()->setPlainText( "Offline help" );
    f.m_view.helpText()->show();
    f.m_overlay.updateOverlay();
    const QObjectList children = f.m_view.children();
    CHECK( children.indexOf( values ) < children.indexOf( f.m_view.helpText() ) );
    f.m_view.helpText()->hide();

    f.m_view.resize( 100, 80 );
    QCoreApplication::processEvents();
    f.m_overlay.updateOverlay();

    if( values->isVisible() )
    {
        CHECK_FALSE( values->geometry().intersects( stats->geometry() ) );
        CHECK( values->geometry().right() < f.m_view.viewport()->width() );
        CHECK( values->geometry().bottom() < f.m_view.viewport()->height() );
    }

    stats->hide();
    QCoreApplication::processEvents();
    f.m_overlay.updateOverlay();
    CHECK( values->geometry().width() <= f.m_view.viewport()->width() );
    f.m_view.resize( 640, 480 );
    QCoreApplication::processEvents();
    CHECK( values->isVisible() );
    CHECK( values->geometry() == initial );
}

/** \brief Disabling and destroying the plugin preserves foreign text and handles either destruction order.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display owns only its own text widget", "[valueDisplay]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::enableOverlay; valueDisplay::disableOverlay; valueDisplay::~valueDisplay;
#endif
    // clang-format on
    Fixture f;
    f.m_view.statusTextText( 0, "camera remains" );
    f.m_view.globalStatsText()->setPlainText( "native statistics" );
    f.m_view.globalStatsText()->setGeometry( 6, 6, 180, 100 );
    f.m_view.globalStatsText()->show();
    REQUIRE( f.configure( "key1=fit.values.x\n" ) == 0 );
    f.value( "fit.values.x", "2" );
    CHECK( f.text() == "x: 2.000" );
    f.m_overlay.disableOverlay();
    CHECK_FALSE( f.m_overlay.overlayEnabled() );
    CHECK_FALSE( f.widget()->isVisible() );
    CHECK( f.m_view.statusText( 0 )->toPlainText() == "camera remains" );
    CHECK( f.m_view.globalStatsText()->toPlainText() == "native statistics" );
    f.m_overlay.enableOverlay();
    CHECK( f.m_overlay.overlayEnabled() );
    CHECK( f.text() == "x: 2.000" );
    CHECK( f.widget()->isVisible() );

    QGraphicsScene    scene;
    rtimvGraphicsView view;
    view.setScene( &scene );
    view.resize( 640, 480 );
    view.show();
    view.statusTextText( 0, "unrelated" );
    QPointer<QTextEdit> owned;

    {
        valueDisplay overlay;
        REQUIRE( attach( overlay, view, &f.m_dictionary, "key1=fit.values.x\n" ) == 0 );
        owned = view.findChild<QTextEdit *>( "valueDisplayText" );
        REQUIRE( !owned.isNull() );
    }

    CHECK( owned.isNull() );
    CHECK( view.statusText( 0 )->toPlainText() == "unrelated" );

    {
        valueDisplay overlay;

        {
            rtimvGraphicsView childFirst;
            childFirst.setScene( &scene );
            childFirst.resize( 640, 480 );
            childFirst.show();
            REQUIRE( attach( overlay, childFirst, &f.m_dictionary, "key1=fit.values.x\n" ) == 0 );
            QCoreApplication::processEvents();
        }

        CHECK( overlay.updateOverlay() == 0 );
        overlay.disableOverlay();
        overlay.enableOverlay();
        CHECK_FALSE( overlay.overlayEnabled() );
    }
}

/** \brief The built shared object exposes the overlay interface and unloads without a connection.
 * \ingroup valueDisplay_unit_test
 */
TEST_CASE( "Value display shared plugin loads and unloads offline", "[valueDisplay][pluginLoad]" )
{
    // clang-format off
#ifdef VALUE_DISPLAY_TEST_DOXYGEN_REF
    valueDisplay::attachOverlay; valueDisplay::updateOverlay; valueDisplay::~valueDisplay;
#endif
    // clang-format on
    const QString pluginPath = qEnvironmentVariable( "VALUE_DISPLAY_PLUGIN" );

    if( pluginPath.isEmpty() )
    {
        SUCCEED( "Set VALUE_DISPLAY_PLUGIN to the built shared object to exercise dynamic plugin loading." );
        return;
    }

    REQUIRE( QFileInfo::exists( pluginPath ) );
    Fixture       f;
    QPluginLoader loader( pluginPath );
    QObject      *instance = loader.instance();
    INFO( loader.errorString().toStdString() );
    REQUIRE( instance != nullptr );
    auto *overlay = qobject_cast<rtimvOverlayInterface *>( instance );
    REQUIRE( overlay != nullptr );
    REQUIRE( attach( *overlay, f.m_view, &f.m_dictionary, "key1=fit.values.x\n" ) == 0 );
    f.value( "fit.values.x", "1.5" );
    REQUIRE( overlay->updateOverlay() == 0 );
    QPointer<QTextEdit> text = f.m_view.findChild<QTextEdit *>( "valueDisplayText" );
    REQUIRE( !text.isNull() );
    CHECK( text->toPlainText() == "x: 1.500" );
    REQUIRE( loader.unload() );
    CHECK( text.isNull() );
}

} // namespace valueDisplayTest
} // namespace libXWCTest

/// Run local Qt/Catch tests without starting any viewer or INDI transport.
int main( int    argc, /**< [in] Argument count. */
          char **argv  /**< [in] Catch options. */
)
{
    QApplication app( argc, argv );
    return Catch::Session().run( argc, argv );
}
