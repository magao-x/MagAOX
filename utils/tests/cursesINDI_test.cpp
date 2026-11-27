/** \file cursesINDI_test.cpp
 * \brief Offline tests for cursesINDI search and selection across table updates.
 * \ingroup cursesINDI_unit_test
 */
#include "../../tests/testXWC.hpp"

#include <array>
#include <cctype>
#include <climits>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string_view>
#include <thread>

#include "../../INDI/libcommon/IndiClient.hpp"
#include "../cursesINDI/cursesTableGrid.hpp"

/// \cond CURSESINDI_TEST_HARNESS
namespace pcf
{
/// Replace the network client while retaining the real property and thread-state types.
class OfflineIndiClient : public IndiConnection
{
  public:
    /// Construct without creating sockets or starting any threads.
    OfflineIndiClient( const std::string &name,     /**< [in] client name */
                       const std::string &version,  /**< [in] client version */
                       const std::string &protocol, /**< [in] protocol version */
                       const std::string &address,  /**< [in] unused server address */
                       int                port /**< [in] unused server port */ );

    /// Discard subscription requests from synthetic property definitions.
    void sendGetProperties( const IndiProperty &property /**< [in] unused property */ );

    /// Reject any attempted property edit in these read-only tests.
    void sendNewProperty( const IndiProperty &property /**< [in] unused property */ );

    /// Reject network message dispatch; tests call property callbacks directly.
    void dispatch( const IndiMessage::Type &type, /**< [in] unused message type */
                   const IndiProperty      &property /**< [in] unused property */ ) override;

    /// Reject starting the network processing loop.
    void update() override;
};

OfflineIndiClient::OfflineIndiClient(
    const std::string &name, const std::string &version, const std::string &protocol, const std::string &, int )
    : IndiConnection( name, version, protocol )
{
}

void OfflineIndiClient::sendGetProperties( const IndiProperty & )
{
}

void OfflineIndiClient::sendNewProperty( const IndiProperty & )
{
    FAIL( "Offline selection tests must not edit properties" );
}

void OfflineIndiClient::dispatch( const IndiMessage::Type &, const IndiProperty & )
{
    FAIL( "Offline selection tests must not dispatch network messages" );
}

void OfflineIndiClient::update()
{
    FAIL( "Offline selection tests must not run the network loop" );
}
} // namespace pcf
/// \endcond

// Dependencies are included above so only the app's base and log opens are replaced.
#define IndiClient OfflineIndiClient
#define open( ... ) open( "/dev/null" )
#include "../cursesINDI/cursesINDI.hpp"
#undef open
#undef IndiClient

namespace libXWCTest
{
namespace cursesINDITest
{

/// \cond CURSESINDI_TEST_HARNESS
/// Own a headless curses screen and a client with no network transport or background threads.
struct Screen
{
    FILE                       *m_input{ nullptr };  ///< Local input stream owned by the fixture.
    FILE                       *m_output{ nullptr }; ///< Temporary screen output owned by the fixture.
    SCREEN                     *m_screen{ nullptr }; ///< Curses screen owned by the fixture.
    std::unique_ptr<cursesINDI> m_client;            ///< Client destroyed before the screen.

    /// Initialize the screen and only the windows used by callbacks and navigation.
    Screen();

    /// Release all windows, then the screen and streams.
    ~Screen();
};

Screen::Screen()
{
    m_input  = fopen( "/dev/null", "r" );
    m_output = tmpfile();
    REQUIRE( m_input );
    REQUIRE( m_output );
    m_screen = newterm( "xterm", m_output, m_input );
    REQUIRE( m_screen );
    resizeterm( 30, 120 );

    m_client                 = std::make_unique<cursesINDI>( "offline-selection-test", "1", "1" );
    m_client->w_interactWin  = newwin( 1, 100, 3, 1 );
    m_client->w_curvalWin    = newwin( 1, 100, 4, 1 );
    m_client->w_attentionWin = newwin( 1, 100, 2, 1 );
}

Screen::~Screen()
{
    for( const auto &row : m_client->m_gridWin )
    {
        for( WINDOW *window : row )
        {
            delwin( window );
        }
    }

    if( m_client->m_boxWin )
    {
        delwin( m_client->m_boxWin );
    }

    m_client.reset();
    endwin();
    delscreen( m_screen );
    fclose( m_output );
    fclose( m_input );
}

/// Create a synthetic property with one element.
pcf::IndiProperty property( const std::string &device,               /**< [in] device name */
                            const std::string &name    = "exposure", /**< [in] property name */
                            const std::string &element = "current" /**< [in] element name */ )
{
    pcf::IndiProperty result( pcf::IndiProperty::Text );
    result.setDevice( device );
    result.setName( name );
    result.setPerm( pcf::IndiProperty::ReadOnly );
    result.setState( pcf::IndiProperty::Ok );
    result.add( pcf::IndiElement( element, "fixture" ) );
    return result;
}

/// Read the selected element identity from the actual rendered table coordinates.
std::string selected( const cursesINDI &client /**< [in] client with a rendered table */ )
{
    for( const auto &[key, element] : client.knownElements )
    {
        if( element.tableRow >= 0 && static_cast<size_t>( element.tableRow ) == client.m_startRow + client.m_currY )
        {
            return key;
        }
    }

    return "";
}
/// \endcond

/** \brief Match device names without case sensitivity, allowing skipped letters in order.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Device search accepts case-insensitive abbreviations", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{ { "camWFS.fsm.state", { "camWFS.fsm", "camWFS", "fsm", "state", 4 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "camwfs" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "CAM" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "WFS" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cWfS" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cfw" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cc" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "camwfd" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "camwfss" ) == -1 );
}

/** \brief Prefer a prefix over a substring and a substring over a subsequence.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Device search ranks match categories", "[cursesINDI]" )
{
    cursesINDI::elementMapT elements{ { "acxwyfz.p.e", { "acxwyfz.p", "acxwyfz", "p", "e", 0 } },
                                      { "bcwf.p.e", { "bcwf.p", "bcwf", "p", "e", 1 } },
                                      { "cwfCam.p.e", { "cwfCam.p", "cwfCam", "p", "e", 2 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "cwf" ) == 2 );

    elements.erase( "cwfCam.p.e" );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwf" ) == 1 );

    elements.erase( "bcwf.p.e" );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwf" ) == 0 );
}

/** \brief Break equal-rank ties in table order and select the device's first displayed row.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Device search preserves alphabetical ties and first rows", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{ { "camwfs.a.a", { "camwfs.a", "camwfs", "a", "a", 4 } },
                                            { "camwfs.a.b", { "camwfs.a", "camwfs", "a", "b", 5 } },
                                            { "camwoofer.a.a", { "camwoofer.a", "camwoofer", "a", "a", 7 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "cam" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "amw" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cw" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwo" ) == 7 );
}

/** \brief Leave selection unchanged for empty queries, no matches, or undisplayed devices.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Unqualified search only matches displayed devices", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{ { "camera.wfs.target", { "camera.wfs", "camera", "wfs", "target", 0 } },
                                            { "wfs.p.e", { "wfs.p", "wfs", "p", "e", -1 } } };

    REQUIRE( cursesINDI::findSearchRow( {}, "cam" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "zzzz" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "wfs" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "target" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "camera.wfs" ) == 0 );
}

/** \brief Re-evaluate a shortened query independently of the previous match.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Device search selects again as the query is shortened", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{ { "camvis.p.e", { "camvis.p", "camvis", "p", "e", 0 } },
                                            { "camwfs.p.e", { "camwfs.p", "camwfs", "p", "e", 1 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "camwfsx" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "camwfs" ) == 1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "camw" ) == 1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cam" ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, "" ) == -1 );
}

/** \brief Fuzzy-match each query segment only against its corresponding name field.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Periods keep fuzzy matches within their fields", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{
        { "camWFS.exposureTime.current", { "camWFS.exposureTime", "camWFS", "exposureTime", "current", 4 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "CWFS.XPT.CRR" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp.cur" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp" ) == 4 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cam.wfs" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exptcurrent" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.cur.exp" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp.cur.extra" ) == -1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp.cur." ) == -1 );
}

/** \brief Empty segments leave their fields unrestricted while typing or deleting periods.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Empty search segments preserve field positions", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{
        { "camwfs.exposure.current", { "camwfs.exposure", "camwfs", "exposure", "current", 0 } },
        { "camwfs.exposure.target", { "camwfs.exposure", "camwfs", "exposure", "target", 1 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs." ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp." ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp.tgt" ) == 1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp." ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs.exp" ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cwfs..tgt" ) == 1 );
    REQUIRE( cursesINDI::findSearchRow( elements, ".exp.tgt" ) == 1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "..tgt" ) == 1 );
    REQUIRE( cursesINDI::findSearchRow( elements, "." ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, ".." ) == 0 );
    REQUIRE( cursesINDI::findSearchRow( elements, "..." ) == -1 );
}

/** \brief Rank property and element matches by category before breaking ties in table order.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Each search field uses prefix substring and subsequence ranking", "[cursesINDI]" )
{
    SECTION( "Property ranking" )
    {
        cursesINDI::elementMapT elements{ { "cam.acxwyfz.e", { "cam.acxwyfz", "cam", "acxwyfz", "e", 0 } },
                                          { "cam.bcwf.e", { "cam.bcwf", "cam", "bcwf", "e", 1 } },
                                          { "cam.cwfCam.e", { "cam.cwfCam", "cam", "cwfCam", "e", 2 } } };

        REQUIRE( cursesINDI::findSearchRow( elements, "cam.cwf" ) == 2 );
        elements.erase( "cam.cwfCam.e" );
        REQUIRE( cursesINDI::findSearchRow( elements, "cam.cwf" ) == 1 );
        elements.erase( "cam.bcwf.e" );
        REQUIRE( cursesINDI::findSearchRow( elements, "cam.cwf" ) == 0 );
    }

    SECTION( "Element ranking" )
    {
        cursesINDI::elementMapT elements{ { "cam.p.acxwyfz", { "cam.p", "cam", "p", "acxwyfz", 0 } },
                                          { "cam.p.bcwf", { "cam.p", "cam", "p", "bcwf", 1 } },
                                          { "cam.p.cwfCam", { "cam.p", "cam", "p", "cwfCam", 2 } } };

        REQUIRE( cursesINDI::findSearchRow( elements, "cam.p.cwf" ) == 2 );
        elements.erase( "cam.p.cwfCam" );
        REQUIRE( cursesINDI::findSearchRow( elements, "cam.p.cwf" ) == 1 );
        elements.erase( "cam.p.bcwf" );
        REQUIRE( cursesINDI::findSearchRow( elements, "cam.p.cwf" ) == 0 );
    }
}

/** \brief Earlier fields take ranking priority and equal ranks retain alphabetical order.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Segment ranking proceeds from device to property to element", "[cursesINDI]" )
{
    const cursesINDI::elementMapT elements{ { "acam.cwf.cwf", { "acam.cwf", "acam", "cwf", "cwf", 0 } },
                                            { "cam.acxwyfz.cwf", { "cam.acxwyfz", "cam", "acxwyfz", "cwf", 1 } },
                                            { "cam.bcwf.acxwyfz", { "cam.bcwf", "cam", "bcwf", "acxwyfz", 2 } },
                                            { "cam.bcwf.bcxwyfz", { "cam.bcwf", "cam", "bcwf", "bcxwyfz", 3 } } };

    REQUIRE( cursesINDI::findSearchRow( elements, "cam.cwf.cwf" ) == 2 );
    REQUIRE( cursesINDI::findSearchRow( elements, "cam.cwf.missing" ) == -1 );
}

/** \brief Order devices by field so an exact camera query precedes its helper's prefix match.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Camera sorts before its helper and wins an exact search", "[cursesINDI][sorting]" )
{
    // clang-format off
#ifdef CURSESINDI_TEST_DOXYGEN_REF
    cursesINDI::redrawTable;
    cursesINDI::keyPressed;
#endif
    // clang-format on
    Screen screen;
    auto  &client = *screen.m_client;
    client.handleDefProperty( property( "camacq-acq", "fsm", "state" ) );
    client.handleDefProperty( property( "camacq", "fsm", "state" ) );
    client.redrawTable();
    CHECK( client.knownElements.at( "camacq.fsm.state" ).tableRow == 0 );
    CHECK( client.knownElements.at( "camacq-acq.fsm.state" ).tableRow == 1 );
    client.moveCurrent( 1, 3 );
    client.keyPressed( 0x1f & 'l' );

    for( char ch : std::string( "camacq.fsm.state" ) )
    {
        client.keyPressed( ch );
    }

    CHECK( selected( client ) == "camacq.fsm.state" );
}

/** \brief Compare property and element fields within each device and retain selection on updates.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Table orders each field independently", "[cursesINDI][sorting]" )
{
    // clang-format off
#ifdef CURSESINDI_TEST_DOXYGEN_REF
    cursesINDI::redrawTable;
    cursesINDI::handleDelProperty;
#endif
    // clang-format on
    Screen screen;
    auto  &client = *screen.m_client;
    client.handleDefProperty( property( "camacq-acq", "fsm", "state" ) );
    client.redrawTable();
    client.handleDefProperty( property( "camacq", "fsm-acq", "state" ) );
    auto fsm = property( "camacq", "fsm", "state-acq" );
    fsm.add( pcf::IndiElement( "state", "fixture" ) );
    client.handleDefProperty( fsm );
    client.redrawTable();
    REQUIRE( client.knownElements.at( "camacq.fsm.state" ).tableRow == 0 );
    REQUIRE( client.knownElements.at( "camacq.fsm.state-acq" ).tableRow == 1 );
    REQUIRE( client.knownElements.at( "camacq.fsm-acq.state" ).tableRow == 2 );
    REQUIRE( client.knownElements.at( "camacq-acq.fsm.state" ).tableRow == 3 );
    REQUIRE( selected( client ) == "camacq-acq.fsm.state" );

    client.moveCurrent( -static_cast<int>( client.m_startRow ), 3 );
    REQUIRE( selected( client ) == "camacq.fsm.state" );
    client.handleDelProperty( fsm );
    client.redrawTable();
    REQUIRE( selected( client ) == "camacq.fsm-acq.state" );
}

/** \brief Preserve the selected element and column when rows are inserted or removed before it.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Table updates preserve selection identity", "[cursesINDI][selection]" )
{
    // clang-format off
#ifdef CURSESINDI_TEST_DOXYGEN_REF
    cursesINDI::handleDefProperty;
    cursesINDI::handleDelProperty;
    cursesINDI::redrawTable;
    cursesINDI::moveCurrent;
#endif
    // clang-format on
    Screen screen;
    auto  &client = *screen.m_client;
    client.handleDefProperty( property( "camwfs" ) );
    client.handleDefProperty( property( "zzcamera" ) );
    client.redrawTable();
    client.moveCurrent( 0, 3 );

    client.handleDefProperty( property( "camaux" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == "camwfs.exposure.current" );
    REQUIRE( client.m_currY == 0 );
    REQUIRE( client.m_currX == 3 );
    REQUIRE( client.m_startRow == 1 );

    client.handleDefProperty( property( "camwfs", "aaa" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == "camwfs.exposure.current" );
    REQUIRE( client.m_startRow == 2 );

    client.handleDelProperty( property( "camaux" ) );
    client.handleDelProperty( property( "camwfs", "aaa" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == "camwfs.exposure.current" );
    REQUIRE( client.m_currY == 0 );
    REQUIRE( client.m_currX == 3 );
    REQUIRE( client.m_startRow == 0 );
}

/** \brief Preserve a scrolled selection and explicit search focus through a redraw.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Scrolled selection survives updates and search focus", "[cursesINDI][selection]" )
{
    // clang-format off
#ifdef CURSESINDI_TEST_DOXYGEN_REF
    cursesINDI::redrawTable;
    cursesINDI::moveCurrent;
    cursesINDI::keyPressed;
    cursesINDI::deviceSearch;
#endif
    // clang-format on
    Screen screen;
    auto  &client = *screen.m_client;

    for( int i = 0; i < 40; ++i )
    {
        client.handleDefProperty( property( std::format( "cam{:02}", i ) ) );
    }

    client.redrawTable();
    client.moveCurrent( 30, 3 );
    const auto key   = selected( client );
    const auto row   = client.m_currY;
    const auto start = client.m_startRow;
    REQUIRE( start > 0 );
    client.keyPressed( 0x1f & 'l' );
    client.handleDefProperty( property( "aaa" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == key );
    REQUIRE( client.m_currY == row );
    REQUIRE( client.m_currX == 3 );
    REQUIRE( client.m_startRow == start + 1 );
    REQUIRE( client.m_deviceSearching );
    REQUIRE( client.m_deviceTarget.empty() );

    client.handleDelProperty( property( "aaa" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == key );
    REQUIRE( client.m_startRow == start );

    for( char ch : std::string( "cam05" ) )
    {
        client.keyPressed( ch );
    }

    REQUIRE( selected( client ) == "cam05.exposure.current" );
    client.handleDefProperty( property( "aaa" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == "cam05.exposure.current" );
    REQUIRE( client.m_deviceTarget == "cam05" );
}

/** \brief Fall forward, then backward, when the selection disappears, and handle an empty table.
 * \ingroup cursesINDI_unit_test
 */
TEST_CASE( "Selection falls back when selected rows disappear", "[cursesINDI][selection]" )
{
    // clang-format off
#ifdef CURSESINDI_TEST_DOXYGEN_REF
    cursesINDI::handleDelProperty;
    cursesINDI::redrawTable;
    cursesINDI::updateCurVal;
#endif
    // clang-format on
    Screen screen;
    auto  &client = *screen.m_client;
    client.handleDefProperty( property( "aaa" ) );
    client.handleDefProperty( property( "camwfs" ) );
    client.handleDefProperty( property( "zzz" ) );
    client.redrawTable();
    client.moveCurrent( 1, 3 );

    pcf::IndiProperty deleted;
    deleted.setDevice( "camwfs" );
    client.handleDelProperty( deleted );
    client.redrawTable();
    REQUIRE( selected( client ) == "zzz.exposure.current" );
    REQUIRE( client.m_currX == 3 );

    client.handleDelProperty( property( "zzz" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == "aaa.exposure.current" );
    REQUIRE( client.m_startRow == 0 );
    REQUIRE( client.m_currY == 0 );

    client.handleDelProperty( property( "aaa" ) );
    client.redrawTable();
    REQUIRE( selected( client ).empty() );
    REQUIRE( client.m_startRow == 0 );
    REQUIRE( client.m_currY == 0 );
    char value[100]{};
    mvwinnstr( client.w_curvalWin, 0, 0, value, sizeof( value ) - 1 );
    REQUIRE( std::string( value ).find_first_not_of( ' ' ) == std::string::npos );

    client.handleDefProperty( property( "camwfs" ) );
    client.redrawTable();
    REQUIRE( selected( client ) == "camwfs.exposure.current" );
}

} // namespace cursesINDITest
} // namespace libXWCTest
