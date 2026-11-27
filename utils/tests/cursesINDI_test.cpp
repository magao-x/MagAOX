/** \file cursesINDI_test.cpp
 * \brief Offline tests for cursesINDI dot-separated search selection.
 * \ingroup cursesINDI_unit_test
 */
#include "../../tests/catch2/catch.hpp"

#include "../cursesINDI/cursesINDI.hpp"

namespace libXWCTest
{
namespace cursesINDITest
{

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

} // namespace cursesINDITest
} // namespace libXWCTest
