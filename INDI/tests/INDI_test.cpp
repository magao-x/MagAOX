/** \file INDI_test.cpp
 * \brief Catch2 tests for the INDI library, including IndiXmlParser XML escaping.
 * \ingroup test_indi
 */
#include "../../tests/catch2/catch.hpp"

#include <arpa/inet.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <thread>
#include <type_traits>
#include <vector>

#include "../INDI/connect_to.h"
#include "../INDI/fq.h"
#include "../INDI/indiapi.h"
#include "../libcommon/Cmd.hpp"
#include "../libcommon/IndiClient.hpp"
#include "../libcommon/IndiConnection.hpp"
#include "../libcommon/IndiDriver.hpp"
#include "../libcommon/IndiElement.hpp"
#include "../libcommon/IndiMessage.hpp"
#include "../libcommon/IndiProperty.hpp"
#include "../libcommon/IndiPropertyMap.hpp"
#include "../libcommon/IndiXmlParser.hpp"
#include "../libcommon/MutexLock.hpp"
#include "../libcommon/ReadWriteLock.hpp"
#include "../libcommon/System.hpp"
#include "../libcommon/SystemSocket.hpp"
#include "../libcommon/Thread.hpp"
#include "../libcommon/TimeStamp.hpp"
#include "../liblilxml/base64.h"
#include "../liblilxml/lilxml.h"

/// Run a check repeatedly from several threads that are released together.
/** \returns true if every iteration of every thread passed, false otherwise.
 */
static bool runConcurrently( size_t nThreads /**< [in] Number of threads, each given its index in [0, nThreads). */,
                             const std::function<bool( size_t )> &check /**< [in] Check run on each iteration; returns
                                                                                false on a mismatch. */ )
{
    std::atomic<bool>        start{ false };
    std::atomic<bool>        mismatch{ false };
    std::vector<std::thread> threads;

    for( size_t index = 0; index < nThreads; ++index )
    {
        threads.emplace_back(
            [&, index]()
            {
                while( !start.load( std::memory_order_acquire ) )
                {
                    std::this_thread::yield();
                }

                for( unsigned iteration = 0; iteration < 10000 && !mismatch.load( std::memory_order_relaxed );
                     ++iteration )
                {
                    if( !check( index ) )
                    {
                        mismatch.store( true, std::memory_order_relaxed );
                    }
                }
            } );
    }

    start.store( true, std::memory_order_release );

    for( auto &thread : threads )
    {
        thread.join();
    }

    return !mismatch.load( std::memory_order_relaxed );
}

namespace IndiXmlParser_test
{

/// Build a property-definition XML message without starting an INDI connection.
static std::string
makeDefTextXml( pcf::IndiProperty &ip /**< [in] Property whose attributes and elements are serialized. */ )
{
    pcf::IndiMessage   im( pcf::IndiMessage::Define, ip );
    pcf::IndiXmlParser parser( im );
    return parser.createXmlString();
}

/// Build a property-update XML message without starting an INDI connection.
static std::string
makeSetTextXml( pcf::IndiProperty &ip /**< [in] Property whose attributes and elements are serialized. */ )
{
    pcf::IndiMessage   im( pcf::IndiMessage::SetProperty, ip );
    pcf::IndiXmlParser parser( im );
    return parser.createXmlString();
}

/// Preserve valid XML escaping throughout property definitions and updates.
SCENARIO( "IndiXmlParser escapes special characters in attributes", "[IndiXmlParser]" )
{
    GIVEN( "a defTextVector property whose label contains a double-quote" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "testDevice" );
        ip.setName( "testProp" );
        ip.setLabel( "Label with \"quotes\"" );
        ip.setState( pcf::IndiProperty::Ok );
        ip.setPerm( pcf::IndiProperty::ReadOnly );

        pcf::IndiElement el( "elem1", "value" );
        ip.add( el );

        WHEN( "XML is generated" )
        {
            std::string xml = makeDefTextXml( ip );

            THEN( "the label attribute must not contain a bare double-quote" )
            {
                // The raw double-quote should have been replaced with &quot;
                REQUIRE( xml.find( "label=\"Label with &quot;quotes&quot;\"" ) != std::string::npos );
            }
        }
    }

    GIVEN( "a defTextVector property whose label contains < and >" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "testDevice" );
        ip.setName( "testProp" );
        ip.setLabel( "Temp <sensor>" );
        ip.setState( pcf::IndiProperty::Ok );
        ip.setPerm( pcf::IndiProperty::ReadOnly );

        pcf::IndiElement el( "elem1", "value" );
        ip.add( el );

        WHEN( "XML is generated" )
        {
            std::string xml = makeDefTextXml( ip );

            THEN( "< and > in the label must be escaped" )
            {
                REQUIRE( xml.find( "label=\"Temp &lt;sensor&gt;\"" ) != std::string::npos );
            }
        }
    }

    GIVEN( "a defTextVector property whose group contains an ampersand" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "testDevice" );
        ip.setName( "testProp" );
        ip.setGroup( "Science & Engineering" );
        ip.setState( pcf::IndiProperty::Ok );
        ip.setPerm( pcf::IndiProperty::ReadOnly );

        pcf::IndiElement el( "elem1", "value" );
        ip.add( el );

        WHEN( "XML is generated" )
        {
            std::string xml = makeDefTextXml( ip );

            THEN( "the ampersand in the group must be escaped as &amp;" )
            {
                REQUIRE( xml.find( "group=\"Science &amp; Engineering\"" ) != std::string::npos );
            }
        }
    }

    GIVEN( "a setTextVector property whose message contains a double-quote" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "testDevice" );
        ip.setName( "testProp" );
        ip.setMessage( "Error: unexpected \"value\"" );

        pcf::IndiElement el( "elem1", "value" );
        ip.add( el );

        WHEN( "XML is generated" )
        {
            std::string xml = makeSetTextXml( ip );

            THEN( "the double-quote in message must be escaped as &quot;" )
            {
                REQUIRE( xml.find( "message=\"Error: unexpected &quot;value&quot;\"" ) != std::string::npos );
            }
        }
    }

    GIVEN( "a defTextVector property whose element label contains < and >" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "testDevice" );
        ip.setName( "testProp" );
        ip.setState( pcf::IndiProperty::Ok );
        ip.setPerm( pcf::IndiProperty::ReadOnly );

        pcf::IndiElement el( "elem1", "value" );
        el.setLabel( "Range <0,100>" );
        ip.add( el );

        WHEN( "XML is generated" )
        {
            std::string xml = makeDefTextXml( ip );

            THEN( "< and > in the element label must be escaped" )
            {
                REQUIRE( xml.find( "label=\"Range &lt;0,100&gt;\"" ) != std::string::npos );
            }
        }
    }

    GIVEN( "a property whose device name contains a double-quote" )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );
        ip.setDevice( "dev\"ice" );
        ip.setName( "testProp" );
        ip.setState( pcf::IndiProperty::Ok );
        ip.setPerm( pcf::IndiProperty::ReadOnly );

        pcf::IndiElement el( "elem1", "value" );
        ip.add( el );

        WHEN( "XML is generated" )
        {
            std::string xml = makeDefTextXml( ip );

            THEN( "the double-quote in device must be escaped as &quot;" )
            {
                REQUIRE( xml.find( "device=\"dev&quot;ice\"" ) != std::string::npos );
            }
        }
    }

    GIVEN( "a createSafeXmlString call with all special characters" )
    {
        WHEN( "escaping ampersand, less-than, greater-than, and double-quote" )
        {
            std::string result = pcf::IndiXmlParser::createSafeXmlString( "a&b<c>d\"e" );

            THEN( "all characters are properly escaped" )
            {
                REQUIRE( result == "a&amp;b&lt;c&gt;d&quot;e" );
            }
        }
    }
}

/// Keep the existing entity substitutions and C-string termination behavior.
TEST_CASE( "IndiXmlParser escaping preserves literal text and entity semantics", "[IndiXmlParser]" )
{
    CHECK( pcf::IndiXmlParser::createSafeXmlString( "" ).empty() );
    CHECK( pcf::IndiXmlParser::createSafeXmlString( "plain text\t\n" ) == "plain text\t\n" );
    CHECK( pcf::IndiXmlParser::createSafeXmlString( "&<>'\"" ) == "&amp;&lt;&gt;&apos;&quot;" );
    CHECK( pcf::IndiXmlParser::createSafeXmlString( "&amp;" ) == "&amp;amp;" );
    CHECK( pcf::IndiXmlParser::createSafeXmlString( std::string( "a\0&b", 4 ) ) == "a" );
}

/// Concurrent definition/update formatting must never reuse another caller's escape buffer.
TEST_CASE( "IndiXmlParser serializes independent messages concurrently", "[IndiXmlParser][concurrency]" )
{
    const std::array<std::string, 4> inputs{ "dm01disp07_aux -> dm01disp07", "cameraHz", "gain & <value> '\"", "" };
    const std::array<std::string, 4> expected{
        "dm01disp07_aux -&gt; dm01disp07", "cameraHz", "gain &amp; &lt;value&gt; &apos;&quot;", "" };

    // Each thread gets its own property so this checks serialization, not concurrent property mutation.
    std::vector<pcf::IndiProperty> properties;

    for( const auto &input : inputs )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Text );
        property.setDevice( "testDevice" );
        property.setName( "mapping" );
        property.setState( pcf::IndiProperty::Ok );
        property.setPerm( pcf::IndiProperty::ReadOnly );
        property.add( pcf::IndiElement( "source", input ) );
        properties.push_back( property );
    }

    CHECK( runConcurrently( inputs.size(),
                            [&]( size_t index )
                            {
                                const auto escaped = pcf::IndiXmlParser::createSafeXmlString( inputs[index] );
                                const auto xml     = index % 2 ? makeSetTextXml( properties[index] )
                                                               : makeDefTextXml( properties[index] );

                                return escaped == expected[index] && xml.find( expected[index] ) != std::string::npos;
                            } ) );
}

/// Parsed messages reserialize through lilxml, whose entity buffer must be private to each thread.
TEST_CASE( "IndiXmlParser reserializes parsed messages concurrently", "[IndiXmlParser][concurrency]" )
{
    const std::array<std::string, 4> values{ "a &lt;b&gt;", "x &amp; y", "&quot;q&apos;", "plain" };

    // Parse and reserialize each message once to build the per-thread references.
    // Parsers are not copyable, so construct them in place.
    std::vector<pcf::IndiXmlParser> parsers( values.size() );
    std::vector<std::string>        references;

    for( size_t index = 0; index < values.size(); ++index )
    {
        std::string error;
        parsers[index].parseXml( "<setTextVector device=\"dev&amp;" + std::to_string( index ) +
                                     "\" name=\"prop\">\n<oneText name=\"elem\">" + values[index] +
                                     "</oneText>\n</setTextVector>\n",
                                 error );

        REQUIRE( error.empty() );
        REQUIRE( parsers[index].getState() == pcf::IndiXmlParser::CompleteState );

        references.push_back( parsers[index].createXmlString() );

        INFO( references.back() );

        CHECK( references.back().find( "device=\"dev&amp;" + std::to_string( index ) + "\"" ) != std::string::npos );
        CHECK( references.back().find( values[index] ) != std::string::npos );
    }

    // Reparsing each reference must reproduce it exactly while other threads print through lilxml.
    CHECK( runConcurrently( values.size(),
                            [&]( size_t index )
                            {
                                pcf::IndiXmlParser reparsed;
                                std::string        error;

                                reparsed.parseXml( references[index], error );

                                return parsers[index].createXmlString() == references[index] &&
                                       reparsed.createXmlString() == references[index];
                            } ) );
}

/// lilxml clones print and reparse a tree, so they must preserve entities and tolerate an empty tree.
TEST_CASE( "lilxml clones parsed trees", "[IndiXmlParser][lilxml]" )
{
    // IndiXmlParser copies would clone in-progress lilxml state, so copying is disabled at compile time.
    static_assert( !std::is_copy_constructible_v<pcf::IndiXmlParser> );
    static_assert( !std::is_copy_assignable_v<pcf::IndiXmlParser> );

    char    xml[] = "<oneText name=\"a&amp;b\">x &lt;y&gt; &quot;z&apos;</oneText>";
    char    ynot[1024];
    XMLEle *root = ::parseXML( xml, ynot );

    REQUIRE( root != nullptr );

    XMLEle *clone = ::cloneXMLEle( root );

    REQUIRE( clone != nullptr );
    CHECK( std::string( ::findXMLAttValu( clone, "name" ) ) == "a&b" );
    CHECK( std::string( ::pcdataXMLEle( clone ) ) == "x <y> \"z'" );
    CHECK( ::cloneXMLEle( nullptr ) == nullptr );

    ::delXMLEle( clone );
    ::delXMLEle( root );
}

} //namespace IndiXmlParser_test

namespace libcommon_test
{

/// Concurrent ISO-8601 formatting must not share gmtime's static broken-down time.
TEST_CASE( "TimeStamp formats ISO-8601 strings concurrently", "[TimeStamp][concurrency]" )
{
    const std::array<timeval, 4> times{
        { { 0, 0 }, { 946684800, 1 }, { 1700000000, 500000 }, { 2000000000, 999999 } } };
    const std::array<std::string, 4> expected{ "1970-01-01T00:00:00.000000Z",
                                               "2000-01-01T00:00:00.000001Z",
                                               "2023-11-14T22:13:20.500000Z",
                                               "2033-05-18T03:33:20.999999Z" };

    CHECK( runConcurrently( times.size(),
                            [&]( size_t index )
                            {
                                return pcf::TimeStamp( times[index] ).getFormattedIso8601Str() == expected[index];
                            } ) );
}

/// Calendar construction must interpret its fields as UTC without modifying the process TZ.
TEST_CASE( "TimeStamp calendar constructor uses UTC without changing TZ", "[TimeStamp]" )
{
    const char       *pcTz = ::getenv( "TZ" );
    const std::string savedTz( pcTz != nullptr ? pcTz : "" );

    ::setenv( "TZ", "MST7", 1 );
    ::tzset();

    pcf::TimeStamp ts( 2023, 11, 14, 22, 13, 20 );

    CHECK( ts.getTimeVal().tv_sec == 1700000000 );
    CHECK( ts.getTimeVal().tv_usec == 0 );
    CHECK( std::string( ::getenv( "TZ" ) ) == "MST7" );

    if( pcTz != nullptr )
    {
        ::setenv( "TZ", savedTz.c_str(), 1 );
    }
    else
    {
        ::unsetenv( "TZ" );
    }

    ::tzset();
}

/// Interface enumeration must produce dotted-quad addresses from caller-owned conversion buffers.
TEST_CASE( "SystemSocket enumerates interface addresses", "[SystemSocket]" )
{
    std::vector<pcf::SystemSocket::Interface> interfaces;

    pcf::SystemSocket::getInterfaces( interfaces );

    for( const auto &interface : interfaces )
    {
        in_addr addr;

        INFO( interface.getName() );
        CHECK( ::inet_pton( AF_INET, interface.getIp().c_str(), &addr ) == 1 );
        CHECK( ::inet_pton( AF_INET, interface.getBroadcast().c_str(), &addr ) == 1 );
    }
}

} //namespace libcommon_test
