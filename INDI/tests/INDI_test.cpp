/** \file INDI_test.cpp
 * \brief Catch2 tests for the INDI library, including IndiXmlParser XML escaping.
 * \ingroup test_indi
 */
#include "../../tests/catch2/catch.hpp"

#include <array>
#include <atomic>
#include <thread>

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
    std::atomic<bool>          start{ false };
    std::atomic<bool>          mismatch{ false };
    std::array<std::thread, 4> threads;

    for( size_t index = 0; index < threads.size(); ++index )
    {
        threads[index] = std::thread(
            [&, index]()
            {
                // Properties are thread-local so this checks serialization, not concurrent property mutation.
                pcf::IndiProperty property( pcf::IndiProperty::Text );
                property.setDevice( "testDevice" );
                property.setName( "mapping" );
                property.setState( pcf::IndiProperty::Ok );
                property.setPerm( pcf::IndiProperty::ReadOnly );
                property.add( pcf::IndiElement( "source", inputs[index] ) );

                while( !start.load( std::memory_order_acquire ) )
                {
                    std::this_thread::yield();
                }

                for( unsigned iteration = 0; iteration < 10000; ++iteration )
                {
                    const auto escaped = pcf::IndiXmlParser::createSafeXmlString( inputs[index] );
                    const auto xml     = index % 2 ? makeSetTextXml( property ) : makeDefTextXml( property );

                    if( escaped != expected[index] || xml.find( expected[index] ) == std::string::npos )
                    {
                        mismatch.store( true, std::memory_order_relaxed );
                        break;
                    }
                }
            } );
    }

    start.store( true, std::memory_order_release );

    for( auto &thread : threads )
    {
        thread.join();
    }

    CHECK_FALSE( mismatch.load( std::memory_order_relaxed ) );
}

} //namespace IndiXmlParser_test
