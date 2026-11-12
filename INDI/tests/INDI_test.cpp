/** \file INDI_test.cpp
  * \brief Catch2 tests for the INDI library, including IndiXmlParser XML escaping.
  *
  * History:
  */
#include "../../tests/catch2/catch.hpp"

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

/// Helper: build a defTextVector XML string from an IndiProperty.
static std::string makeDefTextXml( pcf::IndiProperty &ip )
{
   pcf::IndiMessage im( pcf::IndiMessage::Define, ip );
   pcf::IndiXmlParser parser( im );
   return parser.createXmlString();
}

/// Helper: build a setTextVector XML string from an IndiProperty.
static std::string makeSetTextXml( pcf::IndiProperty &ip )
{
   pcf::IndiMessage im( pcf::IndiMessage::SetProperty, ip );
   pcf::IndiXmlParser parser( im );
   return parser.createXmlString();
}

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

} //namespace IndiXmlParser_test

namespace IndiElement_test
{

/// IndiElement writes bools as 1/0, so they are valid INDI numbers, and reads both 1/0 and true/false.
SCENARIO( "IndiElement bool values are numeric and backwards compatible", "[IndiElement]" )
{
   GIVEN( "bool values written through each setter" )
   {
      pcf::IndiElement elSet( "e" );
      elSet.set( true );

      pcf::IndiElement elAssign( "e" );
      elAssign = false;

      pcf::IndiElement elCtor( "e", true );

      THEN( "they are written as 1 and 0" )
      {
         REQUIRE( elSet.getValue() == "1" );
         REQUIRE( elAssign.getValue() == "0" );
         REQUIRE( elCtor.getValue() == "1" );
         REQUIRE( elSet.get<bool>() == true );
         REQUIRE( elAssign.get<bool>() == false );
      }
   }

   GIVEN( "element strings from new and old writers" )
   {
      THEN( "get<bool> reads true/false, 1/0 and other numbers" )
      {
         REQUIRE( pcf::IndiElement( "e", std::string( "true" ) ).get<bool>() == true );
         REQUIRE( pcf::IndiElement( "e", std::string( "false" ) ).getValue<bool>() == false );
         REQUIRE( pcf::IndiElement( "e", std::string( "1" ) ).get<bool>() == true );
         REQUIRE( pcf::IndiElement( "e", std::string( "0" ) ).getValue<bool>() == false );
         REQUIRE( pcf::IndiElement( "e", std::string( "2.5" ) ).get<bool>() == true );
         REQUIRE( pcf::IndiElement( "e", std::string( "0.0" ) ).get<bool>() == false );
         REQUIRE( pcf::IndiElement( "e", std::string( "junk" ) ).get<bool>() == false );
      }
   }

   GIVEN( "non-bool values" )
   {
      pcf::IndiElement el( "e" );
      el = 1.0 / 3.0;

      THEN( "formatting is unchanged" )
      {
         REQUIRE( el.getValue() == pcf::formatIndiValue( 1.0 / 3.0 ) );
         REQUIRE( el.getValue() == "0.333333333333333" );
         REQUIRE( el.get<double>() == Approx( 1.0 / 3.0 ) );
         REQUIRE( pcf::formatIndiValue( 42 ) == "42" );
      }
   }
}

} // namespace IndiElement_test
