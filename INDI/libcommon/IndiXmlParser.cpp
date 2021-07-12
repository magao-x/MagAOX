/// IndiXmlParser.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "../liblilxml/base64.h"
#include "IndiMessage.hpp"
#include "IndiProperty.hpp"
#include "IndiXmlParser.hpp"

using std::runtime_error;
using std::string;
using std::stringbuf;
using std::stringstream;
using std::boolalpha;
using std::vector;
using std::istream;
using std::ostream;
using pcf::IndiElement;
using pcf::IndiMessage;
using pcf::IndiProperty;
using pcf::IndiXmlParser;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

IndiXmlParser::IndiXmlParser( const string &szProtocolVersion )
{
  init( szProtocolVersion );
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from an existing IndiMessage.

IndiXmlParser::IndiXmlParser( const IndiMessage &imSend,
                              const string &szProtocolVersion )
{
  init( szProtocolVersion );

  const IndiProperty &ipSend = imSend.getProperty();

  // Each Indi message has different attributes depending on the type of message.
  switch ( imSend.getType() )
  {
    case IndiMessage::Define:
      switch ( ipSend.getType() )
      {
        case IndiProperty::BLOB: createDefBLOBVector( ipSend ); break;
        case IndiProperty::Light: createDefLightVector( ipSend ); break;
        case IndiProperty::Number: createDefNumberVector( ipSend ); break;
        case IndiProperty::Switch: createDefSwitchVector( ipSend ); break;
        case IndiProperty::Text: createDefTextVector( ipSend ); break;
        default:
        case IndiProperty::Unknown:
          throw runtime_error( "Unknown INDI property DEF type is not allowed." );
          break;
      }
      break;

    case IndiMessage::Delete:
      createDelProperty( ipSend );
      break;

    case IndiMessage::EnableBLOB:
      createEnableBLOB( ipSend );
      break;

    case IndiMessage::GetProperties:
      createGetProperties( ipSend );
      break;

    case IndiMessage::Message:
      createMessage( ipSend );
      break;

    case IndiMessage::NewProperty:
      switch ( ipSend.getType() )
      {
        case IndiProperty::BLOB: createNewBLOBVector( ipSend ); break;
        case IndiProperty::Number: createNewNumberVector( ipSend ); break;
        case IndiProperty::Switch: createNewSwitchVector( ipSend ); break;
        case IndiProperty::Text: createNewTextVector( ipSend ); break;
        default:
        case IndiProperty::Unknown:
          throw runtime_error( "Unknown INDI property NEW type is not allowed." );
          break;
      }
      break;

    case IndiMessage::SetProperty:
      switch ( ipSend.getType() )
      {
        case IndiProperty::BLOB: createSetBLOBVector( ipSend ); break;
        case IndiProperty::Light: createSetLightVector( ipSend ); break;
        case IndiProperty::Number: createSetNumberVector( ipSend ); break;
        case IndiProperty::Switch: createSetSwitchVector( ipSend ); break;
        case IndiProperty::Text: createSetTextVector( ipSend ); break;
        default:
        case IndiProperty::Unknown:
          throw runtime_error( "Unknown INDI property SET type is not allowed." );
          break;
      }
      break;

    case IndiMessage::Unknown:
      throw runtime_error( "Unknown INDI property type is not allowed." );
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// createSafeXmlString
/// This takes any string and replaces the '&', '<', '>', '\', and '"'
/// with its equivalent XML entity: &amp; &lt; &gt; &apos; and &quot;

string IndiXmlParser::createSafeXmlString( const string &szText )
{
  return ::entityXML( (char*)( szText.c_str() ) );
}

////////////////////////////////////////////////////////////////////////////////
// DefTextVector REQUIRED: device name state perm
// DefTextVector IMPLIED:  label group timeout timestamp message
// DefText REQUIRED: name
// DefText IMPLIED: label

void IndiXmlParser::createDefTextVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "defTextVector";
  string szElementTag = "defText";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );
  if ( ip.hasValidState() == false )
    throw runtime_error( szDescrip + " must have attribute 'state' defined." );
  if ( ip.hasValidPerm() == false )
    throw runtime_error( szDescrip + " must have attribute 'perm' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";
  m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  m_ssXml << " perm=\"" << ip.getPropertyPermString( ip.getPerm() ) << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidLabel() == true )
    m_ssXml << " label=\"" << ip.getLabel() << "\"";
  if ( ip.hasValidGroup() == true )
    m_ssXml << " group=\"" << ip.getGroup() << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    // "implied" means that if they are not defined, don't add them. Adding an
    // empty "implied" attribute to the generated XML is an error.
    if ( ip[ii].hasValidLabel() == true )
      m_ssXml << " label=\"" << ip[ii].getLabel() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << createSafeXmlString( ip[ii].getValue() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// SetTextVector REQUIRED: device name
// SetTextVector IMPLIED:  state timeout timestamp message
// oneText REQUIRED: name

void IndiXmlParser::createSetTextVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "setTextVector";
  string szElementTag = "oneText";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidState() == true )
    m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << createSafeXmlString( ip[ii].getValue() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// NewTextVector REQUIRED: device name
// NewTextVector IMPLIED:  timestamp
// oneText REQUIRED: name

void IndiXmlParser::createNewTextVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "newTextVector";
  string szElementTag = "oneText";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << createSafeXmlString( ip[ii].getValue() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// DefNumberVector REQUIRED: device name state perm
// DefNumberVector IMPLIED:  label group timeout timestamp message
// DefNumber REQUIRED: name format min max step
// DefNumber IMPLIED: label

void IndiXmlParser::createDefNumberVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "defNumberVector";
  string szElementTag = "defNumber";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );
  if ( ip.hasValidState() == false )
    throw runtime_error( szDescrip + " must have attribute 'state' defined." );
  if ( ip.hasValidPerm() == false )
    throw runtime_error( szDescrip + " must have attribute 'perm' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";
  m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  m_ssXml << " perm=\"" << ip.getPropertyPermString( ip.getPerm() ) << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidLabel() == true )
    m_ssXml << " label=\"" << ip.getLabel() << "\"";
  if ( ip.hasValidGroup() == true )
    m_ssXml << " group=\"" << ip.getGroup() << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );
    if ( ip[ii].hasValidFormat() == false )
      throw runtime_error( szElementTag + " must have attribute 'format' defined." );
    if ( ip[ii].hasValidMin() == false )
      throw runtime_error( szElementTag + " must have attribute 'min' defined." );
    if ( ip[ii].hasValidMax() == false )
      throw runtime_error( szElementTag + " must have attribute 'max' defined." );
    if ( ip[ii].hasValidStep() == false )
      throw runtime_error( szElementTag + " must have attribute 'step' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";
    m_ssXml << " format=\"" << ip[ii].getFormat() << "\"";
    m_ssXml << " min=\"" << ip[ii].getMin() << "\"";
    m_ssXml << " max=\"" << ip[ii].getMax() << "\"";
    m_ssXml << " step=\"" << ip[ii].getStep() << "\"";

    // "implied" means that if they are not defined, don't add them. Adding an
    // empty "implied" attribute to the generated XML is an error.
    if ( ip[ii].hasValidLabel() == true )
      m_ssXml << " label=\"" << ip[ii].getLabel() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << createSafeXmlString( ip[ii].getValue() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// SetNumberVector REQUIRED: device name
// SetNumberVector IMPLIED:  state timeout timestamp message
// oneNumber REQUIRED: name

void IndiXmlParser::createSetNumberVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "setNumberVector";
  string szElementTag = "oneNumber";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidState() == true )
    m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << createSafeXmlString( ip[ii].getValue() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// NewNumberVector REQUIRED: device name
// NewNumberVector IMPLIED:  timestamp
// oneNumber REQUIRED: name

void IndiXmlParser::createNewNumberVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "newNumberVector";
  string szElementTag = "oneNumber";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << createSafeXmlString( ip[ii].getValue() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// DefSwitchVector REQUIRED: device name state perm rule
// DefSwitchVector IMPLIED:  label group timeout timestamp message
// DefSwitch REQUIRED: name
// DefSwitch IMPLIED: label

void IndiXmlParser::createDefSwitchVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "defSwitchVector";
  string szElementTag = "defSwitch";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );
  if ( ip.hasValidState() == false )
    throw runtime_error( szDescrip + " must have attribute 'state' defined." );
  if ( ip.hasValidPerm() == false )
    throw runtime_error( szDescrip + " must have attribute 'perm' defined." );
  if ( ip.hasValidRule() == false )
    throw runtime_error( szDescrip + " must have attribute 'rule' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";
  m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  m_ssXml << " perm=\"" << ip.getPropertyPermString( ip.getPerm() ) << "\"";
  m_ssXml << " rule=\"" << ip.getSwitchRuleString( ip.getRule() ) << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidLabel() == true )
    m_ssXml << " label=\"" << ip.getLabel() << "\"";
  if ( ip.hasValidGroup() == true )
    m_ssXml << " group=\"" << ip.getGroup() << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    // "implied" means that if they are not defined, don't add them. Adding an
    // empty "implied" attribute to the generated XML is an error.
    if ( ip[ii].hasValidLabel() == true )
      m_ssXml << " label=\"" << ip[ii].getLabel() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << IndiElement::getSwitchStateString( ip[ii].getSwitchState() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// SetSwitchVector REQUIRED: device name
// SetSwitchVector IMPLIED:  state timeout timestamp message
// oneSwitch REQUIRED: name

void IndiXmlParser::createSetSwitchVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "setSwitchVector";
  string szElementTag = "oneSwitch";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidState() == true )
    m_ssXml << " state=\"" << IndiProperty::getPropertyStateString( ip.getState() ) << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << IndiElement::getSwitchStateString( ip[ii].getSwitchState() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// NewSwitchVector REQUIRED: device name
// NewSwitchVector IMPLIED:  timestamp
// oneSwitch REQUIRED: name

void IndiXmlParser::createNewSwitchVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "newSwitchVector";
  string szElementTag = "oneSwitch";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << IndiElement::getSwitchStateString( ip[ii].getSwitchState() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// DefLightVector REQUIRED: device name state
// DefLightVector IMPLIED:  label group timestamp message
// DefLight REQUIRED: name
// DefLight IMPLIED: label

void IndiXmlParser::createDefLightVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "defLightVector";
  string szElementTag = "defLight";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );
  if ( ip.hasValidState() == false )
    throw runtime_error( szDescrip + " must have attribute 'state' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";
  m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidLabel() == true )
    m_ssXml << " label=\"" << ip.getLabel() << "\"";
  if ( ip.hasValidGroup() == true )
    m_ssXml << " group=\"" << ip.getGroup() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    // "implied" means that if they are not defined, don't add them. Adding an
    // empty "implied" attribute to the generated XML is an error.
    if ( ip[ii].hasValidLabel() == true )
      m_ssXml << " label=\"" << ip[ii].getLabel() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << IndiElement::getLightStateString( ip[ii].getLightState() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// SetLightVector REQUIRED: device name
// SetLightVector IMPLIED:  state timestamp message
// oneLight REQUIRED: name

void IndiXmlParser::createSetLightVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "setLightVector";
  string szElementTag = "oneLight";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidState() == true )
    m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    m_ssXml << IndiElement::getLightStateString( ip[ii].getLightState() ) << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// DefBLOBVector REQUIRED: device name state perm
// DefBLOBVector IMPLIED:  label group timeout timestamp message
// DefBLOB REQUIRED: name
// DefBLOB IMPLIED: label

void IndiXmlParser::createDefBLOBVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "defBLOBVector";
  string szElementTag = "defBLOB";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );
  if ( ip.hasValidState() == false )
    throw runtime_error( szDescrip + " must have attribute 'state' defined." );
  if ( ip.hasValidPerm() == false )
    throw runtime_error( szDescrip + " must have attribute 'perm' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";
  m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  m_ssXml << " perm=\"" << ip.getPropertyPermString( ip.getPerm() ) << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidLabel() == true )
    m_ssXml << " label=\"" << ip.getLabel() << "\"";
  if ( ip.hasValidGroup() == true )
    m_ssXml << " group=\"" << ip.getGroup() << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";

    // "implied" means that if they are not defined, don't add them. Adding an
    // empty "implied" attribute to the generated XML is an error.
    if ( ip[ii].hasValidLabel() == true )
      m_ssXml << " label=\"" << ip[ii].getLabel() << "\"";

    m_ssXml << ">\r\n";

    // Now add the data for the element...
    // todo: This may need to be removed, as the BLOB has no initial value.
    // No data here!
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// SetBLOBVector REQUIRED: device name
// SetBLOBVector IMPLIED:  state timeout timestamp message
// oneBLOB REQUIRED: name size format

void IndiXmlParser::createSetBLOBVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "setBLOBVector";
  string szElementTag = "oneBLOB";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidState() == true )
    m_ssXml << " state=\"" << ip.getPropertyStateString( ip.getState() ) << "\"";
  if ( ip.hasValidTimeout() == true )
    m_ssXml << " timeout=\"" << ip.getTimeout() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );
    if ( ip[ii].hasValidSize() == false )
      throw runtime_error( szElementTag + " must have attribute 'size' defined." );
    if ( ip[ii].hasValidFormat() == false )
      throw runtime_error( szElementTag + " must have attribute 'format' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";
    m_ssXml << " size=\"" << ip[ii].getSize() << "\"";
    m_ssXml << " format=\"" << ip[ii].getFormat() << "\"";

    m_ssXml << ">\r\n";

    // Allocate a big buffer to hold the base64 encoded data.
    vector<unsigned char> vecBase64Buf( 4 * ip[ii].getValue().size() / 3 + 4 );

    // Now base64 encode the data.
    unsigned char *pcBase64Buf = &vecBase64Buf[0];
    unsigned char *pcValueBuf = (unsigned char *)( ip[ii].getValue().c_str() );
    unsigned int  uiValueBufLen = ip[ii].getValue().size();
    // unsigned int uiBase64BufLen =
    ::to64frombits( pcBase64Buf, pcValueBuf, uiValueBufLen );

    // Now add the data for the element...
    m_ssXml << pcBase64Buf << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// NewBLOBVector REQUIRED: device name
// NewBLOBVector IMPLIED:  timestamp
// oneBLOB REQUIRED: name size format

void IndiXmlParser::createNewBLOBVector( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "newBLOBVector";
  string szElementTag = "oneBLOB";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );
  if ( ip.hasValidName() == false )
    throw runtime_error( szDescrip + " must have attribute 'name' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " name=\"" << ip.getName() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";

  m_ssXml << ">\r\n";

  for ( unsigned int ii = 0; ii < ip.getNumElements(); ii++ )
  {
    m_ssXml << "\t<" << szElementTag;

    // "required" means that the attribute must be added and contain a valid
    // entry. If it is not part of the generated XML, it is an error.
    if ( ip[ii].hasValidName() == false )
      throw runtime_error( szElementTag + " must have attribute 'name' defined." );
    if ( ip[ii].hasValidSize() == false )
      throw runtime_error( szElementTag + " must have attribute 'size' defined." );
    if ( ip[ii].hasValidFormat() == false )
      throw runtime_error( szElementTag + " must have attribute 'format' defined." );

    m_ssXml << " name=\"" << ip[ii].getName() << "\"";
    m_ssXml << " size=\"" << ip[ii].getSize() << "\"";
    m_ssXml << " format=\"" << ip[ii].getFormat() << "\"";

    m_ssXml << ">\r\n";

    // Allocate a big buffer to hold the base64 encoded data.
    vector<unsigned char> vecBase64Buf( 4 * ip[ii].getValue().size() / 3 + 4 );

    // Now base64 encode the data.
    unsigned char *pcBase64Buf = &vecBase64Buf[0];
    unsigned char *pcValueBuf = (unsigned char *)( ip[ii].getValue().c_str() );
    unsigned int  uiValueBufLen = ip[ii].getValue().size();
    //unsigned int uiBase64BufLen =
    ::to64frombits( pcBase64Buf, pcValueBuf, uiValueBufLen );

    // Now add the data for the element...
    m_ssXml << pcBase64Buf << "\r\n";
    m_ssXml << "\t</" << szElementTag << ">\r\n";
  }
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// Message REQUIRED:
// Message IMPLIED:  device timestamp message

void IndiXmlParser::createMessage( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "message";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidDevice() == true )
    m_ssXml << " device=\"" << ip.getDevice() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// DelProperty REQUIRED: device
// DelProperty IMPLIED:  name timestamp message

void IndiXmlParser::createDelProperty( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "delProperty";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidName() == true )
    m_ssXml << " name=\"" << ip.getName() << "\"";
  m_ssXml << " timestamp=\"" << ip.getTimeStamp().getFormattedIso8601Str() << "\"";
  if ( ip.hasValidMessage() == true )
    m_ssXml << " message=\"" << ip.getMessage() << "\"";

  m_ssXml << ">\r\n";
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// GetProperties REQUIRED:
// GetProperties IMPLIED:  device name

void IndiXmlParser::createGetProperties( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "getProperties";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidDevice() == true )
    m_ssXml << " device=\"" << ip.getDevice() << "\"";
  if ( ip.hasValidName() == true )
    m_ssXml << " name=\"" << ip.getName() << "\"";

  m_ssXml << " version=\"" << m_szProtocolVersion << "\"";

  m_ssXml << ">\r\n";
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
// EnableBLOB REQUIRED: device
// EnableBLOB IMPLIED:  name

void IndiXmlParser::createEnableBLOB( const IndiProperty &ip )
{
  // What kind of message is this?
  string szTag = "enableBLOB";
  string szDescrip = szTag + " '" + ip.getName() + "'";

  // Clear out the xml stream.
  m_ssXml.str("");
  m_ssXml << "<" << szTag;

  // "required" means that the attribute must be added and contain a valid
  // entry. If it is not part of the generated XML, it is an error.
  if ( ip.hasValidDevice() == false )
    throw runtime_error( szDescrip + " must have attribute 'device' defined." );

  m_ssXml << " device=\"" << ip.getDevice() << "\"";

  // "implied" means that if they are not defined, don't add them. Adding an
  // empty "implied" attribute to the generated XML is an error.
  if ( ip.hasValidName() == true )
    m_ssXml << " name=\"" << ip.getName() << "\"";

  m_ssXml << ">\r\n";
  m_ssXml << IndiProperty::getBLOBEnableString( ip.getBLOBEnable() ) << "\r\n";
  m_ssXml << "</" << szTag << ">\r\n";
}

////////////////////////////////////////////////////////////////////////////////
///  Copy constructor.

IndiXmlParser::IndiXmlParser( const IndiXmlParser &ixp )
{
  m_plxMessage = ::cloneLilXML( ixp.m_plxMessage );
  m_pxeRoot = ::cloneXMLEle( ixp.m_pxeRoot );
  ::memcpy( m_pcMessage, ixp.m_pcMessage, MaxErrorMsgSize );
  m_szPrevXml = ixp.m_szPrevXml;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

IndiXmlParser::~IndiXmlParser()
{
  ::delLilXML( m_plxMessage );
  ::delXMLEle( m_pxeRoot );
}

////////////////////////////////////////////////////////////////////////////////
/// Assigns the internal data of this object from an existing one.

const IndiXmlParser &IndiXmlParser::operator=( const IndiXmlParser &ixpRhs )
{
  if ( &ixpRhs != this )
  {
    ::delLilXML( m_plxMessage );
    ::delXMLEle( m_pxeRoot );
    m_plxMessage = ::cloneLilXML( ixpRhs.m_plxMessage );
    m_pxeRoot = ::cloneXMLEle( ixpRhs.m_pxeRoot );
    ::memcpy( m_pcMessage, ixpRhs.m_pcMessage, MaxErrorMsgSize );
    m_szPrevXml = ixpRhs.m_szPrevXml;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all attributes and reset this object.

void IndiXmlParser::clear()
{
  ::delLilXML( m_plxMessage );
  ::delXMLEle( m_pxeRoot );
  m_plxMessage = ::newLilXML();
  m_pxeRoot = NULL;
  ::memset( m_pcMessage, 0, MaxErrorMsgSize*sizeof(char) );
  m_szPrevXml = "";
  m_ssXml.str( "" );
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize all data - assume we have just been created.

void IndiXmlParser::init( const string &szProtocolVersion )
{
  m_szProtocolVersion = szProtocolVersion;
  m_plxMessage = ::newLilXML();
  m_pxeRoot = NULL;
  ::memset( m_pcMessage, 0, MaxErrorMsgSize*sizeof(char) );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the data formatted as an xml string.

string IndiXmlParser::createXmlString() const
{
  string szXml;

  if ( m_pxeRoot != NULL )
  {
    int nNumChars = 1 + ::sprlXMLEle( m_pxeRoot, 0 );
    //szXml.reserve( nNumChars );
    char *pcXml = ( char * )( ::malloc( nNumChars * sizeof( char ) ) );
    ::memset( pcXml, 0, sizeof( char ) * nNumChars );
    ::sprXMLEle( pcXml, m_pxeRoot, 0 );
    szXml = string( pcXml );
  }
  else if ( m_ssXml.str().size() > 0 )
  {
    szXml = m_ssXml.str();
  }

  return szXml;
}

////////////////////////////////////////////////////////////////////////////////

IndiXmlParser::State IndiXmlParser::getState() const
{
  return ( m_pxeRoot != NULL ) ? ( CompleteState ) : ( IncompleteState );
}

////////////////////////////////////////////////////////////////////////////////
/// Sets all the data in this object from formatted xml. First clears all
/// internal data, then parses the xml.

void IndiXmlParser::parseXml( const char *pcXml,
                              const unsigned int &uiNumBytes,
                              string &szErrorMsg )
{
  parseXml( string( pcXml, uiNumBytes ), szErrorMsg );
}

////////////////////////////////////////////////////////////////////////////////
/// Sets all the data in this object from formatted xml. First clears all
/// internal data, then parses the xml.

void IndiXmlParser::parseXml( const string &szXml,
                              string &szErrorMsg )
{
  // Do we have any text left from a previous call?
  string szCurrXml = m_szPrevXml + szXml;

  // Make sure the message is empty.
  szErrorMsg.clear();
  ::memset( m_pcMessage, 0, MaxErrorMsgSize * sizeof(char) );

  // Reset the internal data.
  ::delXMLEle( m_pxeRoot );
  m_pxeRoot = NULL;

  unsigned int uiProcessed = 0;
  XMLEle *pxeNewRoot = NULL;
  // The pxeNewRoot will be NULL until the document is complete.
  for ( ; uiProcessed < szCurrXml.size() && pxeNewRoot == NULL; uiProcessed++ )
  {
    pxeNewRoot = ::readXMLEle( m_plxMessage,
                               ( int )( szCurrXml[uiProcessed] ),
                               ( char * )( m_pcMessage ) );
    // At this point, if there was an error, pxeNewRoot will be null, and
    // m_pcMessage will contain why.
    if ( ::strlen( ( char * )( m_pcMessage ) ) > 0 )
    {
      // Save the message.
      szErrorMsg = string( ( const char * )( m_pcMessage ) );
      std::cerr << "Error processing XML: '"
                << szXml << "'" << std::endl
                << "Error message: '" << szErrorMsg << "'" << std::endl;
      // Clear out the existing data and start over.
      clear();
      break;
    }
  }

  // When we get here, the message was empty, so there was no error.
  // We may or may not be complete.
  m_pxeRoot = pxeNewRoot;

  // Store the left-over text for the next call.
  m_szPrevXml = szCurrXml.substr( uiProcessed, szCurrXml.size() );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the attribute value with name szAttributeName.

string IndiXmlParser::getAttributeValue( const string &szAttributeName,
    XMLEle *pxeRoot ) const
{
  if ( pxeRoot == NULL )
    return "";

  XMLAtt *attrib = ::findXMLAtt( pxeRoot, szAttributeName.c_str() );

  if ( attrib == NULL )
    return "";

  return string( ::valuXMLAtt( attrib ) );
}

////////////////////////////////////////////////////////////////////////////////
/// Get the IndiMessage created from the xml data.

IndiMessage IndiXmlParser::createIndiMessage() const
{
  IndiMessage::Type tMsgType;
  IndiProperty ipNew;
  string szValue;

  if ( m_pxeRoot != NULL )
  {
    // What type of message is this?
    string szType = string( ::tagXMLEle( m_pxeRoot ) );
    IndiXmlParser::MessageType tType = convertStringToType( szType );

    switch ( tType )
    {
      case UnknownType:
        tMsgType = IndiMessage::Unknown;
        break;
      // Define properties.
      case DefBLOBVector:
        ipNew = IndiProperty( IndiProperty::BLOB );
        tMsgType = IndiMessage::Define;
        break;
      case DefLightVector:
        ipNew = IndiProperty( IndiProperty::Light );
        tMsgType = IndiMessage::Define;
        break;
      case DefNumberVector:
        ipNew = IndiProperty( IndiProperty::Number );
        tMsgType = IndiMessage::Define;
        break;
      case DefSwitchVector:
        ipNew = IndiProperty( IndiProperty::Switch );
        tMsgType = IndiMessage::Define;
        break;
      case DefTextVector:
        ipNew = IndiProperty( IndiProperty::Text );
        tMsgType = IndiMessage::Define;
        break;
      // Delete properties.
      case DelProperty:
        ipNew = IndiProperty( IndiProperty::Unknown );
        tMsgType = IndiMessage::Delete;
        break;
      // Enable blobs for a client.
      case EnableBLOB:
        ipNew = IndiProperty( IndiProperty::Unknown );
        tMsgType = IndiMessage::EnableBLOB;
        break;
      // Command to enable snooping messages from other devices.
      case GetProperties:
        ipNew = IndiProperty( IndiProperty::Unknown );
        tMsgType = IndiMessage::GetProperties;
        break;
      // A message.
      case Message:
        ipNew = IndiProperty( IndiProperty::Unknown );
        tMsgType = IndiMessage::Message;
        break;
      // Update properties.
      case NewBLOBVector:
        ipNew = IndiProperty( IndiProperty::BLOB );
        tMsgType = IndiMessage::NewProperty;
        break;
      case NewNumberVector:
        ipNew = IndiProperty( IndiProperty::Number );
        tMsgType = IndiMessage::NewProperty;
        break;
      case NewSwitchVector:
        ipNew = IndiProperty( IndiProperty::Switch );
        tMsgType = IndiMessage::NewProperty;
        break;
      case NewTextVector:
        ipNew = IndiProperty( IndiProperty::Text );
        tMsgType = IndiMessage::NewProperty;
        break;
      // Set properties.
      case SetBLOBVector:
        ipNew = IndiProperty( IndiProperty::BLOB );
        tMsgType = IndiMessage::SetProperty;
        break;
      case SetLightVector:
        ipNew = IndiProperty( IndiProperty::Light );
        tMsgType = IndiMessage::SetProperty;
        break;
      case SetNumberVector:
        ipNew = IndiProperty( IndiProperty::Number );
        tMsgType = IndiMessage::SetProperty;
        break;
      case SetSwitchVector:
        ipNew = IndiProperty( IndiProperty::Switch );
        tMsgType = IndiMessage::SetProperty;
        break;
      case SetTextVector:
        ipNew = IndiProperty( IndiProperty::Text );
        tMsgType = IndiMessage::SetProperty;
        break;
    };

    // Set the attributes.
    if ( ( szValue = getAttributeValue( "device", m_pxeRoot ) ).size() > 0 )
      ipNew.setDevice( szValue );
    if ( ( szValue = getAttributeValue( "group", m_pxeRoot ) ).size() > 0 )
      ipNew.setGroup( szValue );
    if ( ( szValue = getAttributeValue( "label", m_pxeRoot ) ).size() > 0 )
      ipNew.setLabel( szValue );
    if ( ( szValue = getAttributeValue( "message", m_pxeRoot ) ).size() > 0 )
      ipNew.setMessage( szValue );
    if ( ( szValue = getAttributeValue( "name", m_pxeRoot ) ).size() > 0 )
      ipNew.setName( szValue );
    if ( ( szValue = getAttributeValue( "perm", m_pxeRoot ) ).size() > 0 )
      ipNew.setPerm( IndiProperty::getPropertyPermType( szValue ) );
    if ( ( szValue = getAttributeValue( "rule", m_pxeRoot ) ).size() > 0 )
      ipNew.setRule( IndiProperty::getSwitchRuleType( szValue ) );
    if ( ( szValue = getAttributeValue( "state", m_pxeRoot ) ).size() > 0 )
      ipNew.setState( IndiProperty::getPropertyStateType( szValue ) );
    if ( ( szValue = getAttributeValue( "timeout", m_pxeRoot ) ).size() > 0 )
    {
      stringstream ssTimeout;
      ssTimeout << szValue;
      double xTimeout;
      ssTimeout >> xTimeout;
      ipNew.setTimeout( xTimeout );
    }
    if ( ( szValue = getAttributeValue( "timestamp", m_pxeRoot ) ).size() > 0 )
    {
      TimeStamp tsMod;
      tsMod.fromFormattedIso8601Str( szValue );
      ipNew.setTimeStamp( tsMod );
    }
    if ( ( szValue = getAttributeValue( "version", m_pxeRoot ) ).size() > 0 )
      ipNew.setVersion( szValue );

    // A special case is a BLOB enable message - it has no elements,
    // but has data in it.
    if ( tType == EnableBLOB )
    {
      ipNew = IndiProperty::getBLOBEnableType( ::pcdataXMLEle( m_pxeRoot ) );
    }
    else
    {
      // Now iterate over all the child elements and add them to the message.
      // The first time we call 'nextXMLEle' nInit is 1 to start fresh,
      // then it is 0 and will step through each one to the end.
      int nInit = 1;
      XMLEle *pxeChild = NULL;

      // We need to iterate over each child element.
      while ( pxeChild != NULL || nInit == 1 )
      {
        pxeChild = ::nextXMLEle( m_pxeRoot, nInit );

        // No children = we reached the end of the child elements.
        if ( pxeChild != NULL )
        {
          // We need a new element....
          IndiElement ieNew;

          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "format" ) ) ).size() > 0 )
            ieNew.setFormat( szValue );
          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "label" ) ) ).size() > 0 )
            ieNew.setLabel( szValue );
          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "max" ) ) ).size() > 0 )
            ieNew.setMax( szValue );
          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "min" ) ) ).size() > 0 )
            ieNew.setMin( szValue );
          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "name" ) ) ).size() > 0 )
            ieNew.setName( szValue );
          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "size" ) ) ).size() > 0 )
            ieNew.setSize( szValue );
          if ( ( szValue = string( ::findXMLAttValu( pxeChild, "step" ) ) ).size() > 0 )
            ieNew.setStep( szValue );

          szValue = string( ::pcdataXMLEle( pxeChild ) );

          // The different types have different data...
          switch ( tType )
          {
            case DefLightVector:
            case SetLightVector:
              ieNew.setLightState( IndiElement::getLightStateType( szValue ) );
              break;
            case DefSwitchVector:
            case NewSwitchVector:
            case SetSwitchVector:
              ieNew.setSwitchState( IndiElement::getSwitchStateType( szValue ) );
              break;
            default:
              ieNew.setValue( szValue );
          }

          // Now add this element to the message.
          ipNew.add( ieNew );
        }
        nInit = 0;
      }
    }
  }

  return IndiMessage( tMsgType, ipNew );
}

////////////////////////////////////////////////////////////////////////////////
/// This handles streaming from input and other stream-like constructs.

istream &operator>>( istream &strmIn, IndiXmlParser &ixpRhs )
{
  if ( strmIn.good() )
  {
    stringbuf sbData;
    strmIn.get( sbData );
    // We need a string for the error message, if any.
    string szErrorMsg;
    ixpRhs.parseXml( sbData.str(), szErrorMsg );
  }
  return strmIn;
}

////////////////////////////////////////////////////////////////////////////////
/// This handles streaming to output and other stream-like constructs.

ostream &operator<<( ostream &strmOut, const IndiXmlParser &ixpRhs )
{
  strmOut << ixpRhs.createXmlString();
  return strmOut;
}

////////////////////////////////////////////////////////////////////////////////
// If this xml contains a complete INDI message, this returns true,
// and start and stop will be the indices of the start and stop of it.
/*
IndiMessage::Type IndiXmlParser::extractMessage( const std::string &szXml,
                                                 int &nStart,
                                                 int &nStop )
{
  string szOpenTag;
  string szCloseTag;
  size_t tBegin = string::npos;
  size_t tEnd = string::npos;

  // Do we have an open and close for any INDI Message type?
  szOpenTag = "defBLOBVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/defBLOBVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::DefBLOBVector;
    }
  }
  szOpenTag = "defLightVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/defLightVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::DefLightVector;
    }
  }
  szOpenTag = "defNumberVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/defNumberVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::DefNumberVector;
    }
  }
  szOpenTag = "defSwitchVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/defSwitchVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::DefSwitchVector;
    }
  }
  szOpenTag = "defTextVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/defTextVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::DefTextVector;
    }
  }
  szOpenTag = "delProperty";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/delProperty";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::DelProperty;
    }
  }
  szOpenTag = "enableBLOB";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/enableBLOB";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::EnableBLOB;
    }
  }
  szOpenTag = "getProperties";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/getProperties";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::GetProperties;
    }
  }
  szOpenTag = "message";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/message";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::Message;
    }
  }
  szOpenTag = "newBLOBVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/newBLOBVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::NewBLOBVector;
    }
  }
  szOpenTag = "newNumberVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/newNumberVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::NewNumberVector;
    }
  }
  szOpenTag = "newSwitchVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/newSwitchVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::NewSwitchVector;
    }
  }
  szOpenTag = "newTextVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/newTextVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::NewTextVector;
    }
  }
  szOpenTag = "setBLOBVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/setBLOBVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::SetBLOBVector;
    }
  }
  szOpenTag = "setLightVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/setLightVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::SetLightVector;
    }
  }
  szOpenTag = "setNumberVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/setNumberVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::SetNumberVector;
    }
  }
  szOpenTag = "setSwitchVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/setSwitchVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::SetSwitchVector;
    }
  }
  szOpenTag = "setTextVector";
  if ( tBegin = szXml.find( szOpenTag ) != string::npos )
  {
    // Find the first close after the open, above.
    szCloseTag = "/setTextVector";
    if ( tEnd = szXml.find( szCloseTag, szOpenTag.size() ) != string::npos )
    {
      nStart = (int)( tBegin );
      nStop = (int)( tEnd ) + szCloseTag.size();
      return IndiMessage::SetTextVector;
    }
  }

  // If we got here, we do not have a valid XML document (yet).
  return IndiMessage::UnknownType;
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiXmlParser::convertTypeToString( const IndiXmlParser::MessageType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case UnknownType:
      szType = "";
      break;

    // Define properties.
    case DefBLOBVector:
      szType = "defBLOBVector";
      break;
    case DefLightVector:
      szType = "defLightVector";
      break;
    case DefNumberVector:
      szType = "defNumberVector";
      break;
    case DefSwitchVector:
      szType = "defSwitchVector";
      break;
    case DefTextVector:
      szType = "defTextVector";
      break;

    // Delete properties.
    case DelProperty:
      szType = "delProperty";
      break;

    // Enable blobs for a client.
    case EnableBLOB:
      szType = "enableBLOB";
      break;

    // Command to enable snooping messages from other devices.
    case GetProperties:
      szType = "getProperties";
      break;

    // A message.
    case Message:
      szType = "message";
      break;

    // Update properties.
    case NewBLOBVector:
      szType = "newBLOBVector";
      break;
    case NewNumberVector:
      szType = "newNumberVector";
      break;
    case NewSwitchVector:
      szType = "newSwitchVector";
      break;
    case NewTextVector:
      szType = "newTextVector";
      break;

    // Set properties.
    case SetBLOBVector:
      szType = "setBLOBVector";
      break;
    case SetLightVector:
      szType = "setLightVector";
      break;
    case SetNumberVector:
      szType = "setNumberVector";
      break;
    case SetSwitchVector:
      szType = "setSwitchVector";
      break;
    case SetTextVector:
      szType = "setTextVector";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the protocol version of this software.

void IndiXmlParser::setProtocolVersion( const string &szProtocolVersion )
{
  m_szProtocolVersion = szProtocolVersion;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the protocol version of this software.

string IndiXmlParser::getProtocolVersion() const
{
  return m_szProtocolVersion;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the tag.

IndiXmlParser::MessageType IndiXmlParser::convertStringToType( const string &szTag )
{
  MessageType tType = UnknownType;

  // Define properties.
  if ( szTag == "defBLOBVector" )
    tType = DefBLOBVector;
  else if ( szTag == "defLightVector" )
    tType = DefLightVector;
  else if ( szTag == "defNumberVector" )
    tType = DefNumberVector;
  else if ( szTag == "defSwitchVector" )
    tType = DefSwitchVector;
  else if ( szTag == "defTextVector" )
    tType = DefTextVector;

  // Delete properties.
  else if ( szTag == "delProperty" )
    tType = DelProperty;

  // Enable blobs for a client.
  else if ( szTag == "enableBLOB" )
    tType = EnableBLOB;

  // Command to enable snooping messages from other devices.
  else if ( szTag == "getProperties" )
    tType = GetProperties;

  // A message.
  else if ( szTag == "message" )
    tType = Message;

  // Update properties.
  else if ( szTag == "newBLOBVector" )
    tType = NewBLOBVector;
  else if ( szTag == "newNumberVector" )
    tType = NewNumberVector;
  else if ( szTag == "newSwitchVector" )
    tType = NewSwitchVector;
  else if ( szTag == "newTextVector" )
    tType = NewTextVector;

  // Set properties.
  else if ( szTag == "setBLOBVector" )
    tType = SetBLOBVector;
  else if ( szTag == "setLightVector" )
    tType = SetLightVector;
  else if ( szTag == "setNumberVector" )
    tType = SetNumberVector;
  else if ( szTag == "setSwitchVector" )
    tType = SetSwitchVector;
  else if ( szTag == "setTextVector" )
    tType = SetTextVector;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
