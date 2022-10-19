/// IndiXmlParser.hpp
///
/// @author Paul Grenz
////////////////////////////////////////////////////////////////////////////////

#ifndef INDI_XML_PARSER_HPP
#define INDI_XML_PARSER_HPP
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include "../liblilxml/lilxml.h"
#include "IndiProperty.hpp"
#include "IndiMessage.hpp"

namespace pcf
{
class IndiXmlParser
{
  private:
    enum MessageType
    {
      UnknownType = 0,
      // Define properties.
      DefBLOBVector,
      DefLightVector,
      DefNumberVector,
      DefSwitchVector,
      DefTextVector,
      // Delete properties.
      DelProperty,
      // Enable blobs for a client.
      EnableBLOB,
      // Command to enable snooping messages from other devices.
      GetProperties,
      // A message.
      Message,
      // Update properties.
      NewBLOBVector,
      NewNumberVector,
      NewSwitchVector,
      NewTextVector,
      // Set properties.
      SetBLOBVector,
      SetLightVector,
      SetNumberVector,
      SetSwitchVector,
      SetTextVector,
    };

    enum Constants
    {
      MaxErrorMsgSize = 256,
    };

  public:
    enum State
    {
      UnknownState = 0,
      IncompleteState,
      CompleteState
    };

  // Constructor/copy constructor/destructor.
  public:
    /// Constructor.
    IndiXmlParser( const std::string &szProtocolVersion = "1" );
    /// Constructor from an existing IndiMessage.
    IndiXmlParser( const pcf::IndiMessage &imSend,
                   const std::string &szProtocolVersion = "1" );
    /// Copy constructor.
    IndiXmlParser( const IndiXmlParser &ixpRhs );
    /// Destructor.
    virtual ~IndiXmlParser();

  // Operators.
  public:
    /// Assigns the internal data of this object from an existing one.
    const IndiXmlParser &operator= ( const IndiXmlParser &ixpRhs );

  // Methods.
  public:
    /// Reset this object.
    void clear();
    /// Returns the data formatted as an xml string.
    std::string createXmlString() const;
    /// Get the IndiMessage created from the xml data.
    pcf::IndiMessage createIndiMessage() const;
    // Return the INDI protocol version.
    std::string getProtocolVersion() const;
    /// Gets the state, showing whether the XML document is complete.
    State getState() const;
    /// Sets all the data in this object from formatted xml. First clears all
    /// internal data, then parses the xml. May throw.
    void parseXml( const std::string &szXml,
                   std::string &szErrorMsg );
    /// Sets all the data in this object from formatted xml. First clears all
    /// internal data, then parses the xml. May throw.
    void parseXml( const char *pcXml,
                   const unsigned int &uiNumBytes,
                   std::string &szErrorMsg );
    // Set the version of the INDI protocol.
    void setProtocolVersion( const std::string &szProtocolVersion );
    /// This takes any string and replaces the '&', '<', '>', '\', and '"'
    /// with its equivalent XML entity: &amp; &lt; &gt; &apos; and &quot;
    static std::string createSafeXmlString( const std::string &szText );

  // Helper functions.
  private:
    /// Returns the string type given the enumerated type.
    static std::string convertTypeToString( const MessageType &tType );
    /// Returns the enumerated type given the tag.
    static MessageType convertStringToType( const std::string &szTag );
    // Commands
    void createDelProperty( const pcf::IndiProperty &ip );
    void createEnableBLOB( const pcf::IndiProperty &ip );
    void createGetProperties( const pcf::IndiProperty &ip );
    // Messages
    void createMessage( const pcf::IndiProperty &ip );
    // Definitions
    void createDefBLOBVector( const pcf::IndiProperty &ip );
    void createDefLightVector( const pcf::IndiProperty &ip );
    void createDefNumberVector( const pcf::IndiProperty &ip );
    void createDefSwitchVector( const pcf::IndiProperty &ip );
    void createDefTextVector( const pcf::IndiProperty &ip );
    // Updates
    void createNewBLOBVector( const pcf::IndiProperty &ip );
    void createNewNumberVector( const pcf::IndiProperty &ip );
    void createNewSwitchVector( const pcf::IndiProperty &ip );
    void createNewTextVector( const pcf::IndiProperty &ip );
    // Assignments
    void createSetBLOBVector( const pcf::IndiProperty &ip );
    void createSetLightVector( const pcf::IndiProperty &ip );
    void createSetNumberVector( const pcf::IndiProperty &ip );
    void createSetSwitchVector( const pcf::IndiProperty &ip );
    void createSetTextVector( const pcf::IndiProperty &ip );
    /// Returns the string attribute type.
    std::string getAttributeValue( const std::string &szAttributeName,
                                   XMLEle *pxeRoot ) const;
    void init( const std::string &szProtocolVersion = "1" );

  // Members.
  private:
    /// A pointer to the entire data structure which holds the xml data.
    /// This is used to parse XML data.
    mutable LilXML *m_plxMessage;
    /// This structure holds the actual data structure for the XML.
    mutable XMLEle *m_pxeRoot;
    /// Used to hold the error message which is used in the lilxml calls.
    char m_pcMessage[MaxErrorMsgSize];
    /// This is left-over text from the previous call that is part of the
    /// next xml document.
    std::string m_szPrevXml;
    /// This is constructed XML stored in a stringstream.
    std::stringstream m_ssXml;
    /// This is the INDI protocol version.
    std::string m_szProtocolVersion;

}; // class IndiXmlParser
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

///  this handles streaming from input and other stream-like constructs.
std::istream &operator>> ( std::istream &strmIn, pcf::IndiXmlParser &rhs );
///  this handles streaming to output and other stream-like constructs.
std::ostream &operator<< ( std::ostream &strmOut, const pcf::IndiXmlParser &rhs );

////////////////////////////////////////////////////////////////////////////////

#endif // INDI_XML_PARSER_HPP
