/** \file IndiXmlParser.hpp
 * \brief Parse INDI XML messages and serialize property publications.
 */
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
/// Converts between INDI XML documents and IndiMessage objects.
/** Instances are not thread-safe; use one parser per thread or message. Serialization of independent
 * instances may run concurrently.
 */
class IndiXmlParser
{
  private:
    /// INDI XML document types this parser recognizes.
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

    /// Fixed buffer sizes.
    enum Constants
    {
      MaxErrorMsgSize = 256,
    };

  public:
    /// Whether a complete XML document has been parsed.
    enum State
    {
      UnknownState = 0,
      IncompleteState,
      CompleteState
    };

  // Constructor/copy constructor/destructor.
  public:
    /// Construct an empty parser ready to receive XML.
    IndiXmlParser( const std::string &szProtocolVersion = "1" /**< [in] INDI protocol version for generated XML. */ );

    /// Construct the XML for an outgoing message.
    /** \throws std::runtime_error if the message lacks a required attribute or has an unknown type.
     */
    IndiXmlParser( const pcf::IndiMessage &imSend /**< [in] Message to serialize. */,
                   const std::string &szProtocolVersion = "1" /**< [in] INDI protocol version for generated XML. */ );

    /// Copying is disabled: lilxml cannot clone a parser's in-progress state.
    IndiXmlParser( const IndiXmlParser &ixpRhs /**< [in] Unused. */ ) = delete;

    /// Destructor.
    virtual ~IndiXmlParser();

  // Operators.
  public:
    /// Assignment is disabled: lilxml cannot clone a parser's in-progress state.
    const IndiXmlParser &operator= ( const IndiXmlParser &ixpRhs /**< [in] Unused. */ ) = delete;

  // Methods.
  public:
    /// Reset this object.
    void clear();

    /// Returns the data formatted as an xml string.
    std::string createXmlString() const;

    /// Get the IndiMessage created from the xml data.
    pcf::IndiMessage createIndiMessage() const;

    /// Return the INDI protocol version.
    std::string getProtocolVersion() const;

    /// Gets the state, showing whether the XML document is complete.
    State getState() const;

    /// Parse the next complete XML document from new text plus any left over from the previous call.
    /** Discards the previously parsed document. Text after a completed document is kept for the next
     * call, so call again with an empty string until getState() is no longer CompleteState.
     */
    void parseXml( const std::string &szXml /**< [in] Newly received XML text. */,
                   std::string &szErrorMsg /**< [out] Parser error, or empty on success. */ );

    /// Parse the next complete XML document from a byte buffer.
    void parseXml( const char *pcXml /**< [in] Newly received XML bytes. */,
                   const unsigned int &uiNumBytes /**< [in] Number of bytes in pcXml. */,
                   std::string &szErrorMsg /**< [out] Parser error, or empty on success. */ );

    /// Set the version of the INDI protocol.
    void setProtocolVersion( const std::string &szProtocolVersion /**< [in] New INDI protocol version. */ );

    /// Escape '&', '<', '>', '\'' and '"' as &amp; &lt; &gt; &apos; and &quot;.
    /** Stops at the first embedded null character. Thread-safe.
     * \returns the escaped text.
     */
    static std::string createSafeXmlString( const std::string &szText /**< [in] Text to escape. */ );

  // Helper functions.
  private:
    /// Returns the string type given the enumerated type.
    static std::string convertTypeToString( const MessageType &tType /**< [in] Message type. */ );

    /// Returns the enumerated type given the tag.
    static MessageType convertStringToType( const std::string &szTag /**< [in] XML element tag. */ );

    // Commands
    /// Build a delProperty command.
    void createDelProperty( const pcf::IndiProperty &ip /**< [in] Property to delete. */ );

    /// Build an enableBLOB command.
    void createEnableBLOB( const pcf::IndiProperty &ip /**< [in] Property whose BLOB policy is sent. */ );

    /// Build a getProperties command.
    void createGetProperties( const pcf::IndiProperty &ip /**< [in] Property selecting the request scope. */ );

    // Messages
    /// Build a message command.
    void createMessage( const pcf::IndiProperty &ip /**< [in] Property carrying the device and message. */ );

    // Definitions
    /// Build a defBLOBVector document.
    void createDefBLOBVector( const pcf::IndiProperty &ip /**< [in] Property to define. */ );

    /// Build a defLightVector document.
    void createDefLightVector( const pcf::IndiProperty &ip /**< [in] Property to define. */ );

    /// Build a defNumberVector document.
    void createDefNumberVector( const pcf::IndiProperty &ip /**< [in] Property to define. */ );

    /// Build a defSwitchVector document.
    void createDefSwitchVector( const pcf::IndiProperty &ip /**< [in] Property to define. */ );

    /// Build a defTextVector document.
    void createDefTextVector( const pcf::IndiProperty &ip /**< [in] Property to define. */ );

    // Updates
    /// Build a newBLOBVector document.
    void createNewBLOBVector( const pcf::IndiProperty &ip /**< [in] Property with requested values. */ );

    /// Build a newNumberVector document.
    void createNewNumberVector( const pcf::IndiProperty &ip /**< [in] Property with requested values. */ );

    /// Build a newSwitchVector document.
    void createNewSwitchVector( const pcf::IndiProperty &ip /**< [in] Property with requested values. */ );

    /// Build a newTextVector document.
    void createNewTextVector( const pcf::IndiProperty &ip /**< [in] Property with requested values. */ );

    // Assignments
    /// Build a setBLOBVector document.
    void createSetBLOBVector( const pcf::IndiProperty &ip /**< [in] Property with current values. */ );

    /// Build a setLightVector document.
    void createSetLightVector( const pcf::IndiProperty &ip /**< [in] Property with current values. */ );

    /// Build a setNumberVector document.
    void createSetNumberVector( const pcf::IndiProperty &ip /**< [in] Property with current values. */ );

    /// Build a setSwitchVector document.
    void createSetSwitchVector( const pcf::IndiProperty &ip /**< [in] Property with current values. */ );

    /// Build a setTextVector document.
    void createSetTextVector( const pcf::IndiProperty &ip /**< [in] Property with current values. */ );

    /// Returns an attribute value, or an empty string if the element or attribute is missing.
    std::string getAttributeValue( const std::string &szAttributeName /**< [in] Attribute to look up. */,
                                   XMLEle *pxeRoot /**< [in] Element holding the attribute. */ ) const;

    /// Initialize all data for a newly constructed parser.
    void init( const std::string &szProtocolVersion = "1" /**< [in] INDI protocol version for generated XML. */ );

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

/// Feed stream text up to the next newline to the parser.
std::istream &operator>> ( std::istream &strmIn /**< [in] Stream holding XML text. */,
                           pcf::IndiXmlParser &rhs /**< [in,out] Parser receiving the text. */ );

/// Write the parser's current XML document to a stream.
std::ostream &operator<< ( std::ostream &strmOut /**< [in,out] Destination stream. */,
                           const pcf::IndiXmlParser &rhs /**< [in] Parser to serialize. */ );

////////////////////////////////////////////////////////////////////////////////

#endif // INDI_XML_PARSER_HPP
