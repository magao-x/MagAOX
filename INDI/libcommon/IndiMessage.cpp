/// IndiMessage.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include "IndiMessage.hpp"

using std::string;
using pcf::IndiProperty;
using pcf::IndiMessage;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

IndiMessage::IndiMessage()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with type and property - this will be used often.

IndiMessage::IndiMessage( const Type &tType, const IndiProperty &ipMsg ) : m_ipMsg(ipMsg), m_tType(tType)
{
}

////////////////////////////////////////////////////////////////////////////////
///  Copy constructor.

IndiMessage::IndiMessage( const IndiMessage &imRhs ) : m_ipMsg(imRhs.m_ipMsg), m_tType(imRhs.m_tType)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

IndiMessage::~IndiMessage()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Assigns the internal data of this object from an existing one.

const IndiMessage &IndiMessage::operator=( const IndiMessage &imRhs )
{
  if ( &imRhs != this )
  {
    m_ipMsg = imRhs.m_ipMsg;
    m_tType = imRhs.m_tType;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the message type.

const IndiMessage::Type &IndiMessage::getType() const
{
  return m_tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the Indi property contained here.

const IndiProperty &IndiMessage::getProperty() const
{
  return m_ipMsg;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the Indi property contained here.

IndiProperty &IndiMessage::getProperty()
{
  return m_ipMsg;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the Indi property contained here.

void IndiMessage::setProperty( const IndiProperty &ipMsg )
{
  m_ipMsg = ipMsg;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiMessage::convertTypeToString( const IndiMessage::Type &tType )
{
  string szType = "";

  switch ( tType )
  {
    case Unknown:
      szType = "Unknown"; break;
    // Define properties.
    case Define:
      szType = "Define"; break;
    // Delete properties.
    case Delete:
      szType = "Delete"; break;
    // Enable blobs for a client.
    case EnableBLOB:
      szType = "EnableBLOB"; break;
    // Command to enable snooping messages from other devices.
    case GetProperties:
      szType = "GetProperties"; break;
    // A message.
    case Message:
      szType = "Message"; break;
    // Update properties.
    case NewProperty:
      szType = "NewProperty"; break;
    // Set properties.
    case SetProperty:
      szType = "SetProperty"; break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the tag.

IndiMessage::Type IndiMessage::convertStringToType( const string &szTag )
{
  Type tType = Unknown;

  if ( szTag == "Define" )
    tType = Define;
  else if ( szTag == "Delete" )
    tType = Delete;
  else if ( szTag == "EnableBLOB" )
    tType = EnableBLOB;
  else if ( szTag == "GetProperties" )
    tType = GetProperties;
  else if ( szTag == "Message" )
    tType = Message;
  else if ( szTag == "NewProperty" )
    tType = NewProperty;
  else if ( szTag == "SetProperty" )
    tType = SetProperty;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this is a 'def' (define) message, false otherwise.
/*
bool IndiMessage::isDefType() const
{
  return ( m_tType == Define );
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Returns true if this is a 'new' (update) message, false otherwise.
/*
bool IndiMessage::isNewType() const
{
  return ( m_tType == NewProperty );
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Returns true if this is a 'set' (assign) message, false otherwise.
/*
bool IndiMessage::isSetType() const
{
  return ( m_tType == SetProperty );
}
*/
////////////////////////////////////////////////////////////////////////////////


