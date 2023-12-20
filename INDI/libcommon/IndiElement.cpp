/// IndiElement.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <sstream>
#include <iostream>           // for std::cerr
#include <stdexcept>          // for std::runtime_error
#include "IndiElement.hpp"

using std::boolalpha;
using std::runtime_error;
using std::string;
using std::stringstream;
using pcf::IndiElement;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

IndiElement::IndiElement()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name - this will be used often.

IndiElement::IndiElement( const string &szName ) : m_szName(szName)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name and a string value.

IndiElement::IndiElement( const string &szName,
                          const string &szValue ) : m_szName(szName), m_szValue(szValue)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name and a char pointer value.

IndiElement::IndiElement( const string &szName,
                          const char *pcValue ) : m_szName(szName), m_szValue(pcValue)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name and a LightStateType value.

IndiElement::IndiElement( const string &szName,
                          const LightStateType &tValue ) : m_szName(szName), m_lsValue(tValue)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name and a SwitchStateType value.

IndiElement::IndiElement( const string &szName,
                          const SwitchStateType &tValue ) : m_szName(szName), m_ssValue(tValue)
{
}

////////////////////////////////////////////////////////////////////////////////
///  Copy constructor.

IndiElement::IndiElement( const IndiElement &ieRhs ) : m_szFormat(ieRhs.m_szFormat), m_szLabel(ieRhs.m_szLabel),
                                                         m_szMax(ieRhs.m_szMax), m_szMin(ieRhs.m_szMin), m_szName(ieRhs.m_szName),
                                                          m_szSize(ieRhs.m_szSize), m_szStep(ieRhs.m_szStep), m_szValue(ieRhs.m_szValue),
                                                           m_lsValue(ieRhs.m_lsValue), m_ssValue(ieRhs.m_ssValue)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

IndiElement::~IndiElement()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Assigns the internal data of this object from an existing one.

const IndiElement &IndiElement::operator=( const IndiElement &ieRhs )
{
  if ( &ieRhs != this )
  {
    pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

    m_szFormat = ieRhs.m_szFormat;
    m_szLabel = ieRhs.m_szLabel;
    m_szMax = ieRhs.m_szMax;
    m_szMin = ieRhs.m_szMin;
    m_szName = ieRhs.m_szName;
    m_szSize = ieRhs.m_szSize;
    m_szStep = ieRhs.m_szStep;
    m_szValue = ieRhs.m_szValue;
    m_lsValue = ieRhs.m_lsValue;
    m_ssValue = ieRhs.m_ssValue;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if we have an exact match (value as well).

bool IndiElement::operator==( const IndiElement &ieRhs ) const
{
  if ( &ieRhs == this )
    return true;

  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  return ( m_szFormat == ieRhs.m_szFormat &&
           m_szLabel == ieRhs.m_szLabel &&
           m_szMax == ieRhs.m_szMax &&
           m_szMin == ieRhs.m_szMin &&
           m_szName == ieRhs.m_szName &&
           m_szSize == ieRhs.m_szSize &&
           m_szStep == ieRhs.m_szStep &&
           m_szValue == ieRhs.m_szValue &&
           m_lsValue == ieRhs.m_lsValue &&
           m_ssValue == ieRhs.m_ssValue );
}

////////////////////////////////////////////////////////////////////////////////

string IndiElement::createString() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  stringstream ssOutput;
  ssOutput << "{ "
           << "\"name\" : \"" << m_szName << "\" , "
           << "\"value\" : \"" << m_szValue << "\" , "
           << "\"lightstate\" : \"" << getLightStateString( m_lsValue ) << "\" , "
           << "\"switchstate\" : \"" << getSwitchStateString( m_ssValue ) << "\" , "
           << "\"label\" : \"" << m_szLabel << "\" , "
           << "\"format\" : \"" << m_szFormat << "\" , "
           << "\"max\" : \"" << m_szMax << "\" , "
           << "\"min\" : \"" << m_szMin << "\" , "
           << "\"size\" : \"" << m_szSize << "\" , "
           << "\"step\" : \"" << m_szStep << "\" "
           << "} ";
  return ssOutput.str();
}


////////////////////////////////////////////////////////////////////////////////
/// Remove all attributes and reset this object.

void IndiElement::clear()
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  m_szFormat = "%g";
  m_szLabel = "";
  m_szMax = "0";
  m_szMin = "0";
  m_szName = "";
  m_szSize = "0";
  m_szStep = "0";
  m_szValue = "";
  m_lsValue = UnknownLightState;
  m_ssValue = UnknownSwitchState;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the format attribute.

const string &IndiElement::getFormat() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szFormat;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the label attribute.

const string &IndiElement::getLabel() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szLabel;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of a light.

IndiElement::operator IndiElement::LightStateType() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_lsValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of a light.

IndiElement::LightStateType IndiElement::getLightState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_lsValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the max attribute.

const string &IndiElement::getMax() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szMax;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the min attribute.

const string &IndiElement::getMin() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szMin;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the name attribute.

const string &IndiElement::getName() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szName;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the step attribute.

const string &IndiElement::getStep() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szStep;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the size attribute.

const std::string & IndiElement::getSize() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of a switch.

IndiElement::operator IndiElement::SwitchStateType() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_ssValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of a switch.

IndiElement::SwitchStateType IndiElement::getSwitchState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_ssValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the value (not LightState or SwitchState) a numeric?

bool IndiElement::isNumeric() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  int iValue;
  std::stringstream ssValue( m_szValue );

  // Try to stream the data into the int variable.
  // If we fail, this value is not numeric.
  ssValue >> iValue;
  return ssValue.good();
//  return ( ssValue >> iValue );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the bool value. If the value is 0, "false", or any other non-value.
/*
IndiElement::operator bool() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  bool oValue;
  std::stringstream ssValue( m_szValue );

  // Try to stream the data into the variable.
  ssValue >> oValue;
  if ( ssValue.fail() == true )
  {
    throw ( runtime_error( string( "IndiElement '" ) + m_szName + "' value '" +
                           m_szValue + "' is not a bool.") );
  }

  return oValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the double value.

IndiElement::operator double() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  double xValue;
  std::stringstream ssValue( m_szValue );

  // Try to stream the data into the variable.
  ssValue >> xValue;
  if ( ssValue.fail() == true )
  {
    throw ( runtime_error( string( "IndiElement '" ) + m_szName + "' value '" +
                           m_szValue + "' is not a double.") );
  }

  return xValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the float value.

IndiElement::operator float() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  float eValue;
  std::stringstream ssValue( m_szValue );

  // Try to stream the data into the variable.
  ssValue >> eValue;
  if ( ssValue.fail() == true )
  {
    throw ( runtime_error( string( "IndiElement '" ) + m_szName + "' value '" +
                           m_szValue + "' is not a float.") );
  }

  return eValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the int value.

IndiElement::operator int() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  int iValue;
  std::stringstream ssValue( m_szValue );

  // Try to stream the data into the variable.
  ssValue >> iValue;
  if ( ssValue.fail() == true )
  {
    throw ( runtime_error( string( "IndiElement '" ) + m_szName + "' value '" +
                           m_szValue + "' is not an int.") );
  }

  return iValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string value.

IndiElement::operator string() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the unsigned int value.

IndiElement::operator unsigned int() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  unsigned int uiValue;
  std::stringstream ssValue( m_szValue );

  // Try to stream the data into the variable.
  ssValue >> uiValue;
  if ( ssValue.fail() == true )
  {
    throw ( runtime_error( string( "IndiElement '" ) + m_szName + "' value '" +
                           m_szValue + "' is not an unsigned int.") );
  }

  return uiValue;
}
*/
////////////////////////////////////////////////////////////////////////////////
/// Returns the value as type string.

string IndiElement::get() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value as type string.

string IndiElement::getValue() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string value as an array of char. The size argument should
/// contain the number of bytes in the buffer 'pcValue', it will be returned
/// with the number of bytes actually copied.

void IndiElement::getValue( char *pcValue, unsigned int &uiSize ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  // Modify the number of bytes to copy. It will be the lesser of the two sizes.
  uiSize = ( uiSize > m_szValue.size() ) ? ( m_szValue.size() ) : ( uiSize );

  ::memcpy( pcValue, m_szValue.c_str(), uiSize );
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setFormat( const string &szFormat )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szFormat = szFormat;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setLabel( const string &szLabel )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szLabel = szLabel;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setMax( const string &szMax )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szMax = szMax;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setMin( const string &szMin )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szMin = szMin;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setName( const string &szName )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szName = szName;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setStep( const string &szStep )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szStep = szStep;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setSize( const string &szSize )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szSize = szSize;
}

////////////////////////////////////////////////////////////////////////////////
/// This is an alternate way of calling 'setLightState'.

const IndiElement::LightStateType &IndiElement::operator=( const LightStateType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_lsValue = tValue;
  return tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setLightState( const LightStateType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_lsValue = tValue;
}

////////////////////////////////////////////////////////////////////////////////
/// This is an alternate way of calling 'setSwitchState'.

const IndiElement::SwitchStateType &IndiElement::operator=( const SwitchStateType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_ssValue = tValue;
  return tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiElement::setSwitchState( const SwitchStateType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_ssValue = tValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the data contained in the value as a string.

void IndiElement::setValue( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szValue = szValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the data contained in the value as a string.

void IndiElement::setValue( const char *pcValue,
                            const unsigned int &uiSize )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szValue.assign( const_cast<char *>( pcValue ), uiSize );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidFormat() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szFormat.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidLabel() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szLabel.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidLightState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_lsValue != UnknownLightState );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidMax() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szMax.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidMin() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szMin.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidName() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szName.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidSize() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szSize.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidStep() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szStep.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidSwitchState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_ssValue != UnknownSwitchState );
}

////////////////////////////////////////////////////////////////////////////////

bool IndiElement::hasValidValue() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szValue.size() > 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiElement::LightStateType IndiElement::getLightStateType( const string &szType )
{
  LightStateType tType = UnknownLightState;

  if ( szType == "Idle" )
    tType = Idle;
  else if ( szType == "Ok" )
    tType = Ok;
  else if ( szType == "Busy" )
    tType = Busy;
  else if ( szType == "Alert" )
    tType = Alert;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiElement::getLightStateString( const LightStateType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case UnknownLightState:
      szType = "";
      break;
    case Idle:
      szType = "Idle";
      break;
    case Ok:
      szType = "Ok";
      break;
    case Busy:
      szType = "Busy";
      break;
    case Alert:
      szType = "Alert";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiElement::SwitchStateType IndiElement::getSwitchStateType( const string &szType )
{
  SwitchStateType tType = UnknownSwitchState;

  if ( szType == "Off" )
    tType = Off;
  else if ( szType == "On" )
    tType = On;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiElement::getSwitchStateString( const SwitchStateType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case UnknownSwitchState:
      szType = "";
      break;
    case Off:
      szType = "Off";
      break;
    case On:
      szType = "On";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiElement::convertTypeToString( const Type &tType )
{
  string szType = "UnknownType";

  switch ( tType )
  {
    case UnknownType:
      szType = "";
      break;

    // Define properties.
    case DefBLOB:
      szType = "defBLOB";
      break;
    case DefLight:
      szType = "defLight";
      break;
    case DefNumber:
      szType = "defNumber";
      break;
    case DefSwitch:
      szType = "defSwitch";
      break;
    case DefText:
      szType = "defText";
      break;

    // Update or set properties.
    case OneBLOB:
      szType = "oneBLOB";
      break;
    case OneLight:
      szType = "oneLight";
      break;
    case OneNumber:
      szType = "oneNumber";
      break;
    case OneSwitch:
      szType = "oneSwitch";
      break;
    case OneText:
      szType = "oneText";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the tag.

IndiElement::Type IndiElement::convertStringToType( const string &szTag )
{
  Type tType = UnknownType;

  // Define properties.
  if ( szTag == "defBLOB" )
    tType = DefBLOB;
  else if ( szTag == "defLight" )
    tType = DefLight;
  else if ( szTag == "defNumber" )
    tType = DefNumber;
  else if ( szTag == "defSwitch" )
    tType = DefSwitch;
  else if ( szTag == "defText" )
    tType = DefText;

  // Update or set properties.
  else if ( szTag == "oneBLOB" )
    tType = OneBLOB;
  else if ( szTag == "oneLight" )
    tType = OneLight;
  else if ( szTag == "oneNumber" )
    tType = OneNumber;
  else if ( szTag == "oneSwitch" )
    tType = OneSwitch;
  else if ( szTag == "oneText" )
    tType = OneText;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
