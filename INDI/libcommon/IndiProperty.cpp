/// IndiProperty.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include "IndiProperty.hpp"

using std::runtime_error;
using std::string;
using std::stringstream;
using std::map;
using pcf::TimeStamp;
using pcf::IndiElement;
using pcf::IndiProperty;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

IndiProperty::IndiProperty()
{
  //m_tPerm = UnknownPropertyPerm;
  //m_oRequested = false;
  //m_tRule = UnknownSwitchRule;
  //m_tState = UnknownPropertyState;
  //m_xTimeout = 0.0f;
  //m_beValue = UnknownBLOBEnable;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with type - this will be used often.

IndiProperty::IndiProperty( const Type &tType ) : m_tType(tType)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with type, device and name - this will be used often.

IndiProperty::IndiProperty( const Type &tType,
                            const string &szDevice,
                            const string &szName ) :  m_szDevice(szDevice), m_szName(szName), m_tType(tType)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with type, device, name, state, and perm.

IndiProperty::IndiProperty( const Type &tType,
                            const string &szDevice,
                            const string &szName,
                            const PropertyStateType &tState,
                            const PropertyPermType &tPerm,
                            const SwitchRuleType &tRule ) : m_szDevice(szDevice), m_szName(szName), m_tPerm(tPerm),
                                                               m_tRule(tRule), m_tState(tState), m_tType(tType)
{
}

////////////////////////////////////////////////////////////////////////////////
///  Copy constructor.

IndiProperty::IndiProperty(const IndiProperty &ipRhs ) : m_szDevice(ipRhs.m_szDevice), m_szGroup(ipRhs.m_szGroup), m_szLabel(ipRhs.m_szLabel),
                                                           m_szMessage(ipRhs.m_szMessage), m_szName(ipRhs.m_szName), m_tPerm(ipRhs.m_tPerm),
                                                            m_tRule(ipRhs.m_tRule), m_tState(ipRhs.m_tState), m_xTimeout(ipRhs.m_xTimeout),
                                                              m_oRequested(ipRhs.m_oRequested),  m_tsTimeStamp(ipRhs.m_tsTimeStamp),
                                                                m_szVersion(ipRhs.m_szVersion), m_beValue(ipRhs.m_beValue), 
                                                                 m_mapElements(ipRhs.m_mapElements), m_tType(ipRhs.m_tType)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

IndiProperty::~IndiProperty()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Assigns the internal data of this object from an existing one.

const IndiProperty &IndiProperty::operator=( const IndiProperty &ipRhs )
{
  if ( &ipRhs != this )
  {
    pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

    m_szDevice = ipRhs.m_szDevice;
    m_szGroup = ipRhs.m_szGroup;
    m_szLabel = ipRhs.m_szLabel;
    m_szMessage = ipRhs.m_szMessage;
    m_szName = ipRhs.m_szName;
    m_tPerm = ipRhs.m_tPerm;
    m_oRequested = ipRhs.m_oRequested;
    m_tRule = ipRhs.m_tRule;
    m_tState = ipRhs.m_tState;
    m_xTimeout = ipRhs.m_xTimeout;
    m_tsTimeStamp = ipRhs.m_tsTimeStamp;
    m_szVersion = ipRhs.m_szVersion;
    m_beValue = ipRhs.m_beValue;

    m_mapElements = ipRhs.m_mapElements;
    m_tType = ipRhs.m_tType;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// This is an alternate way of calling 'setBLOBEnable'.

const IndiProperty::BLOBEnableType &IndiProperty::operator=( const BLOBEnableType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_beValue = tValue;
  return tValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if we have an exact match (value as well).

bool IndiProperty::operator==( const IndiProperty &ipRhs ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  // If we are comparing ourself to ourself - easy!
  if ( &ipRhs == this )
    return true;

  // If they are different sizes, they are different.
  if ( ipRhs.m_mapElements.size() != m_mapElements.size() )
    return false;

  // We need some iterators for each of the maps.
  map<string, IndiElement>::const_iterator itrRhs = ipRhs.m_mapElements.end();
  map<string, IndiElement>::const_iterator itr = m_mapElements.begin();
  for ( ; itr != m_mapElements.end(); ++itr )
  {
    // Can we find an element of the same name in the other map?
    itrRhs = ipRhs.m_mapElements.find( itr->first );

    // If we can't find the name, these are different.
    if ( itrRhs == ipRhs.m_mapElements.end() )
      return false;

    // If we found it, and they don't match, these are different.
    if ( !(itrRhs->second == itr->second ) )
      return false;
  }

  // Otherwise the maps are identical and it comes down to the
  // attributes here matching.
  return ( m_szDevice == ipRhs.m_szDevice &&
    m_szGroup == ipRhs.m_szGroup &&
    m_szLabel == ipRhs.m_szLabel &&
    m_szMessage == ipRhs.m_szMessage &&
    m_szName == ipRhs.m_szName &&
    m_tPerm == ipRhs.m_tPerm &&
    //m_oRequested == ipRhs.m_oRequested &&
    m_tRule == ipRhs.m_tRule &&
    m_tState == ipRhs.m_tState &&
    //m_xTimeout == ipRhs.m_xTimeout &&
    //m_tsTimeStamp ==ipRhs.m_tsTimeStamp &&  // Don't compare!
    m_szVersion == ipRhs.m_szVersion &&
    m_beValue == ipRhs.m_beValue &&
    m_tType == ipRhs.m_tType );
}

////////////////////////////////////////////////////////////////////////////////
/// Ensures that a name conforms to the INDI standard and can be used as
/// an identifier. This means:
///     1) No ' ' - these will be converted to '_'.
///     2) No '.' - these will be converted to '___'.
///     3) No Unprintable chars - these will be converted to '_'.

string IndiProperty::scrubName( const string &szName )
{
  string szScrubbed( szName );

  // We are replacing one char with multiple chars, so we have to do it
  // the long way first.
  size_t pp = 0;
  size_t rr = 0;
  while ( ( rr = szScrubbed.find( '.', pp ) ) != string::npos )
  {
    szScrubbed.replace( rr, 1, "___" );
    pp = rr + 1;
  }

  // These are one-for-one replacements, so we can do them in-place.
  std::replace_if( szScrubbed.begin(), szScrubbed.end(),
                   std::not1( std::ptr_fun( ::isalnum ) ), '_' );

  return szScrubbed;
}

////////////////////////////////////////////////////////////////////////////////
/// Compares one property with another instance. The values may not match,
/// but the type must match, as must the device and name. The names of all the
/// elements must match as well.

bool IndiProperty::compareProperty( const IndiProperty &ipComp ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  // If we are comparing ourself to ourself - easy!
  if ( &ipComp == this )
    return true;

  if ( ipComp.getType() != m_tType )
    return false;

  if ( ipComp.getDevice() != m_szDevice || ipComp.getName() != m_szName )
    return false;

  // If they are different sizes, they are different.
  if ( ipComp.m_mapElements.size() != m_mapElements.size() )
    return false;

  // We need some iterators for each of the maps.
  map<string, IndiElement>::const_iterator itrComp = ipComp.m_mapElements.end();
  map<string, IndiElement>::const_iterator itr = m_mapElements.begin();
  for ( ; itr != m_mapElements.end(); ++itr )
  {
    // Can we find an element of the same name in the other map?
    itrComp = ipComp.m_mapElements.find( itr->first );

    // If we can't find the name, these are different.
    if ( itrComp == ipComp.m_mapElements.end() )
      return false;
  }

  // If we got here, we are identical.
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Compares one element value contained in this class with another
/// instance. The type must match, as must the device and name.
/// The name of this element must match as well.

bool IndiProperty::compareValue( const IndiProperty &ipComp,
                                 const std::string &szElementName ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  // If we are comparing ourself to ourself - easy!
  if ( &ipComp == this )
    return true;

  if ( ipComp.getType() != m_tType )
    return false;

  if ( ipComp.getDevice() != m_szDevice || ipComp.getName() != m_szName )
    return false;

  // Can we find this element in this map? If not, we fail.
  map<string, IndiElement>::const_iterator itr =
      m_mapElements.find( szElementName );
  if ( itr == m_mapElements.end() )
    return false;

  // Can we find this element in the other map? If not, we fail.
  map<string, IndiElement>::const_iterator itrComp =
      ipComp.m_mapElements.find( szElementName );
  if ( itrComp == ipComp.m_mapElements.end() )
    return false;

  // If we found it, and the values don't match, these are different.
  if ( itr->second.getValue() != itrComp->second.getValue() )
    return false;

  // If we got here, we are identical.
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Compares all the element values contained in this class with another
/// instance. The type must match, as must the device and name. The number
/// and names of all the elements must match as well.

bool IndiProperty::compareValues( const IndiProperty &ipComp ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  // If we are comparing ourself to ourself - easy!
  if ( &ipComp == this )
    return true;

  if ( ipComp.getType() != m_tType )
    return false;

  if ( ipComp.getDevice() != m_szDevice || ipComp.getName() != m_szName )
    return false;

  // If they are different sizes, they are different.
  if ( ipComp.m_mapElements.size() != m_mapElements.size() )
    return false;

  // We need some iterators for each of the maps.
  map<string, IndiElement>::const_iterator itrComp = ipComp.m_mapElements.end();
  map<string, IndiElement>::const_iterator itr = m_mapElements.begin();
  for ( ; itr != m_mapElements.end(); ++itr )
  {
    // Can we find an element of the same name in the other map?
    itrComp = ipComp.m_mapElements.find( itr->first );

    // If we can't find the name, these are different.
    if ( itrComp == ipComp.m_mapElements.end() )
      return false;

    // If we found it, and the values don't match, these are different.
    if ( itrComp->second.getValue() != itr->second.getValue() )
      return false;
  }

  // If we got here, we are identical.
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Compares one element value contained in this class with another
/// instance. The type must match, as must the device and name.
/// The name of this element must match as well, and the value must be a new,
/// non-blank value.

bool IndiProperty::hasNewValue( const IndiProperty &ipComp,
                                const std::string &szElementName ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  // If we are comparing ourself to ourself - easy - there is no new value,
  // so we must return false.
  if ( &ipComp == this )
    return false;

  // The types don't match, so we can't be compared.
  if ( ipComp.getType() != m_tType )
    return false;

  if ( ipComp.getDevice() != m_szDevice || ipComp.getName() != m_szName )
    return false;

  // Can we find this element in this map? If not, we fail.
  map<string, IndiElement>::const_iterator itr =
      m_mapElements.find( szElementName );
  if ( itr == m_mapElements.end() )
    return false;

  // Can we find this element in the other map? If not, we fail.
  map<string, IndiElement>::const_iterator itrComp =
      ipComp.m_mapElements.find( szElementName );
  if ( itrComp == ipComp.m_mapElements.end() )
    return false;

  // If we found it, and the values don't match and the 'comp' is not
  // empty, the 'comp' is an update.
  if ( itr->second.getValue() != itrComp->second.getValue() &&
      itrComp->second.getValue().length() > 0 )
    return true;

  // If we got here, we are identical or the 'comp' is empty.
  return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a string with each attribute enumerated.

string IndiProperty::createString() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  stringstream ssOutput;
  ssOutput << "{ "
           << "\"device\" : \"" << m_szDevice << "\" , "
           << "\"name\" : \"" << m_szName << "\" , "
           << "\"type\" : \"" << convertTypeToString( m_tType ) << "\" , "
           << "\"group\" : \"" << m_szGroup << "\" , "
           << "\"label\" : \"" << m_szLabel << "\" , "
           << "\"timeout\" : \"" << m_xTimeout << "\" , "
           << "\"version\" : \"" << m_szVersion << "\" , "
           << "\"timestamp\" : \"" << m_tsTimeStamp.getFormattedIso8601Str() << "\" , "
           << "\"perm\" : \"" << getPropertyPermString( m_tPerm ) << "\" , "
           << "\"rule\" : \"" << getSwitchRuleString( m_tRule ) << "\" , "
           << "\"state\" : \"" << getPropertyStateString( m_tState ) << "\" , "
           << "\"BLOBenable\" : \"" << getBLOBEnableString( m_beValue ) << "\" , "
           << "\"message\" : \"" << m_szMessage << "\" "
           << "\"elements\" : [ \n";

  map<string, IndiElement>::const_iterator itr = m_mapElements.begin();
  for ( ; itr != m_mapElements.end(); ++itr )
  {
    ssOutput << "    ";
    if ( itr != m_mapElements.begin() )
      ssOutput << " , ";
    ssOutput << itr->second.createString();
  }

  ssOutput << "\n] "
           << " } ";

  return ssOutput.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Create a name for this property based on the device name and the
/// property name. A '.' is used as the chracter to join them together.
/// This key should be unique for all indi devices.

string IndiProperty::createUniqueKey() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szDevice + "." + m_szName;
}

////////////////////////////////////////////////////////////////////////////////

bool IndiProperty::hasValidBLOBEnable() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_beValue != UnknownBLOBEnable );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty'device' attribute.

bool IndiProperty::hasValidDevice() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szDevice.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'group' attribute.

bool IndiProperty::hasValidGroup() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szGroup.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'label' attribute.

bool IndiProperty::hasValidLabel() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szLabel.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'message' attribute.

bool IndiProperty::hasValidMessage() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szMessage.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'name' attribute.

bool IndiProperty::hasValidName() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szName.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'perm' attribute.

bool IndiProperty::hasValidPerm() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_tPerm != UnknownPropertyPerm );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'rule' attribute.

bool IndiProperty::hasValidRule() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_tRule != UnknownSwitchRule );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'state' attribute.

bool IndiProperty::hasValidState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_tState != UnknownPropertyState );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'timeout' attribute.

bool IndiProperty::hasValidTimeout() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_xTimeout != 0.0f );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'timestamp' attribute.

bool IndiProperty::hasValidTimeStamp() const
{
  // todo: Timestamp is always valid.... this is a weak point.
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'version' attribute.

bool IndiProperty::hasValidVersion() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_szVersion.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the name of the device.

const string &IndiProperty::getDevice() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szDevice;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the BLOB enabled state.

const IndiProperty::BLOBEnableType &IndiProperty::getBLOBEnable() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_beValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the group attribute.

const string &IndiProperty::getGroup() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szGroup;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the label attribute.

const string &IndiProperty::getLabel() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szLabel;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the message attribute.

const string &IndiProperty::getMessage() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szMessage;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the name attribute.

const string &IndiProperty::getName() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szName;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the perm attribute.

const IndiProperty::PropertyPermType &IndiProperty::getPerm() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_tPerm;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the rule stored in the message.

const IndiProperty::SwitchRuleType &IndiProperty::getRule() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_tRule;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of the device.

const IndiProperty::PropertyStateType &IndiProperty::getState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_tState;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the timeout of the device.

const double &IndiProperty::getTimeout() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_xTimeout;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the timestamp stored in the message.

const TimeStamp &IndiProperty::getTimeStamp() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_tsTimeStamp;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the message type.

const IndiProperty::Type &IndiProperty::getType() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the version stored in the message.

const string &IndiProperty::getVersion() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_szVersion;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether or not this property has been requested by a client.
/// This is not managed automatically.

const bool &IndiProperty::isRequested() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_oRequested;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all elements from this object.

void IndiProperty::clear()
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_mapElements.clear();
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setBLOBEnable( const BLOBEnableType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_beValue = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setDevice( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szDevice = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setGroup( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szGroup = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setLabel( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szLabel = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setMessage( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szMessage = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setName( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szName = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setPerm( const IndiProperty::PropertyPermType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_tPerm = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setRequested( const bool &oRequested )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_oRequested = oRequested;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setRule( const IndiProperty::SwitchRuleType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_tRule = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setState( const IndiProperty::PropertyStateType &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_tState = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setTimeout( const double &xValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_xTimeout = xValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setTimeStamp( const TimeStamp &tsValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_tsTimeStamp = tsValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setVersion( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_szVersion = szValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the number of elements.

unsigned int IndiProperty::getNumElements() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_mapElements.size() );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element named szName.
/// Throws exception if name is not found.

const IndiElement& IndiProperty::at( const string& szName ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  map<string, IndiElement>::const_iterator itr = m_mapElements.find( szName );

  if ( itr == m_mapElements.end() )
    throw runtime_error( string( "Element name '" ) + szName + "' not found." );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element named szName.
/// Throws exception if name is not found.

IndiElement& IndiProperty::at( const string& szName )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  map<string, IndiElement>::iterator itr = m_mapElements.find( szName );

  if ( itr == m_mapElements.end() )
    throw runtime_error( string( "Element name '" ) + szName + "' not found." );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element at index uiIndex.
/// Throws exception if index is out of bounds.

const IndiElement& IndiProperty::at( const unsigned int& uiIndex ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  if ( uiIndex > m_mapElements.size() - 1 )
    throw Excep( ErrIndexOutOfBounds );

  map<string, IndiElement>::const_iterator itr = m_mapElements.begin();
  std::advance( itr, uiIndex );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element at index uiIndex.
/// Throws exception if index is out of bounds.

IndiElement& IndiProperty::at( const unsigned int& uiIndex )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  if ( uiIndex > m_mapElements.size() - 1 )
    throw Excep( ErrIndexOutOfBounds );

  map<string, IndiElement>::iterator itr = m_mapElements.begin();
  std::advance( itr, uiIndex );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element named szName.
/// Throws exception if name is not found.

const IndiElement& IndiProperty::operator[]( const string& szName ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  map<string, IndiElement>::const_iterator itr = m_mapElements.find( szName );

  if ( itr == m_mapElements.end() )
    throw runtime_error( string( "Element name '" ) + szName + "' not found." );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element named szName.
/// Throws exception if name is not found.

IndiElement& IndiProperty::operator[]( const string& szName )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  map<string, IndiElement>::iterator itr = m_mapElements.find( szName );

  if ( itr == m_mapElements.end() )
    throw runtime_error( string( "Element name '" ) + szName + "' not found." );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element at an index (zero-based).
/// Throws exception if name is not found.

const IndiElement& IndiProperty::operator[]( const unsigned int& uiIndex ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  if ( uiIndex > m_mapElements.size() - 1 )
    throw Excep( ErrIndexOutOfBounds );

  map<string, IndiElement>::const_iterator itr = m_mapElements.begin();
  std::advance( itr, uiIndex );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element at an index (zero-based).
/// Throws exception if name is not found.

IndiElement& IndiProperty::operator[]( const unsigned int& uiIndex )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  if ( uiIndex > m_mapElements.size() - 1 )
    throw Excep( ErrIndexOutOfBounds );

  map<string, IndiElement>::iterator itr = m_mapElements.begin();
  std::advance( itr, uiIndex );

  return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the entire map of elements.

const map<string, IndiElement> &IndiProperty::getElements() const
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  return m_mapElements;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the entire map of elements.

void IndiProperty::setElements( const map<string, IndiElement> &mapElements )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_mapElements = mapElements;
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the value of an element, adds it if it doesn't exist.

void IndiProperty::update( const IndiElement &ieNew )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  // Actually add it to the map, or update it.
  m_mapElements[ ieNew.getName() ] = ieNew;

  m_tsTimeStamp = TimeStamp::now();
}

////////////////////////////////////////////////////////////////////////////////
/// Adds an element if it doesn't exist. If it does exist, this is a no-op.

void IndiProperty::addIfNoExist( const IndiElement &ieNew )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  map<string, IndiElement>::const_iterator itr =
    m_mapElements.find( ieNew.getName() );

  if ( itr == m_mapElements.end() )
  {
    // Actually add it to the map.
    m_mapElements[ ieNew.getName() ] = ieNew;
    m_tsTimeStamp = TimeStamp::now();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a new IndiElement.
/// Throws if the element already exists.

void IndiProperty::add( const IndiElement &ieNew )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  map<string, IndiElement>::const_iterator itr =
    m_mapElements.find( ieNew.getName() );

  if ( itr != m_mapElements.end() )
    throw Excep( ErrElementAlreadyExists );

  // Actually add it to the map.
  m_mapElements[ ieNew.getName() ] = ieNew;

  m_tsTimeStamp = TimeStamp::now();
}

////////////////////////////////////////////////////////////////////////////////
/// Removes an element named 'szElementName'.
/// Throws if the element doesn't exist.

void IndiProperty::remove( const string &szElementName )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  map<string, IndiElement>::iterator itr =
    m_mapElements.find( szElementName );

  if ( itr == m_mapElements.end() )
    throw Excep( ErrCouldntFindElement );

  // Actually delete the element.
  m_mapElements.erase( itr );
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the value of an element named 'szElementName'.
/// Throws if the element doesn't exist.

void IndiProperty::update( const string &szElementName,
                           const IndiElement &ieUpdate )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  map<string, IndiElement>::iterator itr =
    m_mapElements.find( szElementName );

  if ( itr == m_mapElements.end() )
    throw Excep( ErrCouldntFindElement );

  itr->second = ieUpdate;

  m_tsTimeStamp = TimeStamp::now();
}

////////////////////////////////////////////////////////////////////////////////
///  Returns true if the element 'szElementName' exists, false otherwise.

bool IndiProperty::find( const string &szElementName ) const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  map<string, IndiElement>::const_iterator itr =
    m_mapElements.find( szElementName );

  return ( itr != m_mapElements.end() );
}

////////////////////////////////////////////////////////////////////////////////
///  return the message concerning the error.
/// @param nErr The error to look up the message for.

string IndiProperty::getErrorMsg( const int &nErr )
{
  string szMsg;
  switch ( nErr )
  {
    //  errors defined in this class.
    case ErrNone:
      szMsg = "No Error";
      break;
    case ErrCouldntFindElement:
      szMsg = "Could not find element";
      break;
    case ErrElementAlreadyExists:
      szMsg = "Element already exists";
      break;
    case ErrIndexOutOfBounds:
      szMsg = "Index out of bounds";
      break;
    default:
      szMsg = "Unknown error";
      break;
  }
  return szMsg;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiProperty::BLOBEnableType IndiProperty::getBLOBEnableType( const string &szType )
{
  BLOBEnableType tType = UnknownBLOBEnable;

  if ( szType == "Never" )
    tType = Never;
  else if ( szType == "Also" )
    tType = Also;
  else if ( szType == "Only" )
    tType = Only;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getBLOBEnableString( const BLOBEnableType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case UnknownBLOBEnable:
      szType = "";
      break;
    case Never:
      szType = "Never";
      break;
    case Also:
      szType = "Also";
      break;
    case Only:
      szType = "Only";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiProperty::PropertyStateType IndiProperty::getPropertyStateType( const string &szType )
{
  PropertyStateType tType = UnknownPropertyState;

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

string IndiProperty::getPropertyStateString( const PropertyStateType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case UnknownPropertyState:
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

IndiProperty::SwitchRuleType IndiProperty::getSwitchRuleType( const string &szType )
{
  SwitchRuleType tType = UnknownSwitchRule;

  if ( szType == "OneOfMany" )
    tType = OneOfMany;
  else if ( szType == "AtMostOne" )
    tType = AtMostOne;
  else if ( szType == "AnyOfMany" )
    tType = AnyOfMany;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getSwitchRuleString( const SwitchRuleType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case OneOfMany:
      szType = "OneOfMany";
      break;
    case AtMostOne:
      szType = "AtMostOne";
      break;
    case AnyOfMany:
      szType = "AnyOfMany";
      break;
    case UnknownSwitchRule:
      szType = "";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiProperty::PropertyPermType IndiProperty::getPropertyPermType( const string &szType )
{
  PropertyPermType tType = UnknownPropertyPerm;

  if ( szType == "ro" )
    tType = ReadOnly;
  else if ( szType == "wo" )
    tType = WriteOnly;
  else if ( szType == "rw" )
    tType = ReadWrite;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getPropertyPermString( const PropertyPermType &tType )
{
  string szType = "";

  switch ( tType )
  {
    case UnknownPropertyPerm:
      szType = "";
      break;
    case ReadOnly:
      szType = "ro";
      break;
    case WriteOnly:
      szType = "wo";
      break;
    case ReadWrite:
      szType = "rw";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::convertTypeToString( const Type &tType )
{
  string szType = "";

  switch ( tType )
  {
    case IndiProperty::Unknown:
      szType = "";
      break;
    case IndiProperty::BLOB:
      szType = "BLOB";
      break;
    case IndiProperty::Light:
      szType = "Light";
      break;
    case IndiProperty::Number:
      szType = "Number";
      break;
    case IndiProperty::Switch:
      szType = "Switch";
      break;
    case IndiProperty::Text:
      szType = "Text";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the tag.

IndiProperty::Type IndiProperty::convertStringToType( const string &szTag )
{
  Type tType = IndiProperty::Unknown;

  // Define properties.
  if ( szTag == "BLOB" )
    tType = IndiProperty::BLOB;
  else if ( szTag == "Light" )
    tType = IndiProperty::Light;
  else if ( szTag == "Number" )
    tType = IndiProperty::Number;
  else if ( szTag == "Switch" )
    tType = IndiProperty::Switch;
  else if ( szTag == "Text" )
    tType = IndiProperty::Text;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
