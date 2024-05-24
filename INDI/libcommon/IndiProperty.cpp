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

IndiProperty::IndiProperty()
{
}

IndiProperty::IndiProperty( const Type &tType ) : m_type(tType)
{
}

IndiProperty::IndiProperty( const Type &tType,
                            const string &szDevice,
                            const string &szName ) :  m_device(szDevice), m_name(szName), m_type(tType)
{
}

IndiProperty::IndiProperty( const Type &tType,
                            const string &szDevice,
                            const string &szName,
                            const PropertyState &tState,
                            const PropertyPerm &tPerm,
                            const SwitchRule &tRule ) : m_device(szDevice), m_name(szName), m_perm(tPerm),
                                                               m_rule(tRule), m_state(tState), m_type(tType)
{
}

IndiProperty::IndiProperty(const IndiProperty &ipRhs ) : m_device(ipRhs.m_device), m_group(ipRhs.m_group), m_label(ipRhs.m_label),
                                                           m_message(ipRhs.m_message), m_name(ipRhs.m_name), m_perm(ipRhs.m_perm),
                                                            m_rule(ipRhs.m_rule), m_state(ipRhs.m_state), m_timeout(ipRhs.m_timeout),
                                                              m_requested(ipRhs.m_requested),  m_timeStamp(ipRhs.m_timeStamp),
                                                                m_version(ipRhs.m_version), m_beValue(ipRhs.m_beValue), 
                                                                 m_mapElements(ipRhs.m_mapElements), m_type(ipRhs.m_type)
{
}

IndiProperty::~IndiProperty()
{
}

const IndiProperty &IndiProperty::operator=( const IndiProperty &ipRhs )
{
  if ( &ipRhs != this )
  {
    pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

    m_device = ipRhs.m_device;
    m_group = ipRhs.m_group;
    m_label = ipRhs.m_label;
    m_message = ipRhs.m_message;
    m_name = ipRhs.m_name;
    m_perm = ipRhs.m_perm;
    m_requested = ipRhs.m_requested;
    m_rule = ipRhs.m_rule;
    m_state = ipRhs.m_state;
    m_timeout = ipRhs.m_timeout;
    m_timeStamp = ipRhs.m_timeStamp;
    m_version = ipRhs.m_version;
    m_beValue = ipRhs.m_beValue;

    m_mapElements = ipRhs.m_mapElements;
    m_type = ipRhs.m_type;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// This is an alternate way of calling 'setBLOBEnable'.

const IndiProperty::BLOBEnable &IndiProperty::operator=( const BLOBEnable &tValue )
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
  return ( m_device == ipRhs.m_device &&
    m_group == ipRhs.m_group &&
    m_label == ipRhs.m_label &&
    m_message == ipRhs.m_message &&
    m_name == ipRhs.m_name &&
    m_perm == ipRhs.m_perm &&
    //m_requested == ipRhs.m_requested &&
    m_rule == ipRhs.m_rule &&
    m_state == ipRhs.m_state &&
    //m_timeout == ipRhs.m_timeout &&
    //m_timeStamp ==ipRhs.m_timeStamp &&  // Don't compare!
    m_version == ipRhs.m_version &&
    m_beValue == ipRhs.m_beValue &&
    m_type == ipRhs.m_type );
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

  if ( ipComp.getType() != m_type )
    return false;

  if ( ipComp.getDevice() != m_device || ipComp.getName() != m_name )
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

  if ( ipComp.getType() != m_type )
    return false;

  if ( ipComp.getDevice() != m_device || ipComp.getName() != m_name )
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
  if ( itr->second.value() != itrComp->second.value() )
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

  if ( ipComp.getType() != m_type )
    return false;

  if ( ipComp.getDevice() != m_device || ipComp.getName() != m_name )
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
    if ( itrComp->second.value() != itr->second.value() )
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
  if ( ipComp.getType() != m_type )
    return false;

  if ( ipComp.getDevice() != m_device || ipComp.getName() != m_name )
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
  if ( itr->second.value() != itrComp->second.value() &&
      itrComp->second.value().length() > 0 )
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
           << "\"device\" : \"" << m_device << "\" , "
           << "\"name\" : \"" << m_name << "\" , "
           << "\"type\" : \"" << convertTypeToString( m_type ) << "\" , "
           << "\"group\" : \"" << m_group << "\" , "
           << "\"label\" : \"" << m_label << "\" , "
           << "\"timeout\" : \"" << m_timeout << "\" , "
           << "\"version\" : \"" << m_version << "\" , "
           << "\"timestamp\" : \"" << m_timeStamp.getFormattedIso8601Str() << "\" , "
           << "\"perm\" : \"" << getPropertyPermString( m_perm ) << "\" , "
           << "\"rule\" : \"" << getSwitchRuleString( m_rule ) << "\" , "
           << "\"state\" : \"" << getPropertyStateString( m_state ) << "\" , "
           << "\"BLOBenable\" : \"" << getBLOBEnableString( m_beValue ) << "\" , "
           << "\"message\" : \"" << m_message << "\" "
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
  return m_device + "." + m_name;
}

////////////////////////////////////////////////////////////////////////////////

bool IndiProperty::hasValidBLOBEnable() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_beValue != BLOBEnable::Unknown );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty'device' attribute.

bool IndiProperty::hasValidDevice() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_device.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'group' attribute.

bool IndiProperty::hasValidGroup() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_group.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'label' attribute.

bool IndiProperty::hasValidLabel() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_label.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'message' attribute.

bool IndiProperty::hasValidMessage() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_message.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'name' attribute.

bool IndiProperty::hasValidName() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_name.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'perm' attribute.

bool IndiProperty::hasValidPerm() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_perm != PropertyPerm::Unknown );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'rule' attribute.

bool IndiProperty::hasValidRule() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_rule != SwitchRule::Unknown );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'state' attribute.

bool IndiProperty::hasValidState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_state != PropertyState::Unknown );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if this contains a non-empty 'timeout' attribute.

bool IndiProperty::hasValidTimeout() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return ( m_timeout != 0.0f );
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
  return ( m_version.size() != 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the name of the device.

const string &IndiProperty::getDevice() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_device;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the BLOB enabled state.

const IndiProperty::BLOBEnable &IndiProperty::getBLOBEnable() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_beValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the group attribute.

const string &IndiProperty::getGroup() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_group;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the label attribute.

const string &IndiProperty::getLabel() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_label;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the message attribute.

const string &IndiProperty::getMessage() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_message;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the name attribute.

const string &IndiProperty::getName() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_name;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value of the perm attribute.

const IndiProperty::PropertyPerm &IndiProperty::getPerm() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_perm;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the rule stored in the message.

const IndiProperty::SwitchRule &IndiProperty::getRule() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_rule;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the state of the device.

const IndiProperty::PropertyState &IndiProperty::getState() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_state;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the timeout of the device.

const double &IndiProperty::getTimeout() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_timeout;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the timestamp stored in the message.

const TimeStamp &IndiProperty::getTimeStamp() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_timeStamp;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the message type.

const IndiProperty::Type &IndiProperty::getType() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_type;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the version stored in the message.

const string &IndiProperty::getVersion() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_version;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether or not this property has been requested by a client.
/// This is not managed automatically.

const bool &IndiProperty::isRequested() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
  return m_requested;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all elements from this object.

void IndiProperty::clear()
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_mapElements.clear();
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setBLOBEnable( const BLOBEnable &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_beValue = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setDevice( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_device = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setGroup( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_group = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setLabel( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_label = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setMessage( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_message = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setName( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_name = szValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setPerm( const IndiProperty::PropertyPerm &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_perm = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setRequested( const bool &oRequested )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_requested = oRequested;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setRule( const IndiProperty::SwitchRule &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_rule = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setState( const IndiProperty::PropertyState &tValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_state = tValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setTimeout( const double &xValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_timeout = xValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setTimeStamp( const TimeStamp &tsValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_timeStamp = tsValue;
}

////////////////////////////////////////////////////////////////////////////////

void IndiProperty::setVersion( const string &szValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
  m_version = szValue;
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
    throw Excep( Error::IndexOutOfBounds );

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
    throw Excep( Error::IndexOutOfBounds );

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
    throw Excep( Error::IndexOutOfBounds );

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
    throw Excep( Error::IndexOutOfBounds );

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
  m_mapElements[ ieNew.name() ] = ieNew;

  m_timeStamp = TimeStamp::now();
}

////////////////////////////////////////////////////////////////////////////////
/// Adds an element if it doesn't exist. If it does exist, this is a no-op.

void IndiProperty::addIfNoExist( const IndiElement &ieNew )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  map<string, IndiElement>::const_iterator itr =
    m_mapElements.find( ieNew.name() );

  if ( itr == m_mapElements.end() )
  {
    // Actually add it to the map.
    m_mapElements[ ieNew.name() ] = ieNew;
    m_timeStamp = TimeStamp::now();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a new IndiElement.
/// Throws if the element already exists.

void IndiProperty::add( const IndiElement &ieNew )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  map<string, IndiElement>::const_iterator itr =
    m_mapElements.find( ieNew.name() );

  if ( itr != m_mapElements.end() )
    throw Excep( Error::ElementAlreadyExists );

  // Actually add it to the map.
  m_mapElements[ ieNew.name() ] = ieNew;

  m_timeStamp = TimeStamp::now();
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
    throw Excep( Error::CouldntFindElement );

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
    throw Excep( Error::CouldntFindElement );

  itr->second = ieUpdate;

  m_timeStamp = TimeStamp::now();
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

string IndiProperty::getErrorMsg( const Error &nErr )
{
  string szMsg;
  switch ( nErr )
  {
    //  errors defined in this class.
    case Error::None:
      szMsg = "No Error";
      break;
    case Error::CouldntFindElement:
      szMsg = "Could not find element";
      break;
    case Error::ElementAlreadyExists:
      szMsg = "Element already exists";
      break;
    case Error::IndexOutOfBounds:
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

IndiProperty::BLOBEnable IndiProperty::getBLOBEnable( const string &szType )
{
  BLOBEnable tType = BLOBEnable::Unknown;

  if ( szType == "Never" )
  {
    tType = BLOBEnable::Never;
  }
  else if ( szType == "Also" )
  {
    tType = BLOBEnable::Also;
  }
  else if ( szType == "Only" )
  {
    tType = BLOBEnable::Only;
  }

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getBLOBEnableString( const BLOBEnable &tType )
{
  string szType = "";

  switch ( tType )
  {
    case BLOBEnable::Unknown:
      szType = "";
      break;
    case BLOBEnable::Never:
      szType = "Never";
      break;
    case BLOBEnable::Also:
      szType = "Also";
      break;
    case BLOBEnable::Only:
      szType = "Only";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiProperty::PropertyState IndiProperty::getPropertyState( const string &szType )
{
  PropertyState tType = PropertyState::Unknown;

  if ( szType == "Idle" )
  {
    tType = PropertyState::Idle;
  }
  else if ( szType == "Ok" )
  {
    tType = PropertyState::Ok;
  }
  else if ( szType == "Busy" )
  {
    tType = PropertyState::Busy;
  }
  else if ( szType == "Alert" )
  {
    tType = PropertyState::Alert;
  }

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getPropertyStateString( const PropertyState &tType )
{
  string szType = "";

  switch ( tType )
  {
    case PropertyState::Unknown:
      szType = "";
      break;
    case PropertyState::Idle:
      szType = "Idle";
      break;
    case PropertyState::Ok:
      szType = "Ok";
      break;
    case PropertyState::Busy:
      szType = "Busy";
      break;
    case PropertyState::Alert:
      szType = "Alert";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiProperty::SwitchRule IndiProperty::getSwitchRule( const string &szType )
{
  SwitchRule tType = SwitchRule::Unknown;

  if ( szType == "OneOfMany" )
  {
    tType = SwitchRule::OneOfMany;
  }
  else if ( szType == "AtMostOne" )
  {
    tType = SwitchRule::AtMostOne;
  }
  else if ( szType == "AnyOfMany" )
  {
    tType = SwitchRule::AnyOfMany;
  }

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getSwitchRuleString( const SwitchRule &tType )
{
  string szType = "";

  switch ( tType )
  {
    case SwitchRule::OneOfMany:
      szType = "OneOfMany";
      break;
    case SwitchRule::AtMostOne:
      szType = "AtMostOne";
      break;
    case SwitchRule::AnyOfMany:
      szType = "AnyOfMany";
      break;
    case SwitchRule::Unknown:
      szType = "";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the string type.

IndiProperty::PropertyPerm IndiProperty::getPropertyPerm( const string &szType )
{
  PropertyPerm tType = PropertyPerm::Unknown;

  if ( szType == "ro" )
  {
    tType = PropertyPerm::ReadOnly;
  }
  else if ( szType == "wo" )
  {
    tType = PropertyPerm::WriteOnly;
  }
  else if ( szType == "rw" )
  {
    tType = PropertyPerm::ReadWrite;
  }

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the string type given the enumerated type.

string IndiProperty::getPropertyPermString( const PropertyPerm &tType )
{
  string szType = "";

  switch ( tType )
  {
    case PropertyPerm::Unknown:
      szType = "";
      break;
    case PropertyPerm::ReadOnly:
      szType = "ro";
      break;
    case PropertyPerm::WriteOnly:
      szType = "wo";
      break;
    case PropertyPerm::ReadWrite:
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
    case Type::Unknown:
      szType = "";
      break;
    case Type::BLOB:
      szType = "BLOB";
      break;
    case Type::Light:
      szType = "Light";
      break;
    case Type::Number:
      szType = "Number";
      break;
    case Type::Switch:
      szType = "Switch";
      break;
    case Type::Text:
      szType = "Text";
      break;
  }

  return szType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the enumerated type given the tag.

IndiProperty::Type IndiProperty::convertStringToType( const string &szTag )
{
  Type tType = Type::Unknown;

  // Define properties.
  if ( szTag == "BLOB" )
    tType = Type::BLOB;
  else if ( szTag == "Light" )
    tType = Type::Light;
  else if ( szTag == "Number" )
    tType = Type::Number;
  else if ( szTag == "Switch" )
    tType = Type::Switch;
  else if ( szTag == "Text" )
    tType = Type::Text;

  return tType;
}

////////////////////////////////////////////////////////////////////////////////
