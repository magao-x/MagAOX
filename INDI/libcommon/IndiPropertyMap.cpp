/// $Id: IndiPropertyMap.cpp
///
/// @author Mark Milton
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include "IndiPropertyMap.hpp"

using std::string;
using std::map;
using std::runtime_error;

using pcf::IndiProperty;
using pcf::IndiDriver;
using pcf::IndiPropertyMap;

////////////////////////////////////////////////////////////////////////////////
/// Standard constructor.

IndiPropertyMap::IndiPropertyMap()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor.

IndiPropertyMap::~IndiPropertyMap()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor - just copy the properties.
/// @param rhs Another version of the driver.

IndiPropertyMap::IndiPropertyMap( const IndiPropertyMap &ipmRhs ) : m_mapProperties(ipmRhs.m_mapProperties)
{ 
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator - rhs the properties.
/// @param rhs Another version of this object.

const IndiPropertyMap &IndiPropertyMap::operator= ( const IndiPropertyMap &ipmRhs )
{
  if ( this != &ipmRhs )
  {
    m_mapProperties = ipmRhs.m_mapProperties;
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a property to this object.
/// @param ipAdd Property to add.
///
void IndiPropertyMap::add( const pcf::IndiProperty &ipAdd )
{
  m_mapProperties[ ipAdd.getName() ] = ipAdd;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all items from this object.

void IndiPropertyMap::clear()
{
  m_mapProperties.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Find matching property in this object (false if not found)
/// @param szPropertyName The property name.

IndiPropertyMap::Iterator IndiPropertyMap::find( const string& szPropertyName )
{
  return m_mapProperties.find( szPropertyName );
}

////////////////////////////////////////////////////////////////////////////////
/// Find matching property in this object (false if not found)
/// @param szPropertyName The property name.

IndiPropertyMap::ConstIterator IndiPropertyMap::find(const string& szPropertyName ) const
{
  return m_mapProperties.find( szPropertyName );
}

////////////////////////////////////////////////////////////////////////////////
/// Find matching property in this object which has a label consisting of
/// 'szLabelText'

IndiPropertyMap::Iterator IndiPropertyMap::findLabel( const string& szLabelText )
{
  map<string, IndiProperty>::iterator itr = m_mapProperties.begin();
  for ( ; itr != m_mapProperties.end(); ++itr )
  {
    if ( itr->second.getLabel() == szLabelText )
    {
      return itr;
    }
  }

  // If we get here, we didn't find it.
  return m_mapProperties.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Find matching property in this object which has a label consisting of
/// 'szLabelText'

IndiPropertyMap::ConstIterator IndiPropertyMap::findLabel( const string& szLabelText ) const
{
  map<string, IndiProperty>::const_iterator itr = m_mapProperties.begin();
  for ( ; itr != m_mapProperties.end(); ++itr )
  {
    if ( itr->second.getLabel() == szLabelText )
    {
      return itr;
    }
  }

  // If we get here, we didn't find it.
  return m_mapProperties.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Find a property which has an element named "szElementName" that has
/// a value of "szElementValue".

IndiPropertyMap::Iterator IndiPropertyMap::find( const string& szElementName,
                                                 const string& szElementValue )
{
  map<string, IndiProperty>::iterator itr = m_mapProperties.begin();
  for ( ; itr != m_mapProperties.end(); ++itr )
  {
    if ( itr->second.find( szElementName ) == true &&
         itr->second[szElementName].get() == szElementValue )
    {
      return itr;
    }
  }

  // If we get here, we didn't find it.
  return m_mapProperties.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Find a property which has an element named "szElementName" that has
/// a value of "szElementValue".

IndiPropertyMap::ConstIterator IndiPropertyMap::find( const string& szElementName,
                                                      const string& szElementValue ) const
{
  map<string, IndiProperty>::const_iterator itr = m_mapProperties.begin();
  for ( ; itr != m_mapProperties.end(); ++itr )
  {
    if ( itr->second.find( szElementName ) == true &&
         itr->second[szElementName].get() == szElementValue )
    {
      return itr;
    }
  }

  // If we get here, we didn't find it.
  return m_mapProperties.end();
}

////////////////////////////////////////////////////////////////////////////////
/// The first item in the map.
/// @return The iterator to the first item.

IndiPropertyMap::Iterator IndiPropertyMap::begin()
{
  return m_mapProperties.begin();
}

////////////////////////////////////////////////////////////////////////////////
/// The first item in the map.
/// @return The iterator to the first item.

IndiPropertyMap::ConstIterator IndiPropertyMap::begin() const
{
  return m_mapProperties.begin();
}

////////////////////////////////////////////////////////////////////////////////
/// Beyond the last item in the map.
/// @return The iterator to the end item.

IndiPropertyMap::Iterator IndiPropertyMap::end()
{
  return m_mapProperties.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Beyond the last item in the map.
/// @return The iterator to the end item.

IndiPropertyMap::ConstIterator IndiPropertyMap::end() const
{
  return m_mapProperties.end();
}

////////////////////////////////////////////////////////////////////////////////
// Return a reference to a property so it can be used.

const IndiProperty &IndiPropertyMap::operator[] ( const string& szPropertyName ) const
{
  return m_mapProperties.find( szPropertyName )->second;
}

////////////////////////////////////////////////////////////////////////////////
// Return a reference to a property so it can be modified.

IndiProperty &IndiPropertyMap::operator[] ( const string& szPropertyName )
{
  return m_mapProperties.find( szPropertyName )->second;
}

////////////////////////////////////////////////////////////////////////////////
/// The number of elements in the map.

uint64_t IndiPropertyMap::size() const
{
  return m_mapProperties.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a property from this object.
/// @param ipAdd Property to remove.
///
void IndiPropertyMap::erase( const string &szPropertyName )
{
  m_mapProperties.erase( szPropertyName );
}

////////////////////////////////////////////////////////////////////////////////
/// Send a SET for the property named szPropertyName. If szPropertyName is
/// empty, a SET will be sent for every property. If the szNewName is not
/// empty, the name of the property being sent will be changed to that name,
/// effectively making each instance sent the same property. In this way,
/// a client will only need to handle one property, but will receive many
/// instances of it. This is useful in handling lists of things, such as
/// sensors or motors, etc.

void IndiPropertyMap::sendSet( const IndiDriver &idParent,
                               const string &szPropertyName,
                               const string &szNewName ) const
{
  // Should we update all properties?
  if ( szPropertyName.length() == 0 )
  {
    map<string, IndiProperty>::const_iterator itr = m_mapProperties.begin();
    for ( ; itr != m_mapProperties.end(); ++itr )
    {
      if ( szNewName.length() == 0 )
      {
        idParent.sendSetProperty( itr->second );
      }
      else
      {
        IndiProperty ipSend = itr->second;
        ipSend.setName( szNewName );
        idParent.sendSetProperty( ipSend );
      }
    }
  }
  else
  {
    map<string, IndiProperty>::const_iterator itr =
        m_mapProperties.find( szPropertyName );
    if ( itr != m_mapProperties.end() )
    {
      // Send with the modified name, if desired.
      if ( szNewName.length() == 0 )
      {
        idParent.sendSetProperty( itr->second );
      }
      else
      {
        IndiProperty ipSend = itr->second;
        ipSend.setName( szNewName );
        idParent.sendSetProperty( ipSend );
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes the property by placing (overwriting) the entry in the map.
/// Afterwards, a DEF will be sent for that property. If the szNewName
/// is not empty, the name of the property being sent will be changed to
/// that name, effectively making each instance sent the same property.
/// In this way, a client will only need to handle one property, but will
/// receive many instances of it. This is useful in handling lists of
/// things, such as sensors or motors, etc.

void IndiPropertyMap::init( const IndiDriver &idParent,
                            const IndiProperty &ipInit,
                            const string &szNewName )
{
  m_mapProperties[ ipInit.getName() ] = ipInit;

  // Send with the modified name, if desired.
  if ( szNewName.length() == 0 )
  {
    idParent.sendDefProperty( ipInit );
  }
  else
  {
    IndiProperty ipSend = ipInit;
    ipSend.setName( szNewName );
    idParent.sendDefProperty( ipSend );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the property by updating the entry in the map. Afterwards,
/// a SET will be sent for that property if it has changed. If the
/// szNewName is not empty, the name of the property being sent will be
/// changed to that name, effectively making each instance sent the same
/// property. In this way, a client will only need to handle one property,
/// but will receive many instances of it. This is useful in handling
/// lists of things, such as sensors or motors, etc.

void IndiPropertyMap::update( const IndiDriver &idParent,
                              const IndiProperty &ipUpdate,
                              const string &szNewName )
{
  map<string, IndiProperty>::iterator itr =
      m_mapProperties.find( ipUpdate.getName() );

  if ( itr == m_mapProperties.end() )
    throw runtime_error( "Can't update property - not found." );

  // Has anything changed with this property?
  if ( ipUpdate == itr->second )
  {
    // The two properties are an exact match, so do nothing.
  }
  // Send with the modified name, if desired.
  else if ( szNewName.length() == 0 )
  {
    itr->second = ipUpdate;
    idParent.sendSetProperty( itr->second );
  }
  else
  {
    itr->second = ipUpdate;
    IndiProperty ipSend = itr->second;
    ipSend.setName( szNewName );
    idParent.sendSetProperty( ipSend );
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Sends a DEF PROPERTIES based on the szPropertyName. If szPropertyName
/// is empty, a DEF is sent for all properties. The DEF will be sent using
/// the idParent driver calls. If the szNewName is not empty, the name
/// of the property being sent will be changed to that name, effectively
/// making each instance sent the same property. In this way, a client
/// will only need to handle one property, but will receive many instances
/// of it. This is useful in handling lists of things, such as sensors
/// or motors, etc.

void IndiPropertyMap::sendDef( const IndiDriver &idParent,
                               const string &szPropertyName,
                               const string &szNewName ) const
{
  // Send all properties if the name is not specified
  if ( szPropertyName.length() == 0 )
  {
    map<string, IndiProperty>::const_iterator itr = m_mapProperties.begin();
    for ( ; itr != m_mapProperties.end(); ++itr )
    {
      if ( szNewName.length() == 0 )
      {
        idParent.sendDefProperty( itr->second );
      }
      else
      {
        IndiProperty ipSend = itr->second;
        ipSend.setName( szNewName );
        idParent.sendDefProperty( ipSend );
      }
    }
  }
  // Otherwise, just send requested properties for the requested name.
  else
  {
    map<string, IndiProperty>::const_iterator itr =
        m_mapProperties.find( szPropertyName );

    if ( itr != m_mapProperties.end() )
    {
      // Send with the modified name, if desired.
      if ( szNewName.length() == 0 )
      {
        idParent.sendDefProperty( itr->second );
      }
      else
      {
        IndiProperty ipSend = itr->second;
        ipSend.setName( szNewName );
        idParent.sendDefProperty( ipSend );
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Updates the state for the property named szPropertyName. If
/// szPropertyName is empty, all states will be updated. Afterwards,
/// a SET will be sent for each property updated. If the szNewName is
/// not empty, the name of the property being sent will be changed to
/// that name, effectively making each instance sent the same property.
/// In this way, a client will only need to handle one property, but
/// will receive many instances of it. This is useful in handling lists
/// of things, such as sensors or motors, etc.

void IndiPropertyMap::updateState( const IndiDriver &idParent,
                                   const IndiProperty::PropertyStateType &pstNew,
                                   const string &szPropertyName,
                                   const string &szNewName )
{
  // Should we update all properties?
  if ( szPropertyName.length() == 0 )
  {
    map<string, IndiProperty>::iterator itr = m_mapProperties.begin();
    for ( ; itr != m_mapProperties.end(); ++itr )
    {
      if ( pstNew != itr->second.getState() )
      {
        itr->second.setState( pstNew );

        if ( szNewName.length() == 0 )
        {
          idParent.sendDefProperty( itr->second );
        }
        else
        {
          IndiProperty ipSend = itr->second;
          ipSend.setName( szNewName );
          idParent.sendDefProperty( ipSend );
        }
      }
    }
  }
  // Update just one property.
  else
  {
    map<string, IndiProperty>::iterator itr = m_mapProperties.find( szPropertyName );
    if ( itr != m_mapProperties.end() )
    {
      if ( pstNew != itr->second.getState() )
      {
        itr->second.setState( pstNew );

        if ( szNewName.length() == 0 )
        {
          idParent.sendDefProperty( itr->second );
        }
        else
        {
          IndiProperty ipSend = itr->second;
          ipSend.setName( szNewName );
          idParent.sendDefProperty( ipSend );
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Send a GET for the property named szPropertyName. If szPropertyName is
/// empty, a GET will be sent for every property.

void IndiPropertyMap::sendGet( IndiDriver &idParent,
                               const string &szPropertyName ) const
{
  // Should we send a GET for all properties?
  if ( szPropertyName.length() == 0 )
  {
    map<string, IndiProperty>::const_iterator itr = m_mapProperties.begin();
    for ( ; itr != m_mapProperties.end(); ++itr )
    {
      idParent.sendGetProperties( itr->second );
    }
  }
  // Issue a get property for every property.
  else
  {
    map<string, IndiProperty>::const_iterator itr =
        m_mapProperties.find( szPropertyName );
    if ( itr != m_mapProperties.end() )
    {
      idParent.sendGetProperties( itr->second );
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
