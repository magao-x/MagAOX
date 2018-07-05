/// IndiPropertyMap.hpp
///
/// @author Mark Milton
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef INDI_PROPERTY_MAP_HPP
#define INDI_PROPERTY_MAP_HPP

#include <exception>
#include <string>
#include "IndiProperty.hpp"
#include "IndiDriver.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace pcf
{
class IndiPropertyMap
{
  // Our iterators
  public:
    typedef std::map<std::string, pcf::IndiProperty>::iterator Iterator;
    typedef std::map<std::string, pcf::IndiProperty>::const_iterator ConstIterator;

  // Construction/destruction/assign/copy
  public:
    IndiPropertyMap();
    ~IndiPropertyMap();
    IndiPropertyMap( const IndiPropertyMap &ipmRhs );
    IndiPropertyMap const  &operator= ( const IndiPropertyMap &ipmRhs );

  // Operators
  public:
    // Return a reference to a property so it can be used.
    const pcf::IndiProperty &operator[] ( const std::string& szPropertyName ) const;
    // Return a reference to a property so it can be modified.
    pcf::IndiProperty &operator[] ( const std::string& szPropertyName );

  // Member functions.
  public:
    /// Add a property to this object.
    void add( const pcf::IndiProperty &ipAdd );
    /// The first item in the map.
    Iterator begin();
    ConstIterator begin() const;
    /// Remove all items in this object.
    void clear();
    /// Past the last item in the map.
    Iterator end();
    ConstIterator end() const;
    /// Find matching property in this object which is named "szPropertyName".
    Iterator find( const std::string& szPropertyName );
    ConstIterator find( const std::string& szPropertyName ) const;
    /// Find matching property in this object which contains a label which
    /// consists of 'szLabelText'.
    Iterator findLabel( const std::string& szLabelText );
    ConstIterator findLabel( const std::string& szLabelText ) const;
    /// Find a property which has an element named "szElementName" that has
    /// a value of "szElementValue".
    Iterator find( const std::string& szElementName,
                   const std::string& szElementValue );
    ConstIterator find( const std::string& szElementName,
                        const std::string& szElementValue ) const;
    /// Remove a property based on the name.
    void erase( const std::string &szPropertyName );
    /// Initializes the property by placing (overwriting) the entry in the map.
    /// Afterwards, a DEF will be sent for that property. If the szNewName
    /// is not empty, the name of the property being sent will be changed to
    /// that name, effectively making each instance sent the same property.
    /// In this way, a client will only need to handle one property, but will
    /// receive many instances of it. This is useful in handling lists of
    /// things, such as sensors or motors, etc.
    void init( const pcf::IndiDriver &idParent,
               const pcf::IndiProperty &ipInit,
               const std::string &szNewName = "" );
    /// Sends a DEF PROPERTIES based on the szPropertyName. If szPropertyName
    /// is empty, a DEF is sent for all properties. The DEF will be sent using
    /// the idParent driver calls. If the szNewName is not empty, the name
    /// of the property being sent will be changed to that name, effectively
    /// making each instance sent the same property. In this way, a client
    /// will only need to handle one property, but will receive many instances
    /// of it. This is useful in handling lists of things, such as sensors
    /// or motors, etc.
    void sendDef( const pcf::IndiDriver &idParent,
                  const std::string &szPropertyName = "",
                  const std::string &szNewName = "" ) const;
    /// Send a GET for the property named szPropertyName. If szPropertyName is
    /// empty, a GET will be sent for every property.
    void sendGet( pcf::IndiDriver &idParent,
                  const std::string &szPropertyName = "" ) const;
    /// Send a SET for the property named szPropertyName. If szPropertyName is
    /// empty, a SET will be sent for every property. If the szNewName is not
    /// empty, the name of the property being sent will be changed to that name,
    /// effectively making each instance sent the same property. In this way,
    /// a client will only need to handle one property, but will receive many
    /// instances of it. This is useful in handling lists of things, such as
    /// sensors or motors, etc.
    void sendSet( const pcf::IndiDriver &idParent,
                  const std::string &szPropertyName = "",
                  const std::string &szNewName = "" ) const;
    /// The number of elements in the map.
    uint64_t size() const;
    /// Updates the property by updating the entry in the map. Afterwards,
    /// a SET will be sent for that property if it has changed. If the
    /// szNewName is not empty, the name of the property being sent will be
    /// changed to that name, effectively making each instance sent the same
    /// property. In this way, a client will only need to handle one property,
    /// but will receive many instances of it. This is useful in handling
    /// lists of things, such as sensors or motors, etc.
    void update( const pcf::IndiDriver &idParent,
                 const pcf::IndiProperty &ipUpdate,
                 const std::string &szNewName = "" );
    /// Updates the state for the property named szPropertyName. If
    /// szPropertyName is empty, all states will be updated. Afterwards,
    /// a SET will be sent for each property updated. If the szNewName is
    /// not empty, the name of the property being sent will be changed to
    /// that name, effectively making each instance sent the same property.
    /// In this way, a client will only need to handle one property, but
    /// will receive many instances of it. This is useful in handling lists
    /// of things, such as sensors or motors, etc.
    void updateState( const pcf::IndiDriver &idParent,
                      const pcf::IndiProperty::PropertyStateType &pstNew,
                      const std::string &szPropertyName = "",
                      const std::string &szNewName = "" );

  // Variables
  private:
    /// The dictionary of properties
    std::map<std::string, pcf::IndiProperty> m_mapProperties;

}; // class IndiPropertyMap
} // namespace mcf

////////////////////////////////////////////////////////////////////////////////

#endif // INDI_PROPERTY_MAP_HPP
