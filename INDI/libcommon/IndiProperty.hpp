/// IndiProperty.hpp
///
/// @author Paul Grenz
///
/// This class represents a list of INDI elements with additional information
/// associated with it. All access is protected by a read-write lock.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef INDI_PROPERTY_HPP
#define INDI_PROPERTY_HPP
#pragma once

#include <string>
#include <map>
#include <exception>
#include "ReadWriteLock.hpp"
#include "TimeStamp.hpp"
#include "IndiElement.hpp"

namespace pcf
{
class IndiProperty
{
  public:
    enum Error
    {
      ErrNone =                   0,
      ErrCouldntFindElement =    -3,
      ErrElementAlreadyExists =  -5,
      ErrIndexOutOfBounds =      -6,
      ErrWrongElementType =      -7,
      ErrUndefined =          -9999
    };

    enum BLOBEnableType
    {
      UnknownBLOBEnable = 0,
      Also = 1,
      Only,
      Never
    };

    enum PropertyStateType
    {
      UnknownPropertyState = 0,
      Alert = 1,
      Busy,
      Ok,
      Idle
    };

    enum SwitchRuleType
    {
      UnknownSwitchRule = 0,
      AnyOfMany = 1,
      AtMostOne,
      OneOfMany
    };

    enum PropertyPermType
    {
      UnknownPropertyPerm = 0,
      ReadOnly = 1,
      ReadWrite,
      WriteOnly
    };

    // These are the types that a property can be.
    // The order and enumeration of this list is important.
    // Do not add or change enumerations here without adjusting
    // the indexing of the 'allowed attributes' list.
    enum Type
    {
      Unknown = 0,
      BLOB,
      Light,
      Number,
      Switch,
      Text,
    };

  public:
    class Excep : public std::exception
    {
      private:
        Excep() {}

      public:
        explicit Excep(const IndiProperty::Error &tCode ) : m_tCode( tCode ) {}

        ~Excep() throw() {}
        const IndiProperty::Error	&getCode() const
        {
          return m_tCode;
        }
        virtual const char* what() const throw()
        {
          return IndiProperty::getErrorMsg( m_tCode ).c_str();
        }
      private:
        IndiProperty::Error m_tCode;
    };

    // Constructor/copy constructor/destructor.
  public:
    /// Constructor.
    IndiProperty();
    /// Constructor with a type. This will be used often.
    explicit IndiProperty( const Type &tType );

    /// Constructor with a type, device and name. This will be used often.
    IndiProperty( const Type &tType,
                  const std::string &szDevice,
                  const std::string &szName );
    /// Constructor with a type, device, name, state, and perm.
    IndiProperty( const Type &tType,
                  const std::string &szDevice,
                  const std::string &szName,
                  const PropertyStateType &tState,
                  const PropertyPermType &tPerm,
                  const SwitchRuleType &tRule = UnknownSwitchRule );
    /// Copy constructor.
    IndiProperty( const IndiProperty &ipRhs );
    /// Destructor.
    virtual ~IndiProperty();

    // Operators.
  public:
    /// Assigns the internal data of this object from an existing one.
    const IndiProperty &operator= ( const IndiProperty &ipRhs );
    /// This is an alternate way of calling 'setBLOBEnable'.
    const BLOBEnableType &operator= ( const BLOBEnableType &tValue );
    /// Returns true if we have an exact match (value as well).
    bool operator== ( const IndiProperty &ipRhs ) const;
    // Return a reference to an element so it can be modified.
    const IndiElement &operator[] ( const std::string& szName ) const;
    IndiElement &operator[] ( const std::string& szName );
    // Return a reference to an element so it can be modified.
    const IndiElement &operator[] ( const unsigned int& uiIndex ) const;
    IndiElement &operator[] ( const unsigned int& uiIndex );

    // Methods.
  public:
    /// Reset this object.
    void clear();
    /// Compares one property with another instance. The values may not match,
    /// but the type must match, as must the device and name. The names of all
    /// the elements must match as well.
    bool compareProperty( const IndiProperty &ipComp ) const;
    /// Compares one element value contained in this class with another
    /// instance. The type must match, as must the device and name.
    /// The name of this element must match as well.
    bool compareValue( const IndiProperty &ipComp,
                       const std::string &szElementName ) const;
    /// Compares all the element values contained in this class with another
    /// instance. The type must match, as must the device and name. The number
    /// and names of all the elements must match as well.
    bool compareValues( const IndiProperty &ipComp ) const;
    /// Returns a string with each attribute enumerated.
    std::string createString() const;
    /// Create a name for this property based on the device name and the
    /// property name. A '.' is used as the character to join them together.
    /// This key should be unique for all indi devices.
    std::string createUniqueKey() const;

    // A getter for blob enable.
    const BLOBEnableType &getBLOBEnable() const;

    // A getter for each attribute.
    const std::string &getDevice() const;
    const std::string &getGroup() const;
    const std::string &getLabel() const;
    const std::string &getMessage() const;
    const std::string &getName() const;
    const PropertyPermType &getPerm() const;
    const SwitchRuleType &getRule() const;
    const PropertyStateType &getState() const;
    const double &getTimeout() const;
    const pcf::TimeStamp &getTimeStamp() const;
    /// There is only a 'getter' for the type, since it can't be changed.
    const Type &getType() const;
    const std::string &getVersion() const;
    /// Compares one element value contained in this class with another
    /// instance. The type must match, as must the device and name.
    /// The name of this element must match as well, and the value must be a
    /// new, non-blank value.
    bool hasNewValue( const IndiProperty &ipComp,
                      const std::string &szElementName ) const;
    const bool &isRequested() const;

    /// Returns the string type given the enumerated type.
    /// Throws exception if string is not found.
    static std::string convertTypeToString( const Type &tType );
    /// Returns the enumerated type given the tag.
    static Type convertStringToType( const std::string &szTag );
    /// Returns the string type given the enumerated type.
    static std::string getBLOBEnableString( const BLOBEnableType &tType );
    /// Returns the enumerated type given the string type.
    static BLOBEnableType getBLOBEnableType( const std::string &szType );
    /// Returns the string type given the enumerated type.
    static std::string getPropertyPermString( const PropertyPermType &tType );
    /// Returns the enumerated type given the string type.
    static PropertyPermType getPropertyPermType( const std::string &szType );
    /// Returns the string type given the enumerated type.
    static std::string getPropertyStateString( const PropertyStateType &tType );
    /// Returns the enumerated type given the string type.
    static PropertyStateType getPropertyStateType( const std::string &szType );
    /// Returns the string type given the enumerated type.
    static std::string getSwitchRuleString( const SwitchRuleType &tType );
    /// Returns the enumerated type given the string type.
    static SwitchRuleType getSwitchRuleType( const std::string &szType );
    /// Returns the message concerning the error.
    static std::string getErrorMsg( const int &nErr );
    /// Ensures that a name conforms to the INDI standard and can be used as
    /// an identifier. This means:
    ///     1) No ' ' - these will be converted to '_'
    ///     2) No '.' - these will be converted to '___'
    ///     3) No Unprintable chars - these will be converted to '_'
    static std::string scrubName( const std::string &szName );

    /// Returns the number of elements.
    unsigned int getNumElements() const;

    /// Returns true if this contains a valid BLOB-enable value.
    bool hasValidBLOBEnable() const;
    /// Returns true if this contains a non-empty 'device' attribute.
    bool hasValidDevice() const;
    /// Returns true if this contains a non-empty 'group' attribute.
    bool hasValidGroup() const;
    /// Returns true if this contains a non-empty 'label' attribute.
    bool hasValidLabel() const;
    /// Returns true if this contains a non-empty 'message' attribute.
    bool hasValidMessage() const;
    /// Returns true if this contains a non-empty 'name' attribute.
    bool hasValidName() const;
    /// Returns true if this contains a non-empty 'perm' attribute.
    bool hasValidPerm() const;
    /// Returns true if this contains a non-empty 'rule' attribute.
    bool hasValidRule() const;
    /// Returns true if this contains a non-empty 'state' attribute.
    bool hasValidState() const;
    /// Returns true if this contains a non-empty 'timeout' attribute.
    bool hasValidTimeout() const;
    /// Returns true if this contains a non-empty 'timestamp' attribute.
    bool hasValidTimeStamp() const;
    /// Returns true if this contains a non-empty 'version' attribute.
    bool hasValidVersion() const;

    /// All the attribute setters.
    void setBLOBEnable( const BLOBEnableType &tValue );
    void setDevice( const std::string &szValue );
    void setGroup( const std::string &szValue );
    void setLabel( const std::string &szValue );
    void setMessage( const std::string &szValue );
    void setName( const std::string &szValue );
    void setPerm( const PropertyPermType &tValue );
    void setRequested( const bool &oRequested );
    void setRule( const SwitchRuleType &tValue );
    void setState( const PropertyStateType &tValue );
    void setTimeout( const double &xValue );
    void setTimeStamp( const pcf::TimeStamp &tsValue );
    void setVersion( const std::string &szValue );

    // Element functions.
  public:
    // Return a reference to an element so it can be modified.
    const IndiElement &at( const std::string& szName ) const;
    IndiElement &at( const std::string& szName );
    // Return a reference to an element so it can be modified.
    const IndiElement &at( const unsigned int& uiIndex ) const;
    IndiElement &at( const unsigned int& uiIndex );
    /// Adds a new element.
    /// Throws if the element already exists.
    void add( const pcf::IndiElement &ieNew );
    /// Adds an element if it doesn't exist. If it does exist, this is a no-op.
    void addIfNoExist( const pcf::IndiElement &ieNew );
    ///  Returns true if the element 'szElementName' exists, false otherwise.
    bool find( const std::string &szElementName ) const;
    /// Get the entire map of elements.
    const std::map<std::string, pcf::IndiElement> &getElements() const;
    /// Removes an element named 'szElementName'.
    /// Throws if the element doesn't exist.
    void remove( const std::string &szElementName );
    /// Set the entire map of elements.
    void setElements( const std::map<std::string, pcf::IndiElement> &mapElements );
    /// Updates the value of an element named 'szElementName'.
    /// Throws if the element doesn't exist.
    void update( const std::string &szElementName,
                 const pcf::IndiElement &ieUpdate );
    /// Updates the value of an element, adds it if it doesn't exist.
    void update( const pcf::IndiElement &ieNew );

    // Members.
  private:
    
    std::string m_szDevice;
    
    std::string m_szGroup;
    
    std::string m_szLabel;
    
    std::string m_szMessage;
    
    std::string m_szName;
    
    PropertyPermType m_tPerm {UnknownPropertyPerm};

    SwitchRuleType m_tRule {UnknownSwitchRule};
    
    PropertyStateType m_tState {UnknownPropertyState};
    
    double m_xTimeout {0.0f};
    
    // This is a flag which can be used to show that this property
    // has been requested by a client. This is not managed automatically.
    bool m_oRequested {false};
    
    pcf::TimeStamp m_tsTimeStamp;
    
    std::string m_szVersion;

    /// This can also be the value.
    BLOBEnableType m_beValue {UnknownBLOBEnable};
    
    /// A dictionary of elements, indexable by name.
    std::map<std::string, pcf::IndiElement> m_mapElements;

    /// The type of this object. It cannot be changed.
    pcf::IndiProperty::Type m_tType {Unknown};
    
    // A read write lock to protect the internal data.
    mutable pcf::ReadWriteLock m_rwData;

}; // class IndiProperty

} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // INDI_PROPERTY_HPP
