/// IndiElement.hpp
///
/// @author Paul Grenz
///
/// This class represents one element in an INDI property. In its most basic
/// form it is a name-value pair with other attributes associated with it.
/// All access is protected by a read-write lock.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef INDI_ELEMENT_HPP
#define INDI_ELEMENT_HPP
#pragma once

#include <stdint.h>
#include <string>
#include <sstream>
#include <exception>
#include "ReadWriteLock.hpp"

namespace pcf
{
class IndiElement
{
  public:
    // These are the possible types for streaming this element.
    enum Type
    {
      UnknownType = 0,
      // Define properties.
      DefBLOB,
      DefLight,
      DefNumber,
      DefSwitch,
      DefText,
      // Used to set or update.
      OneBLOB,
      OneLight,
      OneNumber,
      OneSwitch,
      OneText,
    };

    enum Error
    {
      ErrNone =                    0,
      ErrAttributeIsNotAllowed =  -1,
      ErrTypeNotDefined =         -2,
      ErrIncorrectType =          -3,
      ErrUndefined =           -9999
    };

    enum LightStateType
    {
      UnknownLightState = 0,
      Idle = 1,
      Ok,
      Busy,
      Alert
    };

    enum SwitchStateType
    {
      UnknownSwitchState = 0,
      Off = 1,
      On
    };

    // Constructor/copy constructor/destructor.
  public:
    /// Constructor.
    IndiElement();

    /// Constructor with a name. This will be used often.
    IndiElement( const std::string &szName );

    /// Constructor with a name and a string value.
    IndiElement( const std::string &szName, const std::string &szValue );

    /// Constructor with a name and a char pointer value.
    IndiElement( const std::string &szName, const char *pcValue );

    /// Constructor with a name and a TT value.
    template <class TT> IndiElement( const std::string &szName, const TT &tValue );

    /// Constructor with a name and a LightStateType value.
    IndiElement( const std::string &szName, const LightStateType &tValue );

    /// Constructor with a name and a SwitchStateType value.
    IndiElement( const std::string &szName, const SwitchStateType &tValue );

    /// Copy constructor.
    IndiElement( const IndiElement &ieRhs );

    /// Destructor.
    virtual ~IndiElement();

    // Operators.
  public:
    /// Assigns the internal data of this object from an existing one.
    const IndiElement &operator= ( const IndiElement &ieRhs );
    /// This is an alternate way of calling 'setLightState'.
    const LightStateType &operator= ( const LightStateType &tValue );
    /// This is an alternate way of calling 'setSwitchState'.
    const SwitchStateType &operator= ( const SwitchStateType &tValue );
    /// Returns true if we have an exact match (value as well).
    bool operator==( const IndiElement &ieRhs ) const;
    /// This is an alternate way of calling 'setValue'.
    template <class TT> const TT &operator= ( const TT &tValue );

    // We want the value as a different things.
    operator LightStateType() const;
    operator SwitchStateType() const;

    // Methods.
  public:
    /// Reset this object.
    virtual void clear();
    /// Returns a string with each attribute & value enumerated.
    std::string createString() const;

    // All the different data this can store.
    const std::string &getFormat() const;
    
    const std::string &getLabel() const;
    
    const std::string &getMax() const;
    
    const std::string &getMin() const;
    
    const std::string &getName() const;
    
    const std::string &getSize() const;
    
    const std::string &getStep() const;

    /// Different ways of getting the data in this element.
    LightStateType getLightState() const;
    SwitchStateType getSwitchState() const;
    /// Return the value as type string.
    std::string get() const;
    /// Return the value as type string.
    std::string getValue() const;
    void getValue( char *pcValue, unsigned int &uiSize ) const;
    /// Return the value as type TT.
    template <class TT> TT getValue() const;
    /// Return the value as type TT.
    template <class TT> TT get() const;

    // Are the entries valid (non zero size)?
    bool hasValidFormat() const;
    bool hasValidLabel() const;
    bool hasValidLightState() const;
    bool hasValidMax() const;
    bool hasValidMin() const;
    bool hasValidName() const;
    bool hasValidSize() const;
    bool hasValidStep() const;
    bool hasValidSwitchState() const;
    bool hasValidValue() const;

    // Is the value (not LightState or SwitchState) a numeric?
    bool isNumeric() const;

    /// Returns the string type given the enumerated type.
    static std::string convertTypeToString( const Type &tType );
    /// Returns the enumerated type given the tag.
    static Type convertStringToType( const std::string &szTag );
    /// Returns the string type given the enumerated type.
    static std::string getLightStateString( const LightStateType &tType );
    /// Returns the enumerated type given the string type.
    static LightStateType getLightStateType( const std::string &szType );
    /// Returns the string type given the enumerated type.
    static std::string getSwitchStateString( const SwitchStateType &tType );
    /// Returns the enumerated type given the string type.
    static SwitchStateType getSwitchStateType( const std::string &szType );

    void setFormat( const std::string &szFormat );
    void setLabel( const std::string &szValue );
    void setMax( const std::string &szMax );
    void setMin( const std::string &szMin );
    void setName( const std::string &szName );
    void setSize( const std::string &szSize );
    void setStep( const std::string &szStep );

    template <class TT> void setMax( const TT &ttMax );
    template <class TT> void setMin( const TT &ttMin );
    template <class TT> void setSize( const TT &ttSize );
    template <class TT> void setStep( const TT &ttStep );

    /// Different ways of setting the data in this element.
    void setLightState( const LightStateType &tValue );
    void setSwitchState( const SwitchStateType &tValue );
    void setValue( const std::string &szValue );
    void setValue( const char *pcValue, const unsigned int &uiSize );
    template <class TT> void setValue( const TT &ttValue );
    template <class TT> void set( const TT &ttValue );

    // Members.
  private:
    /// If this is a number or BLOB, this is the 'printf' format.
    std::string m_szFormat {"%g"};

    /// A label, usually used in a GUI.
    std::string m_szLabel;
    
    /// If this is a number, this is its maximum value.
    std::string m_szMax {"0"};
    
    /// If this is a number, this is its minimum value.
    std::string m_szMin {"0"};
    
    /// The name of this element.
    std::string m_szName;
    
    /// If this is a BLOB, this is the number of bytes for it.
    std::string m_szSize {"0"};
    
    /// If this is a number, this is increment for it.
    std::string m_szStep {"0"};
    
    /// This is the value of the data.
    std::string m_szValue;
    
    /// This can also be the value.
    LightStateType m_lsValue {UnknownLightState};
    
    /// This can also be the value.
    SwitchStateType m_ssValue {UnknownSwitchState};
    
    // A read write lock to protect the internal data.
    mutable pcf::ReadWriteLock m_rwData;

}; // class IndiElement
} // namespace pcf


////////////////////////////////////////////////////////////////////////////////
/// Constructor with a name and a TT value.

template <class TT> pcf::IndiElement::IndiElement( const std::string &szName,
                                                   const TT &ttValue ) : m_szName(szName)
{
  std::stringstream ssValue;
  ssValue << std::boolalpha << ttValue;
  m_szValue = ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Get the value as type TT.

template <class TT> TT pcf::IndiElement::get() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  TT tValue;
  //  stream the data into the variable.
  std::stringstream ssValue( m_szValue );
  ssValue >> std::boolalpha >> tValue;
  return tValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Get an value of type TT from the element.

template <class TT> TT pcf::IndiElement::getValue() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  TT tValue;
  //  stream the data into the variable.
  std::stringstream ssValue( m_szValue );
  ssValue >> std::boolalpha >> tValue;
  return tValue;
}


////////////////////////////////////////////////////////////////////////////////
///  set an value of type TT in the element using the "=" operator.

template <class TT> const TT &pcf::IndiElement::operator= ( const TT &ttValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttValue;
  m_szValue = ssValue.str();
  return ttValue;
}

////////////////////////////////////////////////////////////////////////////////
///  set the value from type TT.

template <class TT> void pcf::IndiElement::set( const TT &ttValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttValue;
  m_szValue = ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
///  set an value of type TT in the element.

template <class TT> void pcf::IndiElement::setValue( const TT &ttValue )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttValue;
  m_szValue = ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
///  set a max of type TT in the element.

template <class TT> void pcf::IndiElement::setMax( const TT &ttMax )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttMax;
  m_szMax = ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
///  set a max of type TT in the element.

template <class TT> void pcf::IndiElement::setMin( const TT &ttMin )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttMin;
  m_szMin = ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////
///  set a size of type TT in the element.

template <class TT> void pcf::IndiElement::setSize( const TT &ttSize )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttSize;
  m_szSize = ssValue.str();
}

/*
////////////////////////////////////////////////////////////////////////////////
/// Get the size as an arbitrary type.
template <class TT> typename std::remove_reference<TT>::type pcf::IndiElement::getSize() const
{
  pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );

  typename std::remove_reference<TT>::type tValue;
  //  stream the size into the variable.
  std::stringstream ssValue( m_szSize );
  ssValue >> std::boolalpha >> tValue;
  return tValue;
}*/


////////////////////////////////////////////////////////////////////////////////
///  set a step of type TT in the element.

template <class TT> void pcf::IndiElement::setStep( const TT &ttStep )
{
  pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );

  std::stringstream ssValue;
  ssValue.precision( 15 );
  ssValue << std::boolalpha << ttStep;
  m_szStep = ssValue.str();
}

////////////////////////////////////////////////////////////////////////////////

#endif // INDI_ELEMENT_HPP
