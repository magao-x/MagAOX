/** \file IndiElement.hpp
  *
  * This class represents one element in an INDI property. In its most basic
  * form it is a name-value pair with other attributes associated with it.
  * All access is protected by a read-write lock.
  *
  * @author Paul Grenz (@Steward Observatory, original author)
  * @author Jared Males (@Steward Observatory, refactored for MagAO-X)
  */

#ifndef libcommon_IndiElement_hpp
#define libcommon_IndiElement_hpp


#include <stdint.h>
#include <string>
#include <sstream>
#include <exception>
#include <mutex>
#include <shared_mutex>

/* 2024-04-21 Refactor progress
-- main API rewrite done
-- todo:
   -- switch to format from streams, have single common place for all conversions
   -- tests of conversions
   -- restore template methods for value and light and switch states
   -- implement exclusivity between value and light and switch states based on Type
   -- tests of setting and getting values/states
   -- full comparison operators for value and lightstate and switchstate

*/
namespace pcf
{

namespace internal
{

template<typename TT>
std::string value2string(const TT & value)
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<TT>::digits10 + 1);
    ss << value;
    return ss.str();
}

template<>
inline std::string value2string<std::string>( const std::string & val )
{
    std::string s = val;
    return s;
}

template<>
inline std::string value2string<bool>( const bool & val )
{
    std::stringstream ss;
    ss << std::boolalpha << val;
    return ss.str();
}

template<typename T>
T string2value( const std::string & str)
{
    std::stringstream ss(str);
    T val;
    ss >> std::boolalpha >> val;

    return val;
}

template<>
inline std::string string2value<std::string>( const std::string & str )
{
    std::string val = str;
    return val;
}

}


class IndiElement
{

public:

    // These are the possible types for streaming this element.
  /*  enum class Type
    {
        Unknown = 0,
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
    };*/

    enum class LightState
    {
        Unknown = 0,
        Idle = 1,
        Ok,
        Busy,
        Alert
    };

    enum class SwitchState
    {
        Unknown = 0,
        Off = 1,
        On
    };

    /** \name Member Data
      * @{
      */
protected:

    /// The name of this element.
    std::string m_name;

    /// If this is a number or BLOB, this is the 'printf' format.
    std::string m_format {"%g"};

    /// A label, usually used in a GUI.
    std::string m_label;
    
    /// If this is a number, this is its minimum value.
    std::string m_min {"0"};
    
    /// If this is a number, this is its maximum value.
    std::string m_max {"0"};
    
    /// If this is a number, this is increment for it.
    std::string m_step {"0"};
        
    /// If this is a BLOB, this is the number of bytes for it.
    std::string m_size {"0"};
    
    /// This is the value of the data.
    std::string m_value;
    
    /// This can also be the value.
    LightState m_lightState {LightState::Unknown};
    
    /// This can also be the value.
    SwitchState m_switchState {SwitchState::Unknown};

    // A read write lock to protect the internal data.
    //mutable pcf::ReadWriteLock m_rwData;
    mutable std::shared_mutex m_rwData;

    ///@}

    /** \name Construction and Destruction
      *@{
      */

  public:

    /// Constructor.
    IndiElement();

    /// Constructor with a name. This will be used often.
    IndiElement( const std::string &name );

    /// Constructor with a name and a string value.
    IndiElement( const std::string &name, 
                 const std::string & value 
               );

    /// Constructor with a name and a numeric value
    IndiElement( const std::string &name, 
                 const double & value 
               );

    /// Constructor with a name and a LightState value.
    IndiElement( const std::string &name, 
                 const LightState & value 
               );

    /// Constructor with a name and a SwitchState value.
    IndiElement( const std::string &name, 
                 const SwitchState & value 
               );

    /// Copy constructor.
    IndiElement( const IndiElement &ieRhs );

    /// Destructor.
    virtual ~IndiElement();

    ///@}

    /** \name Member Data Access 
      * @{
      */

    /// Set the element name
    void name( const std::string &name /**< [in] the new name*/);

    /// Get the element name 
    /** \returns the current value of m_name
      */
    const std::string & name() const;

    /// Check if the element name is valid
    /** The name is valid if m_name is non-zero size.
      *
      * \returns true if m_name is valid
      * \returns false if m_name is not valid
      */
    bool hasValidName() const;

    /// Set the element format
    void format( const std::string & format /**< [in] the new format*/);

    /// Get the element format
    /** \returns the current value of m_format
      */
    const std::string & format() const;

    /// Check if the format entry is valid
    /**
      * \returns true if m_format has non-zero size
      * \returns false otherwise
      */
    bool hasValidFormat() const;

    /// Set the element label
    void label( const std::string & label /**< [in] the new label*/);

    /// Get the element label
    /** \returns the current value of m_label
      */
    const std::string & label() const;
    
    /// Check if the label entry is valid
    /**
      * \returns true if m_label has non-zero size
      * \returns false otherwise
      */
    bool hasValidLabel() const;

    /// Set the element's min
    template<typename TT>
    void min( const TT & max /**< [in] the new min*/);
    
    /// Get the element's min
    /** \returns the current value of m_min
      */
    const std::string & min() const;

    /// Get the element's min as type
    /** \returns the current value of m_min as type TT
      */
    template<typename TT>
    const TT & min() const;

    /// Check if the min entry is valid
    /**
      * \returns true if m_min has non-zero size
      * \returns false otherwise
      */
    bool hasValidMin() const;

    /// Set the element max
    template<typename TT>
    void max( const TT & max /**< [in] the new max*/);

    /// Get the element max
    /** \returns the current value of m_max
      */
    const std::string & max() const;

    /// Get the element's max as type
    /** \returns the current value of m_max as type TT
      */
    template<typename TT>
    const TT & max() const;

    /// Check if the max entry is valid
    /**
      * \returns true if m_max has non-zero size
      * \returns false otherwise
      */
    bool hasValidMax() const;

    /// Set the element's step
    template<typename TT>
    void step( const TT & stp /**< [in] the new step*/);

    /// Get the element's step
    /** \returns the current value of m_step
      */
    const std::string & step() const;

    /// Get the element's step as type
    /** \returns the current value of m_step as type TT
      */
    template<typename TT>
    const TT & step() const;

    /// Check if the step entry is valid
    /**
      * \returns true if m_step as non-zero size
      * \returns false otherwise
      */
    bool hasValidStep() const;

    /// Set the element size
    void size( const std::string & size /**< [in] the new size*/);

    /// Set the element size
    void size( const size_t & size /**< [in] the new size*/);

    const std::string & size() const;

    /// Check if the size entry is valid
    /**
      * \returns true if m_size has non-zero size
      * \returns false otherwise
      */
    bool hasValidSize() const;

    /// Set the element's value
    template<typename TT>
    void value( const TT  & val );

    /// Set the element's value
    template <class TT> const TT &operator= ( const TT & val );

    /// Return the value as type string.
    std::string value() const;
    
    /// Return the value as type TT.
    template <class TT> TT value() const;

    /// Check if the value entry is valid
    /**
      * \returns true if m_value as non-zero size
      * \returns false otherwise
      */
    bool hasValidValue() const;

    /// Set the element's light state
    void lightState( const LightState & state /**< [in] the new light state*/);

    /// Set the element's light state
    const LightState & operator= ( const LightState & state /**< [in] the new light state*/ );

    /// Get the element's light state
    /** \returns the current value of m_lightState
      */
    LightState lightState() const;

    /// Check if the light state entry is valid
    /**
      * \returns true if m_lightState is not UnknownLightState
      * \returns false otherwise
      */
    bool hasValidLightState() const;

    /// Set the element's switch state
    void switchState( const SwitchState & state /**< [in] the new switch state*/);

    /// Set the element's switch state
    const SwitchState &operator= ( const SwitchState & state  /**< [in] the new switch state*/);

    /// Get the element's switch state
    /** \returns the current value of m_switchState
      */
    SwitchState switchState() const;

    /// Check if the switch state entry is valid
    /**
      * \returns true if m_switchState is not UnknownSwitchState
      * \returns false otherwise
      */
    bool hasValidSwitchState() const;

    ///@}

    // Operators.
  public:
    /// Assigns the internal data of this object from an existing one.
    const IndiElement &operator= ( const IndiElement &ieRhs );

    /// Returns true if we have an exact match (value as well).
    bool operator==( const IndiElement &ieRhs ) const;

    bool operator==(const std::string & val) const
    {
        return (m_value == val);
    }

    bool operator==(const SwitchState & ss) const
    {
        return (m_switchState == ss);
    }

    bool operator==(const LightState & ls) const
    {
        return (m_lightState == ls);
    }

    template<typename T>
    bool operator==( const T & val ) const;

    // Methods.
  public:
    /// Reset this object.
    virtual void clear();

    /// Returns a string with each attribute & value enumerated.
    std::string createString() const;
    
    /// Returns the string type given the enumerated type.
    static std::string getLightStateString( const LightState &tType );
    
    /// Returns the enumerated type given the string type.
    static LightState getLightState( const std::string &szType );
    
    /// Returns the string type given the enumerated type.
    static std::string getSwitchStateString( const SwitchState &tType );
    
    /// Returns the enumerated type given the string type.
    static SwitchState getSwitchState( const std::string &szType );



}; // class IndiElement

template<typename TT>
void IndiElement::min( const TT  & min )
{
    std::unique_lock wLock(m_rwData);
    m_min = internal::value2string<TT>(min);
}

inline //kept this here instead of cpp for clarity
const std::string & IndiElement::min() const
{
    std::shared_lock rLock(m_rwData);
    return m_min;
}

template<typename TT>
const TT & IndiElement::min() const 
{
    std::shared_lock rLock(m_rwData);
    return internal::string2value<TT>(m_min);
}

template<typename TT>
void IndiElement::max( const TT  & max )
{
    std::unique_lock wLock(m_rwData);
    m_max = internal::value2string<TT>(max);
}

template<typename TT>
const TT & IndiElement::max() const 
{
    std::shared_lock rLock(m_rwData);
    return internal::string2value<TT>(m_max);
}


template<typename TT>
void IndiElement::step( const TT  & stp )
{
    std::unique_lock wLock(m_rwData);
    m_step = internal::value2string<TT>(stp);
}

inline //kept this here instead of cpp for clarity
const std::string &IndiElement::step() const
{
    std::shared_lock rLock(m_rwData);
    return m_step;
}

template <class TT> 
const TT & IndiElement::step() const
{
    std::shared_lock rLock(m_rwData);
    return internal::string2value<TT>(m_step);
}

template<typename TT>
void IndiElement::value( const TT  & val )
{
    std::unique_lock wLock(m_rwData);
    m_value = internal::value2string<TT>(val);
}

inline //kept this here instead of cpp for clarity
std::string IndiElement::value() const
{
    std::shared_lock rLock(m_rwData);
    return m_value;
}

template <class TT> 
TT IndiElement::value() const
{
    std::shared_lock rLock(m_rwData);
    return internal::string2value<TT>(m_value);
}


template <class TT> 
const TT & IndiElement::operator= ( const TT &val )
{
    std::unique_lock wLock(m_rwData);
    m_value = internal::value2string<TT>(val);
    return val;
}





    




} // namespace pcf

#endif // libcommon_IndiElement_hpp
