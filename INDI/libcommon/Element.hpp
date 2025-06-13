/** \file Element.hpp
 *
 * Declarations for the xindi::Element class.
 *
 * @author Paul Grenz (@Steward Observatory, original author)
 * @author Jared Males (@Steward Observatory, refactored for MagAO-X)
 */

#ifndef libxindi_Element_hpp
#define libxindi_Element_hpp

#include <mutex>
#include <shared_mutex>
#include <complex>

#include "Types.hpp"

/* 2024-04-26 Refactor progress
-- main API rewrite done
-- todo:
   -- tests of conversions
   -- implement exclusivity between value and light and switch states based on Type
   -- tests of setting and getting values/states

*/
namespace xindi
{

/** One element in an INDI property
 *
 * This class represents one element in an INDI property. In its most basic
 * form it is a name-value pair with other attributes associated with it.
 *
 * An element has a \ref Type, which must match the \ref Property to which it belongs.  This \ref Type
 * is used to enforce type-correct access to the stored value.
 *
 * The type can be accessed using `Element::value<TT>()` which returns the value as type `TT`.  The value can
 * be changed with `Element::value(const type &)`.
 *
 * All access is protected by a read-write lock.
 *
 */
class Element
{

  public:
    /** \name Member Data
     * @{
     */
  protected:
    /// The type of this element
    Type m_type{ Type::Unknown };

    /// The name of this element.
    std::string m_name;

    /// If this is a number or BLOB, this is the 'printf' format.
    std::string m_format;

    /// A label, usually used in a GUI.
    std::string m_label;

    /// If this is a number, this is its minimum value.
    std::string m_min;

    /// If this is a number, this is its maximum value.
    std::string m_max;

    /// If this is a number, this is the increment for it.
    std::string m_step;

    /// If this is a BLOB, this is the number of bytes for it.
    std::string m_size;

    /// This is the value of the data.
    std::string m_value;

    /// This can also be the value.
    Light m_lightState{ Light::Unknown };

    /// This can also be the value.
    Switch m_switchState{ Switch::Unknown };

    // A read write lock to protect the internal data.
    mutable std::shared_mutex m_rwData;

    ///@}

    /** \name Construction and Destruction
     *@{
     */

  public:
    /// Default Constructor.
    /**
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element();

    /// Constructor with a type.
    /**
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const Type &type /**< [in] the \ref Type of this element*/ );

    /// Constructor with a name.
    /**
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const std::string &name /**< [in] the name of this element */ );

    /// Constructor with a type and a name.
    /**
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const Type        &type, /**< [in] the \ref Type of this element*/
             const std::string &name /**< [in] the name of this element*/ );

    /// Constructor with name and a string value.
    /** This allows construction with the string representation without a conversion.
     *
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const Type        &type, /**< [in] the \ref Type of this element*/
             const std::string &name, /**< [in] the name of this element*/
             const std::string &value /**< [in] the string representation of the value*/ );

    /// Constructor with a name and a value
    /** m_type will be set according to TT. This handles std::string, char *, and numeric types.
     * See overloads for `const char *`, \ref Light, and \ref Switch.
     *
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    template <typename TT>
    Element( const std::string &name, /**< [in] the name of this element*/
             const TT          &value /**< [in] the value of this element*/ );

    /// Constructor with a name and a const char * value
    /** m_type will be set to Type::Text.  This overload is
     * needed to prevent template resolution of `const char *` to pointer addresss (Number).
     *
     * \overload
     *
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const std::string &name, /**< [in] the name of this element*/
             const char        *value /**< [in] the string value*/ );

    /// Constructor with a name and a Switch value
    /** m_type will be set to Type::Switch.
     *
     * \overload
     *
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const std::string &name, /**< [in] the name of this element*/
             const Switch      &value /**< [in] the \ref Switch state*/ );

    /// Constructor with a name and a Light value
    /** m_type will be set to Type::Light.
     *
     * \overload
     *
     * \test Construction of xindi::Element \ref tests_xindi_Element_construction "[test doc]"
     */
    Element( const std::string &name, /**< [in] the name of this element*/
             const Light       &value /**< [in] the \ref Light state*/ );

    /// Copy constructor.
    Element( const Element &ieRhs /**< [in] the existing Element to copy*/ );

    /// Destructor.
    virtual ~Element();

    ///@}

    /** \name Member Data Access
     * @{
     */

    /// Set the element type
    void type( const Type &type /**< [in] the new type*/ );

    /// Get the element type
    /** \returns the current value of m_type
     */
    const Type &type() const;

    /// Check if the element name is valid
    /** The name is valid if m_name is non-zero size.
     *
     * \returns true if m_name is valid
     * \returns false if m_name is not valid
     */
    bool hasValidType() const;

    /// Set the element name
    void name( const std::string &name /**< [in] the new name*/ );

    /// Get the element name
    /** \returns the current value of m_name
     */
    const std::string &name() const;

    /// Check if the element name is valid
    /** The name is valid if m_name is non-zero size.
     *
     * \returns true if m_name is valid
     * \returns false if m_name is not valid
     */
    bool hasValidName() const;

    /// Set the element format
    void format( const std::string &format /**< [in] the new format*/ );

    /// Get the element format
    /** \returns the current value of m_format
     */
    const std::string &format() const;

    /// Check if the format entry is valid
    /**
     * \returns true if m_format has non-zero size
     * \returns false otherwise
     */
    bool hasValidFormat() const;

    /// Set the element label
    void label( const std::string &label /**< [in] the new label*/ );

    /// Get the element label
    /** \returns the current value of m_label
     */
    const std::string &label() const;

    /// Check if the label entry is valid
    /**
     * \returns true if m_label has non-zero size
     * \returns false otherwise
     */
    bool hasValidLabel() const;

    /// Set the element's min
    template <typename TT>
    void min( const TT &max /**< [in] the new min*/ );

    /// Get the element's min
    /** \returns the current value of m_min
     */
    const std::string &min() const;

    /// Get the element's min as type
    /** \returns the current value of m_min as type TT
     */
    template <typename TT>
    const TT &min() const;

    /// Check if the min entry is valid
    /**
     * \returns true if m_min has non-zero size
     * \returns false otherwise
     */
    bool hasValidMin() const;

    /// Set the element max
    template <typename TT>
    void max( const TT &max /**< [in] the new max*/ );

    /// Get the element max
    /** \returns the current value of m_max
     */
    const std::string &max() const;

    /// Get the element's max as type
    /** \returns the current value of m_max as type TT
     */
    template <typename TT>
    const TT &max() const;

    /// Check if the max entry is valid
    /**
     * \returns true if m_max has non-zero size
     * \returns false otherwise
     */
    bool hasValidMax() const;

    /// Set the element's step
    template <typename TT>
    void step( const TT &stp /**< [in] the new step*/ );

    /// Get the element's step
    /** \returns the current value of m_step
     */
    const std::string &step() const;

    /// Get the element's step as type
    /** \returns the current value of m_step as type TT
     */
    template <typename TT>
    const TT &step() const;

    /// Check if the step entry is valid
    /**
     * \returns true if m_step as non-zero size
     * \returns false otherwise
     */
    bool hasValidStep() const;

    /// Set the element size
    void size( const std::string &size /**< [in] the new size*/ );

    /// Set the element size
    void size( const size_t &size /**< [in] the new size*/ );

    /// Get the element's size
    /**
     *  \returns a reference to the size string
     */
    const std::string &size() const;

    /// Check if the size entry is valid
    /**
     * \returns true if m_size has non-zero size
     * \returns false otherwise
     */
    bool hasValidSize() const;

    /// Get the value as a string independent of type
    /**
     * \returns the current \ref m_value
     */
    std::string valueStr() const;

    /// Get the value as a string
    void getValue( std::string &str /**< [out] the value as a string */ ) const;

    /// Get the value as a Light
    void getValue( Light &lst /**< [out] the value as a Light */ ) const;

    /// Get the value as a Switch
    void getValue( Switch &sst /**< [out] the value as a Switch */ ) const;

    /// Get the value as an arbitrary type TT
    template <class TT>
    void getValue( TT &val /**< [out] the value as a TT */ ) const;

    /// Set the element's value
    template <typename TT>
    void value( const TT &val );

    /// Return the value as type string.
    std::string value() const;

    /// Return the value as an arbitrary type TT.
    template <class TT>
    TT value() const;

    /// Check if the value entry is valid
    /**
     * \returns true if m_value as non-zero size
     * \returns false otherwise
     */
    bool hasValidValue() const;

    ///@}

    /** \name Comparison
     * @{
     */

    /// Compare to another Element
    /** \returns true if we have an exact match (value as well).
     *
     */
    bool operator==( const Element &ieRhs /**< [in] the element to compare to*/) const;

    /// Compare the string representation of the value
    /** \returns true if the value is an exact match as a string
     */
    bool operator==( const std::string &val /**< [in] */) const;

    /// Compare the string representation of the value
    /** \returns true if the value is an exact match as a string
     */
    bool operator==( const char * val /**< [in] */) const;

    /// Compare the value numerically
    /** \returns true if the value is an exact match as type TT
     */
    template <typename T>
    bool operator==( const T &val /**< [in] */) const;

    /// Compare as a \ref Light
    /** \returns true if the value is an exact match as a \ref Light
     */
    bool operator==( const Light &ls /**< [in] */) const;

    // Compare as a \ref Switch
    /** \returns true if the value is an exact match as a \ref Switch
     */
    bool operator==( const Switch &ss /**< [in] */) const;

    ///@}

    /** \name Assignment
     * @{
     */

    /// Assigns the internal data of this object from an existing one.
    /**
     * \returns a reference to `this`
     *
     */
    Element &operator=( const Element &ieRhs /**< [in] */);

    /// Set the element's numeric value
    /**
     * \returns a reference to `this`
     *
     * \test Assignment of values to xindi::Element \ref tests_xindi_Element_value_assignment "[test doc]"
     */
    template<typename TT>
    Element & operator=( const TT & val /**< [in] the new value*/ );

    /// Set the element's string value
    /**
     * \returns a reference to `this`
     *
     * \test Assignment of values to xindi::Element \ref tests_xindi_Element_value_assignment "[test doc]"
     */
    Element &operator=( const std::string &val /**< [in] the new value*/ );

    /// Set the element's string value
    /**
     * \returns a reference to `this`
     *
     * \test Assignment of values to xindi::Element \ref tests_xindi_Element_value_assignment "[test doc]"
     */
    Element & operator=( const char * val /**< [in] the new value*/ );

    /// Set the element's switch state
    /**
     * \returns a reference to `this`
     *
     * \test Assignment of values to xindi::Element \ref tests_xindi_Element_value_assignment "[test doc]"
     */
    Element &operator=( const Switch &state /**< [in] the new switch state*/ );

    /// Set the element's light state
    /**
     * \returns a reference to `this`
     *
     * \test Assignment of values to xindi::Element \ref tests_xindi_Element_value_assignment "[test doc]"
     */
    Element &operator=( const Light &state /**< [in] the new light state*/ );

    ///@}

    /// Reset this object.
    virtual void clear();

}; // class Element

template <typename TT>
Element::Element( const std::string &name, const TT &value ) : m_name( name )
{
    m_type  = type2Type<TT>();
    m_value = value2string<TT>( value );
}

template <typename TT>
void Element::min( const TT &min )
{
    std::unique_lock wLock( m_rwData );
    m_min = value2string<TT>( min );
}

template <typename TT>
const TT &Element::min() const
{
    std::shared_lock rLock( m_rwData );
    return string2value<TT>( m_min );
}

template <typename TT>
void Element::max( const TT &max )
{
    std::unique_lock wLock( m_rwData );
    m_max = value2string<TT>( max );
}

template <typename TT>
const TT &Element::max() const
{
    std::shared_lock rLock( m_rwData );
    return string2value<TT>( m_max );
}

template <typename TT>
void Element::step( const TT &stp )
{
    std::unique_lock wLock( m_rwData );
    m_step = value2string<TT>( stp );
}

template <class TT>
const TT &Element::step() const
{
    std::shared_lock rLock( m_rwData );
    return string2value<TT>( m_step );
}

template <class TT>
void Element::getValue( TT &val ) const
{
    std::shared_lock rLock( m_rwData );
    val = string2value<TT>( m_value );
}

template <typename TT>
void Element::value( const TT &val )
{
    std::unique_lock wLock( m_rwData );
    m_value = value2string<TT>( val );
}

template <class TT>
TT Element::value() const
{
    std::shared_lock rLock( m_rwData );

    TT val;

    getValue( val );

    return val;
}

inline bool Element::operator==( const std::string &val ) const
{
    return ( m_value == val );
}

template <class TT>
Element &Element::operator=( const TT &val )
{
    std::unique_lock wLock( m_rwData );
    m_value = value2string<TT>( val );
    return *this;
}

} // namespace xindi

#endif // libxindi_Element_hpp
