/** \file Element.cpp
 *
 * Definitions for the Element class.
 *
 * @author Paul Grenz (@Steward Observatory, original author)
 * @author Jared Males (@Steward Observatory, refactored for MagAO-X)
 */

#include <iostream>  // for std::cerr
#include <stdexcept> // for std::runtime_error
#include <stdint.h>
#include <sstream>
#include <exception>

#include "Element.hpp"

namespace xindi
{

Element::Element()
{
}

Element::Element( const Type &type ) : m_type( type )
{
}

Element::Element( const std::string &name ) : m_name( name )
{
}

Element::Element( const Type &type, const std::string &name ) : m_type( type ), m_name( name )
{
}

Element::Element( const Type &type, const std::string &name, const std::string &value ) : m_type( type ), m_name( name )
{
    if( m_type == Type::Switch )
    {
        m_switchState = string2value<Switch>( value );
    }
    else if( m_type == Type::Light )
    {
        m_lightState = string2value<Light>( value );
    }
    else
    {
        m_value = value;
    }
}

Element::Element( const std::string &name, const char * value ) : m_type(Type::Text), m_name( name ), m_value(value)
{
}

Element::Element( const std::string &name, const Switch &value ) : m_type(Type::Switch), m_name( name ), m_switchState(value)
{
}

Element::Element( const std::string &name, const Light &value ) : m_type(Type::Light), m_name( name ), m_lightState(value)
{
}

Element::Element( const Element &ieRhs )
    : m_type( ieRhs.m_type ), m_name( ieRhs.m_name ), m_format( ieRhs.m_format ), m_label( ieRhs.m_label ),
      m_min( ieRhs.m_min ), m_max( ieRhs.m_max ), m_step( ieRhs.m_step ), m_size( ieRhs.m_size ),
      m_value( ieRhs.m_value ), m_lightState( ieRhs.m_lightState ), m_switchState( ieRhs.m_switchState )
{
}

Element::~Element()
{
}

void Element::type( const Type &type )
{
    std::unique_lock wLock( m_rwData );
    m_type = type;
}

const Type &Element::type() const
{
    std::shared_lock rLock( m_rwData );
    return m_type;
}

bool Element::hasValidType() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_type != Type::Unknown );
}

void Element::name( const std::string &name )
{
    std::unique_lock wLock( m_rwData );
    m_name = name;
}

const std::string &Element::name() const
{
    std::shared_lock rLock( m_rwData );
    return m_name;
}

bool Element::hasValidName() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_name.size() > 0 );
}

void Element::format( const std::string &format )
{
    std::unique_lock wLock( m_rwData );
    m_format = format;
}

const std::string &Element::format() const
{
    std::shared_lock rLock( m_rwData );
    return m_format;
}

bool Element::hasValidFormat() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_format.size() > 0 );
}

void Element::label( const std::string &label )
{
    std::unique_lock wLock( m_rwData );
    m_label = label;
}

const std::string &Element::label() const
{
    std::shared_lock rLock( m_rwData );
    return m_label;
}

bool Element::hasValidLabel() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_label.size() > 0 );
}

bool Element::hasValidMin() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_min.size() > 0 );
}

bool Element::hasValidMax() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_max.size() > 0 );
}

bool Element::hasValidStep() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_step.size() > 0 );
}

void Element::size( const std::string &size )
{
    std::unique_lock wLock( m_rwData );
    m_size = size;
}

void Element::size( const size_t &size )
{
    std::unique_lock wLock( m_rwData );

    std::stringstream value;
    value << size;
    m_size = value.str();
}

const std::string &Element::size() const
{
    std::shared_lock rLock( m_rwData );
    return m_size;
}

bool Element::hasValidSize() const
{
    std::shared_lock rLock( m_rwData );
    return ( m_size.size() > 0 );
}

bool Element::hasValidValue() const
{
    std::shared_lock rLock( m_rwData );

    if( m_type == Type::Switch )
    {
        if( m_switchState != Switch::Unknown )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if( m_type == Type::Light )
    {
        if( m_lightState != Light::Unknown )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return ( m_value.size() > 0 );
    }
}


bool Element::operator==( const Light &ls ) const
{
    return ( m_lightState == ls );
}


bool Element::operator==( const Switch &ss ) const
{
    return ( m_switchState == ss );
}

Element &Element::operator=( const Element &ieRhs )
{
    if( &ieRhs != this )
    {
        std::unique_lock wLock( m_rwData );

        m_format      = ieRhs.m_format;
        m_label       = ieRhs.m_label;
        m_max         = ieRhs.m_max;
        m_min         = ieRhs.m_min;
        m_name        = ieRhs.m_name;
        m_size        = ieRhs.m_size;
        m_step        = ieRhs.m_step;
        m_value       = ieRhs.m_value;
        m_lightState  = ieRhs.m_lightState;
        m_switchState = ieRhs.m_switchState;
    }

    return *this;
}

Element &Element::operator=( const std::string &val )
{
    std::unique_lock wLock( m_rwData );
    m_value = val;
    return *this;
}

Element &Element::operator=( const char * val )
{
    std::unique_lock wLock( m_rwData );
    m_value = val;
    return *this;
}

Element &Element::operator=( const Switch &state )
{
    std::unique_lock wLock( m_rwData );
    m_switchState = state;
    return *this;
}

Element &Element::operator=( const Light &state )
{
    std::unique_lock wLock( m_rwData );
    m_lightState = state;
    return *this;
}

bool Element::operator==( const Element &ieRhs ) const
{
    if( &ieRhs == this )
    {
        return true;
    }

    std::shared_lock rLock( m_rwData );

    return ( m_format == ieRhs.m_format && m_label == ieRhs.m_label && m_max == ieRhs.m_max && m_min == ieRhs.m_min &&
             m_name == ieRhs.m_name && m_size == ieRhs.m_size && m_step == ieRhs.m_step && m_value == ieRhs.m_value &&
             m_lightState == ieRhs.m_lightState && m_switchState == ieRhs.m_switchState );
}

void Element::clear()
{
    std::unique_lock wLock( m_rwData );
    m_format      = "%g";
    m_label       = "";
    m_max         = "0";
    m_min         = "0";
    m_name        = "";
    m_size        = "0";
    m_step        = "0";
    m_value       = "";
    m_lightState  = Light::Unknown;
    m_switchState = Switch::Unknown;
}


} // namespace xindi
