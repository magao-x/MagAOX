/** \file IndiElement.cpp
  *
  *
  * @author Paul Grenz (@Steward Observatory, original author)
  * @author Jared Males (@Steward Observatory, refactored for MagAO-X)
  */

#include <string>
#include <sstream>
#include <iostream>           // for std::cerr
#include <stdexcept>          // for std::runtime_error
#include <cstring>
#include "IndiElement.hpp"

namespace pcf
{

IndiElement::IndiElement()
{
}

IndiElement::IndiElement( const std::string &name ) : m_name(name)
{
}

IndiElement::IndiElement( const std::string &name,
                          const std::string &value ) : m_name(name), m_value(value)
{
}

IndiElement::IndiElement( const std::string &name,
                          const double & value ) : m_name(name)
{
    std::stringstream ssValue;
    ssValue << std::boolalpha << value;
    m_value = ssValue.str();
}

IndiElement::IndiElement( const std::string &name,
                          const LightState &value ) : m_name(name), m_lightState(value)
{
}

IndiElement::IndiElement( const std::string &name,
                          const SwitchState &value ) : m_name(name), m_switchState(value)
{
}

IndiElement::IndiElement( const IndiElement &ieRhs ) : m_format(ieRhs.m_format), m_label(ieRhs.m_label),
                                                         m_max(ieRhs.m_max), m_min(ieRhs.m_min), m_name(ieRhs.m_name),
                                                          m_size(ieRhs.m_size), m_step(ieRhs.m_step), m_value(ieRhs.m_value),
                                                           m_lightState(ieRhs.m_lightState), m_switchState(ieRhs.m_switchState)
{
}

IndiElement::~IndiElement()
{
}

void IndiElement::format( const std::string & format )
{
    std::unique_lock wLock(m_rwData);
    m_format = format;
}

const std::string & IndiElement::format() const
{
    std::shared_lock rLock(m_rwData);
    return m_format;
}

bool IndiElement::hasValidFormat() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_format.size() > 0 );
}

void IndiElement::label( const std::string & label )
{
    std::unique_lock wLock(m_rwData);
    m_label = label;
}

const std::string & IndiElement::label() const
{
    std::shared_lock rLock(m_rwData);  
    return m_label;
}

bool IndiElement::hasValidLabel() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_label.size() > 0 );
}

void IndiElement::min( const std::string & min )
{
    std::unique_lock wLock(m_rwData);
    m_min = min;
}

void IndiElement::min( const double & min )
{
    std::unique_lock wLock(m_rwData);

    std::stringstream ssValue;
    ssValue.precision( 15 );
    ssValue << min;
    m_min = ssValue.str();
}

const std::string &IndiElement::min() const
{
    std::shared_lock rLock(m_rwData);
    return m_min;
}

bool IndiElement::hasValidMin() const
{
    std::shared_lock rLock(m_rwData);    
    return ( m_min.size() > 0 );
}

void IndiElement::max( const std::string & max )
{
    std::unique_lock wLock(m_rwData);
    m_max = max;
}

void IndiElement::max( const double & max )
{
    std::unique_lock rLock(m_rwData);

    std::stringstream ssValue;
    ssValue.precision( 15 );
    ssValue << max;
    m_max = ssValue.str();
}

const std::string &IndiElement::max() const
{
    std::shared_lock rLock(m_rwData);
    return m_max;
}

bool IndiElement::hasValidMax() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_max.size() > 0 );
}

void IndiElement::name( const std::string &name )
{
    std::unique_lock wLock(m_rwData);
    m_name = name;
}

const std::string &IndiElement::name() const
{
    std::shared_lock rLock(m_rwData);
    return m_name;
}

bool IndiElement::hasValidName() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_name.size() > 0 );
}

void IndiElement::size( const std::string & size )
{
    std::unique_lock wLock(m_rwData);
    m_size = size;
}

void IndiElement::size( const double & size )
{
    std::unique_lock wLock(m_rwData);

    std::stringstream value;
    value.precision( 15 );
    value << size;
    m_size = value.str();
}

const std::string & IndiElement::size() const
{
    std::shared_lock rLock(m_rwData);  
    return m_size;
}


bool IndiElement::hasValidSize() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_size.size() > 0 );
}

void IndiElement::step( const std::string & step )
{
    std::unique_lock wLock(m_rwData);
    m_step = step;
}

void IndiElement::step( const double & step )
{
    std::unique_lock wLock(m_rwData);

    std::stringstream value;
    value.precision( 15 );
    value << step;
    m_step = value.str();
}

const std::string &IndiElement::step() const
{
    std::shared_lock rLock(m_rwData);
    return m_step;
}

bool IndiElement::hasValidStep() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_step.size() > 0 );
}

void IndiElement::value( const std::string & val )
{
    std::unique_lock wLock(m_rwData);    
    m_value = val;
}

void IndiElement::value( const double & val )
{
    std::unique_lock wLock(m_rwData);   
    
    std::stringstream value;
    value.precision( 15 );
    value << std::boolalpha << val;
    m_value = value.str();
}

std::string IndiElement::value() const
{
    std::shared_lock rLock(m_rwData);
    return m_value;
}

bool IndiElement::hasValidValue() const
{
    //pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
    std::shared_lock rLock(m_rwData);

    return ( m_value.size() > 0 );
}

void IndiElement::lightState( const LightState & state )
{
    std::unique_lock wLock(m_rwData);
    m_lightState = state;
}

const IndiElement::LightState & IndiElement::operator=( const LightState & state )
{
    std::unique_lock wLock(m_rwData);
    m_lightState = state;
    return m_lightState;
}

IndiElement::LightState IndiElement::lightState() const
{
    std::shared_lock rLock(m_rwData);
    return m_lightState;
}

bool IndiElement::hasValidLightState() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_lightState != LightState::Unknown );
}

void IndiElement::switchState( const SwitchState & state )
{
    std::unique_lock wLock(m_rwData);  
    m_switchState = state;
}

const IndiElement::SwitchState & IndiElement::operator=( const SwitchState & state )
{
    std::unique_lock wLock(m_rwData);  
    m_switchState = state;
    return m_switchState;
}

IndiElement::SwitchState IndiElement::switchState() const
{
    std::shared_lock rLock(m_rwData);
    return m_switchState;
}

bool IndiElement::hasValidSwitchState() const
{
    std::shared_lock rLock(m_rwData);
    return ( m_switchState != SwitchState::Unknown );
}












const IndiElement &IndiElement::operator=( const IndiElement &ieRhs )
{
    if ( &ieRhs != this )
    {
        //pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
        std::unique_lock wLock(m_rwData);

        m_format = ieRhs.m_format;
        m_label = ieRhs.m_label;
        m_max = ieRhs.m_max;
        m_min = ieRhs.m_min;
        m_name = ieRhs.m_name;
        m_size = ieRhs.m_size;
        m_step = ieRhs.m_step;
        m_value = ieRhs.m_value;
        m_lightState = ieRhs.m_lightState;
        m_switchState = ieRhs.m_switchState;
    }
    
    return *this;
}


bool IndiElement::operator==( const IndiElement &ieRhs ) const
{
    if ( &ieRhs == this )
    {
        return true;
    }

    //pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
    std::shared_lock rLock(m_rwData);

    return ( m_format == ieRhs.m_format &&
             m_label == ieRhs.m_label &&
             m_max == ieRhs.m_max &&
             m_min == ieRhs.m_min &&
             m_name == ieRhs.m_name &&
             m_size == ieRhs.m_size &&
             m_step == ieRhs.m_step &&
             m_value == ieRhs.m_value &&
             m_lightState == ieRhs.m_lightState &&
             m_switchState == ieRhs.m_switchState );
}

std::string IndiElement::createString() const
{
    //pcf::ReadWriteLock::AutoRLock rwAuto( &m_rwData );
    std::shared_lock rLock(m_rwData);

    std::stringstream ssOutput;
    ssOutput << "{ "
             << "\"name\" : \"" << m_name << "\" , "
             << "\"value\" : \"" << m_value << "\" , "
             << "\"lightstate\" : \"" << getLightStateString( m_lightState ) << "\" , "
             << "\"switchstate\" : \"" << getSwitchStateString( m_switchState ) << "\" , "
             << "\"label\" : \"" << m_label << "\" , "
             << "\"format\" : \"" << m_format << "\" , "
             << "\"max\" : \"" << m_max << "\" , "
             << "\"min\" : \"" << m_min << "\" , "
             << "\"size\" : \"" << m_size << "\" , "
             << "\"step\" : \"" << m_step << "\" "
             << "} ";
    return ssOutput.str();
}



void IndiElement::clear()
{
    //pcf::ReadWriteLock::AutoWLock rwAuto( &m_rwData );
    std::unique_lock wLock(m_rwData);
    m_format = "%g";
    m_label = "";
    m_max = "0";
    m_min = "0";
    m_name = "";
    m_size = "0";
    m_step = "0";
    m_value = "";
    m_lightState = LightState::Unknown;
    m_switchState = SwitchState::Unknown;
}








IndiElement::LightState IndiElement::getLightState( const std::string &szType )
{

    LightState tType = LightState::Unknown;

    if ( szType == "Idle" )
        tType = LightState::Idle;
    else if ( szType == "Ok" )
        tType = LightState::Ok;
    else if ( szType == "Busy" )
        tType = LightState::Busy;
    else if ( szType == "Alert" )
        tType = LightState::Alert;

    return tType;
}

std::string IndiElement::getLightStateString( const LightState &tType )
{

  std::string szType = "";

  switch ( tType )
  {
    case LightState::Unknown:
      szType = "";
      break;
    case LightState::Idle:
      szType = "Idle";
      break;
    case LightState::Ok:
      szType = "Ok";
      break;
    case LightState::Busy:
      szType = "Busy";
      break;
    case LightState::Alert:
      szType = "Alert";
      break;
  }

  return szType;
}

IndiElement::SwitchState IndiElement::getSwitchState( const std::string &szType )
{

  SwitchState tType = SwitchState::Unknown;

  if ( szType == "Off" )
    tType = SwitchState::Off;
  else if ( szType == "On" )
    tType = SwitchState::On;

  return tType;
}

std::string IndiElement::getSwitchStateString( const SwitchState &tType )
{

  std::string szType = "";

  switch ( tType )
  {
    case SwitchState::Unknown:
      szType = "";
      break;
    case SwitchState::Off:
      szType = "Off";
      break;
    case SwitchState::On:
      szType = "On";
      break;
  }

  return szType;
}

std::string IndiElement::convertTypeToString( const Type &tType )
{
    
    std::string szType = "UnknownType";

    switch ( tType )
    {
        case Type::Unknown:
            szType = "";
            break;

        // Define properties.
        case Type::DefBLOB:
            szType = "defBLOB";
            break;
        case Type::DefLight:
            szType = "defLight";
            break;
        case Type::DefNumber:
            szType = "defNumber";
            break;
        case Type::DefSwitch:
            szType = "defSwitch";
            break;
        case Type::DefText:
            szType = "defText";
            break;

        // Update or set properties.
        case Type::OneBLOB:
            szType = "oneBLOB";
            break;
        case Type::OneLight:
            szType = "oneLight";
            break;
        case Type::OneNumber:
            szType = "oneNumber";
            break;
        case Type::OneSwitch:
            szType = "oneSwitch";
            break;
        case Type::OneText:
            szType = "oneText";
            break;
  }

  return szType;
}

IndiElement::Type IndiElement::convertStringToType( const std::string &szTag )
{
    Type tType = Type::Unknown;

    // Define properties.
    if ( szTag == "defBLOB" )
    {
        tType = Type::DefBLOB;
    }
    else if ( szTag == "defLight" )
    {
        tType = Type::DefLight;
    }
    else if ( szTag == "defNumber" )
    {
        tType = Type::DefNumber;
    }
    else if ( szTag == "defSwitch" )
    {
        tType = Type::DefSwitch;
    }
    else if ( szTag == "defText" )
    {
        tType = Type::DefText;
    }
    else if ( szTag == "oneBLOB" )
    {
        tType = Type::OneBLOB;
    }
    else if ( szTag == "oneLight" )
    {
        tType = Type::OneLight;
    }
    else if ( szTag == "oneNumber" )
    {    
        tType = Type::OneNumber;
    }
    else if ( szTag == "oneSwitch" )
    {
        tType = Type::OneSwitch;
    }
    else if ( szTag == "oneText" )
    {    
        tType = Type::OneText;
    }

    return tType;
}

}//namespace pcf

