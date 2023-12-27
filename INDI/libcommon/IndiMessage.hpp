/// IndiMessage.hpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#ifndef INDI_MESSAGE_HPP
#define INDI_MESSAGE_HPP
#pragma once

#include <string>
#include "IndiProperty.hpp"
#include "IndiElement.hpp"

namespace pcf
{
class IndiMessage
{
  public:
    // These are the types that a message can be.
    enum Type
    {
      Unknown = 0,
      Define,
      Delete,
      EnableBLOB,
      GetProperties,
      Message,
      NewProperty,
      SetProperty,
    };

  // Constructor/copy constructor/destructor.
  public:
    /// Constructor.
    IndiMessage();
    /// Constructor with a type and property. This will be used often.
    IndiMessage( const Type &tType, const pcf::IndiProperty &ipSend );
    /// Copy constructor.
    IndiMessage( const IndiMessage &imRhs );
    /// Destructor.
    virtual ~IndiMessage();

    // Operators.
  public:
    /// Assigns the internal data of this object from an existing one.
    const IndiMessage &operator= ( const IndiMessage &imRhs );

    // Methods.
  public:
    /// Returns the string type given the enumerated type.
    static std::string convertTypeToString( const Type &tType );
    /// Returns the enumerated type given the tag.
    static Type convertStringToType( const std::string &szTag );
    /// The property contained here.
    const pcf::IndiProperty &getProperty() const;
    pcf::IndiProperty &getProperty();
    /// There is only a 'getter' for the type.
    const IndiMessage::Type &getType() const;
    // Set the property.
    void setProperty( const IndiProperty &ipMsg );

    // Members.
  private:
    
    /// The property contained here.
    pcf::IndiProperty m_ipMsg;

    /// The type of this object. It cannot be changed.
    Type m_tType {Unknown};

}; // class IndiMessage
} // namespace pcf

////////////////////////////////////////////////////////////////////////////////

#endif // INDI_MESSAGE_HPP
