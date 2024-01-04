/** \file indiCompRules.hpp
  * \brief The rules for the MagAO-X stateRuleEngine
  *
  * \ingroup stateRuleEngine_files
  */

#ifndef stateRuleEngine_indiCompRules_hpp
#define stateRuleEngine_indiCompRules_hpp 

#include <variant>
#include <mx/mxException.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
                                         //Included here for standalone testing of this file

/// Logical comparisons for the INDI rules
enum class ruleComparison
{
    Eq,        ///< Equal
    Neq,       ///< Not equal
    Lt,        ///< Less than
    Gt,        ///< Greater than
    LtEq,      ///< Less than or equal to
    GtEq,      ///< Greater than or equal to
    And,       ///< boolean and
    Nand,      ///< boolean nand
    Or,        ///< boolean or
    Nor,       ///< boolean nor
    Xor = Neq, ///< boolean xor, equivalent to not equal
    Xnor = Eq  ///< boolean xnor, equivalent to equal
};

/// Get the \ref ruleComparison member from a string representation.
/** Needed for processing configuration files
  */
ruleComparison string2comp( const std::string & cstr )
{
    if(cstr == "Eq")
    {
        return ruleComparison::Eq;
    }
    else if(cstr == "Neq")
    {
        return ruleComparison::Neq;
    }
    else if(cstr == "Lt")
    {
        return ruleComparison::Lt;
    }
    else if(cstr == "Gt")
    {
        return ruleComparison::Gt;
    }
    else if(cstr == "LtEq")
    {
        return ruleComparison::LtEq;
    }
    else if(cstr == "GtEq")
    {
        return ruleComparison::GtEq;
    }
    else if(cstr == "And")
    {
        return ruleComparison::And;
    }
    else if(cstr == "Nand")
    {
        return ruleComparison::Nand;
    }
    else if(cstr == "Or")
    {
        return ruleComparison::Or;
    }
    else if(cstr == "Nor")
    {
        return ruleComparison::Nor;
    }
    else if(cstr == "Xor")
    {
        return ruleComparison::Xor;
    }
    else if(cstr == "Xnor")
    {
        return ruleComparison::Xnor;
    }
    else
    {
        mxThrowException(mx::err::invalidarg, "string2comp", cstr + " is not a valid comparison");
    }
}

/// Reporting priorities for rules
enum class rulePriority
{
    none,    ///< Don't publish
    info,    ///< For information only
    caution, ///< Caution -- make sure you know what you're doing
    warning, ///< Warning -- something is probably wrong, you should check
    alert    ///< Alert -- something is definitely wrong, you should take action
};

/// Get the \ref rulePriority member from a string representation.
/** Needed for processing configuration files
  */
rulePriority string2priority(const std::string & pstr)
{
    if(pstr == "none")
    {
        return rulePriority::none;
    }
    else if(pstr == "info")
    {
        return rulePriority::info;
    }
    else if(pstr == "caution")
    {
        return rulePriority::caution;
    }
    else if(pstr == "warning")
    {
        return rulePriority::warning;
    }
    else if(pstr == "alert")
    {
        return rulePriority::alert;
    }
    else
    {
        mxThrowException(mx::err::invalidarg, "string2priority", pstr + " is not a valid priority");
    }
}

/// Virtual base-class for all rules
/** Provides error handling and comparison functions.
  * Derived classes must implemented valid() and value(). 
  */
struct indiCompRule
{
public:

    /// In-band error reporting type
    typedef std::variant<bool, std::string> boolorerr_t;

    /// Check if returned value indicates an error
    bool isError(boolorerr_t rv /**< [in] the return value to check*/) 
    {
        return (rv.index() > 0);
    }

protected:

    /// The reporting priority for this rule
    rulePriority m_priority {rulePriority::none};

    /// The message used for notifications
    std::string m_message;

    /// The comparison for this rule
    ruleComparison m_comparison {ruleComparison::Eq};

public:

    /// Virtual destructor
    virtual ~indiCompRule()
    {}

    /// Set priority of this rule
    void priority( const rulePriority & p /**< [in] the new priority */)
    {
        m_priority = p;
    }

    /// Get the rule priority
    /**
      * \returns the current rule priority
      */
    const rulePriority & priority()
    {
        return m_priority;
    }

    /// Set the message
    void message(const std::string & m /**< [in] the new message*/)
    {
        m_message = m;
    }

    /// Get the message 
    /**
      * \returns the current message
      */
    const std::string & message()
    {
        return m_message;
    }

    /// Set the comparison for this rule
    void comparison( const ruleComparison & c /**< [in] the new comparison*/)
    {
        m_comparison = c;
    } 

    /// Get the rule comparison
    /**
      * \returns the current rule comparison
      *
      */
    const ruleComparison & comparison()
    {
        return m_comparison;
    }

    /// Report whether the rule is valid as configured
    /** If not valid, the return value is a std::string with the reason.
      * If valid, the return value is a bool set to true. 
      */
    virtual boolorerr_t valid() = 0;

    /// Get the value of this rule
    /**
      * \returns the result of the comparison defined by the rule
      */
    virtual bool value() = 0; 

    /// Compare two strings
    /** String comparison can only be Eq or Neq.
      *
      * \returns true if the comparison is true
      * \returns false if the comparison is false
      * \returns std::string with error message if the comparison is not valid
      */
    boolorerr_t compTxt( const std::string & str1, ///< [in] the first string to compare
                         const std::string & str2  ///< [in] the second string to compare
                       )
    {
        boolorerr_t rv = false;

        switch(m_comparison)
        {
            case ruleComparison::Eq:
                if(str1 == str2) rv = true;
                break;
            case ruleComparison::Neq:
                if(str1 != str2) rv = true;
                break;
            default:
                rv = "operator not valid for string comparison";
        }

        return rv;
    }

    /// Compare two switches
    /** Switch comparison can only be Eq or Neq.
      *
      * \returns true if the comparison is true
      * \returns false if the comparison is false
      * \returns std::string with error message if the comparison is not valid
      */
    boolorerr_t compSw( const pcf::IndiElement::SwitchStateType & sw1, ///< [in] the first switch to compare
                        const pcf::IndiElement::SwitchStateType & sw2  ///< [in] the first switch to compare
                      )
    {
        boolorerr_t rv = false;

        switch(m_comparison)
        {
            case ruleComparison::Eq:
                if(sw1 == sw2) rv = true;
                break;
            case ruleComparison::Neq:
                if(sw1 != sw2) rv = true;
                break;
            default:
                rv = "operator not valid for switch comparison";
        }

        return rv;
    }

    /// Compare two numbers
    /** The comparison is (num1 comp num2), e.g. (num1 \< num2).
      * A tolerance is included for floating point equality.
      *
      * \returns true if the comparison is true
      * \returns false if the comparison is false
      * \returns std::string with error message if the comparison is not valid
      */
    boolorerr_t compNum( const double & num1, ///< [in] the first number to compare
                         const double & num2, ///< [in] the second number to compare
                         const double & tol   ///< [in] the tolerance for the comparison
                       )
    {
        boolorerr_t rv = false;

        switch(m_comparison)
        {
            case ruleComparison::Eq:
                if( fabs(num1 - num2) <= tol ) rv = true;
                break;
            case ruleComparison::Neq:
                if( fabs(num1 - num2) > tol ) rv = true;
                break;
            case ruleComparison::Lt:
                if( num1 < num2) rv = true;
                break;
            case ruleComparison::Gt:
                if( num1 > num2) rv = true;
                break;
            case ruleComparison::LtEq:
                if( fabs(num1 - num2) <= tol ) rv = true;
                else if (num1 < num2) rv = true;
                break;
            case ruleComparison::GtEq:
                if( fabs(num1 - num2) <= tol ) rv = true;
                else if (num1 > num2) rv = true;
                break;
            default:
                rv = "operator not valid for compNum";
        }

        return rv;
    }

    /// Compare two booleans
    /** 
      * \returns true if the comparison is true
      * \returns false if the comparison is false
      * \returns std::string with error message if the comparison is not valid
      */
    boolorerr_t compBool( const bool & b1, ///< [in] the first bool to compare
                          const bool & b2  ///< [in] the second bool to compare
                        )
    {
        boolorerr_t rv = false;

        switch(m_comparison)
        {
            case ruleComparison::Eq:
                if(b1 == b2) rv = true;
                break;
            case ruleComparison::Neq:
                if(b1 != b2) rv = true;
                break;
            case ruleComparison::And:
                if(b1 && b2) rv = true;
                break;
            case ruleComparison::Nand:
                if(!(b1 && b2)) rv = true;
                break;
            case ruleComparison::Or:
                if(b1 || b2) rv = true;
                break;
            case ruleComparison::Nor:
                if(!b1 && !b2) rv = true;
                break;
            default:
                rv = "operator not valid for ruleCompRule";
        }

        return rv;
    }
};

/// A rule base class for testing an element in one property
struct onePropRule : public indiCompRule 
{

protected:

    int m_type; ///< The property type, from pcf::IndiProperty::Type

    pcf::IndiProperty * m_property {nullptr}; ///< Pointer to the property 
    
    std::string m_element; ///< The element name within the property

public:

    //Default c'tor is deleted, you must supply the property type
    onePropRule() = delete;

    /// Constructor.  You must provide the property type to construct a onePropRule
    explicit onePropRule( int type ) : m_type(type /**< The property type, from pcf::IndiProperty::Type*/)
    {}

    /// Set the property pointer
    /**
      * \throws mx::err::invalidarg if \p property is nullptr
      * \throws mx::err::invalidconfig if the supplied property has the wrong type
      */
    void property( pcf::IndiProperty * property /**< [in] the new property pointer*/)
    {
        if(property == nullptr)
        {
            mxThrowException(mx::err::invalidarg, "onePropRule::property", "property is nullptr");
        }

        if(property->getType() != m_type)
        {
            mxThrowException(mx::err::invalidconfig, "onePropRule::property", "property is not correct type");
        }

        m_property = property;
    }

    /// Get the property pointer
    /**
      * \returns the current value of m_property
      */
    const pcf::IndiProperty * property()
    {
        return m_property;
    }

    /// Set the element name
    void element(const std::string & el /**< [in] the new element name*/)
    {
        m_element = el;
    }

    /// Get the element name
    /**
      * \returns the current value of m_element
      */
    const std::string & element()
    {
        return m_element;
    }

    /// Check if this rule is valid 
    /** The rule is valid if the property pointer is not null, and the element
      * is contained within the property.
      *
      * If not valid, the return value is a std::string with the reason.
      * If valid, the return value is a bool set to true. 
      */
    virtual boolorerr_t valid()
    {
        boolorerr_t rv;
        if(m_property == nullptr)
        {
            rv = "property is null";
        }
        else if(!m_property->find(m_element))
        {
            rv = "element is not found";
        }
        else
        {
            rv = true;
        }

        return rv;
    }
};

/// A rule base class for testing elements in two properties
struct twoPropRule : public indiCompRule 
{

protected:

    int m_type; ///< The property type, from pcf::IndiProperty::Type

    pcf::IndiProperty * m_property1 {nullptr}; ///< Pointer to the first property
    
    std::string m_element1; ///< The element name within the first property

    pcf::IndiProperty * m_property2 {nullptr}; ///< Pointer to the second property
    
    std::string m_element2; ///< The element name within the second property

public:

    //Default c'tor is deleted, you must supply the property type
    twoPropRule() = delete;

    /// Constructor.  You must provide the property type to construct a twoPropRule
    explicit twoPropRule( int type ) : m_type(type /**< The property type, from pcf::IndiProperty::Type*/)
    {}

    /// Set the first property pointer
    /**
      * \throws mx::err::invalidarg if \p property is nullptr
      * \throws mx::err::invalidconfig if the supplied property has the wrong type
      */
    void property1( pcf::IndiProperty * property /**< [in] the new property pointer*/)
    {
        if(property == nullptr)
        {
            mxThrowException(mx::err::invalidarg, "twoPropRule::property1", "property is nullptr");
        }

        if(property->getType() != m_type)
        {
            mxThrowException(mx::err::invalidconfig, "twoPropRule::property1", "property is not correct type");
        }

        m_property1 = property;
    }

    /// Get the first property pointer
    /**
      * \returns the current value of m_property1
      */
    const pcf::IndiProperty * property1()
    {
        return m_property1;
    }

    /// Set the first element name
    void element1(const std::string & el /**< [in] the new element name*/)
    {
        m_element1 = el;
    }

    /// Get the first element name
    /**
      * \returns the current value of m_element1
      */
    const std::string & element1()
    {
        return m_element1;
    }

    /// Set the second property pointer
    /**
      * \throws mx::err::invalidarg if \p property is nullptr
      * \throws mx::err::invalidconfig if the supplied property has the wrong type
      */
    void property2( pcf::IndiProperty * property /**< [in] the new property pointer*/)
    {
        if(property == nullptr)
        {
            mxThrowException(mx::err::invalidarg, "twoPropRule::property2", "property is nullptr");
        }

        if(property->getType() != m_type)
        {
            mxThrowException(mx::err::invalidconfig, "twoPropRule::property2", "property is not correct type");
        }

        m_property2 = property;
    }

    /// Get the second property pointer
    /**
      * \returns the current value of m_property2
      */
    const pcf::IndiProperty * property2()
    {
        return m_property2;
    }

    /// Set the second element name
    void element2(const std::string & el /**< [in] the new element name*/)
    {
        m_element2 = el;
    }

    /// Get the second element name
    /**
      * \returns the current value of m_element2
      */
    const std::string & element2()
    {
        return m_element2;
    }

    /// Check if this rule is valid 
    /** The rule is valid if both property pointers are not null, and the elements
      * are contained within their respective properties.
      *
      * If not valid, the return value is a std::string with the reason.
      * If valid, the return value is a bool set to true. 
      */
    virtual boolorerr_t valid()
    {
        boolorerr_t rv;
        
        if(m_property1 == nullptr)
        {
            rv = "property1 is null";
            return rv;
        }
        
        if(!m_property1->find(m_element1))
        {
            rv = "element1 is not found";
            return rv;
        }

        if(m_property2 == nullptr)
        {
            rv = "property2 is null";
            return rv;
        }
        
        if(!m_property2->find(m_element2))
        {
            rv = "element2 is not found";
            return rv;
        }

        rv = true;

        return rv;
    }
};

/// Compare the value of a number element to a target
/** 
  */
struct numValRule : public onePropRule
{

public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "numVal";

protected:

    double m_target {0}; ///< The target value for comparison
    double m_tol {1e-6}; ///< The tolerance for the comparison

public:

    /// Default c'tor.
    numValRule() : onePropRule(pcf::IndiProperty::Number)
    {}

    /// Set the target for the comparison
    void target( const double & tgt /**< [in] The new target*/)
    {
        m_target = tgt;
    }

    /// Get the target
    /**
      * \returns the current value of m_target
      */
    const double & target()
    {
        return m_target;
    }

    /// Set the tolerance
    /** This is used for equality comparison to allow for floating point precision
      * and text conversions in INDI.  Set to 0 for strict comparison.
      *
      * \throws mx::err:invalidarg if the new value is negative
      */
    void tol( const double & t /**< [in] the new tolerance*/)
    {
        if(t < 0)
        {
            mxThrowException(mx::err::invalidarg, "numValRule::tol", "tolerance can't be negative");
        }

        m_tol = t;
    }

    /// Get the tolerance
    /**
      * \returns the current value of m_tol
      */
    const double & tol()
    {
        return m_tol;
    }

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "numValRule::value", std::get<std::string>(rv));
        }
        
        double val = (*m_property)[m_element].get<double>();
        
        rv = compNum(val, m_target, m_tol);
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "numValRule::value", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

/// Compare the value of a text element to a target value
/** Can only be Eq or Neq. 
  */
struct txtValRule : public onePropRule
{

public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "txtVal";

protected:
    std::string m_target; ///< The target value for comparison

public:

    /// Default c'tor.
    txtValRule() : onePropRule(pcf::IndiProperty::Text)
    {}

    /// Set the target for the comparison
    void target(const std::string & target /**< [in] The new target*/)
    {
        m_target = target;
    }

    /// Get the target
    /**
      * \returns the current value of m_target
      */
    const std::string & target()
    {
        return m_target;
    }

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "txtValRule::value", std::get<std::string>(rv));
        }

        rv = compTxt((*m_property)[m_element].get(), m_target);
        if(isError(rv))                
        {
            mxThrowException(mx::err::invalidconfig, "txtValRule::value()", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

/// Compare the value of a switch to a target value
/** Can only be Eq or Neq to On or Off.
  */
struct swValRule : public onePropRule 
{

public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "swVal";

protected:
    pcf::IndiElement::SwitchStateType m_target {pcf::IndiElement::UnknownSwitchState}; ///< The target value for comparison

public:

    /// Default c'tor.
    swValRule() : onePropRule(pcf::IndiProperty::Switch )
    {}

    /// Set the target for the comparison
    void target(const pcf::IndiElement::SwitchStateType & ss /**< [in] The new target*/)
    {
        m_target = ss;
    }

    /// Set the target for the comparison
    /** This version provided for config file processing.
      *
      * \throws mx::err::invalidarg if switchState is something other than "On" or Off
      */
    void target(const std::string & switchState /**< [in] The new target*/)
    {
        if(switchState == "On")
        {
            m_target = pcf::IndiElement::On;
        }
        else if(switchState == "Off")
        {
            m_target = pcf::IndiElement::Off;
        }
        else
        {
            mxThrowException(mx::err::invalidarg, "swValRule::target", "invalid switch state");
        }
    }

    /// Get the target
    /**
      * \returns the current value of m_target
      */
    const pcf::IndiElement::SwitchStateType & target()
    {
        return m_target;
    }

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "swValRule::value", std::get<std::string>(rv));
        }

        rv = compSw((*m_property)[m_element].getSwitchState(), m_target);
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompSwRule::value()", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

/// Compare two elements based on their numeric values
struct elCompNumRule : public twoPropRule
{    

public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "elCompNum";

protected:
    double m_tol {1e-6}; ///< The tolerance for the comparison

public:

    /// Default c'tor.
    elCompNumRule() : twoPropRule(pcf::IndiProperty::Number)
    {}

    /// Set the tolerance
    /** This is used for equality comparison to allow for floating point precision
      * and text conversions in INDI.  Set to 0 for strict comparison.
      *
      * \throws mx::err:invalidarg if the new value is negative
      */
    void tol( const double & t /**< [in] the new tolerance*/)
    {
        if(t < 0)
        {
            mxThrowException(mx::err::invalidarg, "numValRule::tol", "tolerance can't be negative");
        }

        m_tol = t;
    }

    /// Get the tolerance
    /**
      * \returns the current value of m_tol
      */
    const double & tol()
    {
        return m_tol;
    }

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompNumRule::value", std::get<std::string>(rv));
        }

        rv = compNum((*m_property1)[m_element1].get<double>(), (*m_property2)[m_element2].get<double>(), m_tol);
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompNumRule::value()", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

/// Compare two elements based on their text values
struct elCompTxtRule : public twoPropRule
{ 
public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "elCompTxt";

    /// Default c'tor.
    elCompTxtRule() : twoPropRule(pcf::IndiProperty::Text)
    {}

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompTxtRule::value", std::get<std::string>(rv));
        }

        rv = compTxt((*m_property1)[m_element1].get(), (*m_property2)[m_element2].get());
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompTxtRule::value()", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

/// Compare two elements based on their switch values
struct elCompSwRule : public twoPropRule
{

public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "elCompSw";

    /// Default c'tor.
    elCompSwRule() : twoPropRule(pcf::IndiProperty::Switch)
    {}

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompSwRule::value", std::get<std::string>(rv));
        }

        rv = compSw((*m_property1)[m_element1].getSwitchState(), (*m_property2)[m_element2].getSwitchState());
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "elCompSwRule::value()", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

/// A rule to compare two rules
/** 
  *
  */
struct ruleCompRule: public indiCompRule
{

public:

    /// Name of this rule, used by config system
    static constexpr char name[] = "ruleComp";

protected:
    indiCompRule * m_rule1 {nullptr}; ///< rule one
    indiCompRule * m_rule2 {nullptr}; ///< rule two

public:

    /// Set the pointer to the first rule
    void rule1( indiCompRule * r /**< [in] the new pointer to rule1*/)
    {
        m_rule1 = r;
    }

    /// Get the pointer to the first rule
    /**
      * \returns the current value of m_rule1
      */
    const indiCompRule * rule1()
    {
        return m_rule1;
    }

    /// Set the pointer to the second rule
    void rule2( indiCompRule * r /**< [in] the new pointer to rule2*/)
    {
        m_rule2 = r;
    }

    /// Get the pointer to the first rule
    /**
      * \returns the current value of m_rule2
      */
    const indiCompRule * rule2()
    {
        return m_rule2;
    }

    /// Check if this rule is valid 
    /** The rule is valid if the rule pointers are not nullptr, and if each rule is itself valid.
      *
      * If not valid, the return value is a std::string with the reason.
      * If valid, the return value is a bool set to true. 
      */
    virtual boolorerr_t valid()
    {
        boolorerr_t rv;
        if(m_rule1 == nullptr)
        {
            rv = "rule1 is nullptr";
        }
        else if(m_rule2 == nullptr)
        {
            rv = "rule2 is nullptr";
        }
        else
        {
            rv = m_rule1->valid();
            if(isError(rv)) 
            {
                return rv;
            }

            rv = m_rule2->valid();
            if(isError(rv)) 
            {
                return rv;
            }

            rv = true;
        }

        return rv;
    }

    /// Get the value of this rule
    /** First checks if the rule is currently valid.  The performs the comparison and returns the result.
      *
      * \returns the value of the comparison, true or false
      *
      * \throws mx::err::invalidconfig if the rule is not currently valid
      * \throws mx::err::invalidconfig on an error from the comparison
      *
      */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "ruleCompRule::value", std::get<std::string>(rv));
        }

        rv = compBool(m_rule1->value(), m_rule2->value());
        if(isError(rv))
        {
            mxThrowException(mx::err::invalidconfig, "ruleCompRule::value", std::get<std::string>(rv));
        }

        return std::get<bool>(rv);
    }
};

#endif //stateRuleEngine_indiCompRules_hpp
