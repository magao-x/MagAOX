/** \file indiCompRules.hpp
 * \brief The rules for the MagAO-X stateRuleEngine
 *
 * \ingroup stateRuleEngine_files
 */

#ifndef stateRuleEngine_indiCompRules_hpp
#define stateRuleEngine_indiCompRules_hpp

#include <variant>
#include <vector>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
                                         //Included here for standalone testing of this file

/// Logical comparisons for the INDI rules
enum class ruleComparison
{
    Eq,   ///< Equal
    Neq,  ///< Not equal
    Lt,   ///< Less than
    Gt,   ///< Greater than
    LtEq, ///< Less than or equal to
    GtEq, ///< Greater than or equal to
    And,  ///< boolean and
    Nand, ///< boolean nand
    Or,   ///< boolean or
    Nor,  ///< boolean nor
    Imply,
    Nimply,
    Xor  = Neq, ///< boolean xor, equivalent to not equal
    Xnor = Eq   ///< boolean xnor, equivalent to equal
};

/// Get the \ref ruleComparison member from a string representation.
/** Needed for processing configuration files
 */
ruleComparison string2comp( const std::string &cstr )
{
    if( cstr == "Eq" )
    {
        return ruleComparison::Eq;
    }
    else if( cstr == "Neq" )
    {
        return ruleComparison::Neq;
    }
    else if( cstr == "Lt" )
    {
        return ruleComparison::Lt;
    }
    else if( cstr == "Gt" )
    {
        return ruleComparison::Gt;
    }
    else if( cstr == "LtEq" )
    {
        return ruleComparison::LtEq;
    }
    else if( cstr == "GtEq" )
    {
        return ruleComparison::GtEq;
    }
    else if( cstr == "And" )
    {
        return ruleComparison::And;
    }
    else if( cstr == "Nand" )
    {
        return ruleComparison::Nand;
    }
    else if( cstr == "Or" )
    {
        return ruleComparison::Or;
    }
    else if( cstr == "Nor" )
    {
        return ruleComparison::Nor;
    }
    else if( cstr == "Xor" )
    {
        return ruleComparison::Xor;
    }
    else if( cstr == "Xnor" )
    {
        return ruleComparison::Xnor;
    }
    else if( cstr == "Imply" )
    {
        return ruleComparison::Imply;
    }
    else if( cstr == "Nimply" )
    {
        return ruleComparison::Nimply;
    }
    else
    {
        throw mx::exception( mx::error_t::invalidarg, cstr + " is not a valid comparison" );
    }
}

/// Get the string representation of a \ref ruleComparison member.
/** Needed for processing configuration files.
 */
std::string comp2string( const ruleComparison &comparison /**< [in] the comparison enum value to stringify */ )
{
    switch( comparison )
    {
    case ruleComparison::Eq:
        return "Eq";
    case ruleComparison::Neq:
        return "Neq";
    case ruleComparison::Lt:
        return "Lt";
    case ruleComparison::Gt:
        return "Gt";
    case ruleComparison::LtEq:
        return "LtEq";
    case ruleComparison::GtEq:
        return "GtEq";
    case ruleComparison::And:
        return "And";
    case ruleComparison::Nand:
        return "Nand";
    case ruleComparison::Or:
        return "Or";
    case ruleComparison::Nor:
        return "Nor";
    case ruleComparison::Imply:
        return "Imply";
    case ruleComparison::Nimply:
        return "Nimply";
    default:
        throw mx::exception( mx::error_t::invalidarg, "comparison is not valid" );
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
rulePriority string2priority( const std::string &pstr )
{
    if( pstr == "none" )
    {
        return rulePriority::none;
    }
    else if( pstr == "info" )
    {
        return rulePriority::info;
    }
    else if( pstr == "caution" )
    {
        return rulePriority::caution;
    }
    else if( pstr == "warning" )
    {
        return rulePriority::warning;
    }
    else if( pstr == "alert" )
    {
        return rulePriority::alert;
    }
    else
    {
        throw mx::exception( mx::error_t::invalidarg, pstr + " is not a valid priority" );
    }
}

/// Virtual base-class for all rules
/** Provides error handling and comparison functions.
 * Derived classes must implement valid() and value().
 */
struct indiCompRule
{
  public:
    /// In-band error reporting type
    typedef std::variant<bool, std::string> boolorerr_t;

    /// Check if returned value indicates an error
    bool isError( boolorerr_t rv /**< [in] the return value to check*/ )
    {
        return ( rv.index() > 0 );
    }

    static constexpr double default_info_msg_delay    = 0; // Send once
    static constexpr double default_caution_msg_delay = 60;
    static constexpr double default_warning_msg_delay = 30;
    static constexpr double default_alert_msg_delay   = 5;

  protected:
    /// The reporting priority for this rule
    rulePriority m_priority{ rulePriority::none };

    /// The message used for notifications
    std::string m_message;

    timespec m_lastMsg{ 0, 0 }; ///< Time the message was last sent

    double m_messageDelay{ 0 }; ///< Delay between sending messages

    int m_messageCount{ 0 }; ///< Number of times the message has been sent

    /// The comparison for this rule
    ruleComparison m_comparison{ ruleComparison::Eq };

  public:
    /// Virtual destructor
    virtual ~indiCompRule()
    {
    }

    /// Get the default comparison for this rule type.
    /**
     * \returns the comparison that should be used when `comp` is omitted from configuration
     */
    virtual ruleComparison defaultComparison() const
    {
        return ruleComparison::Eq;
    }

    /// Set priority of this rule
    /** Also sets the message delay, to default for priority if not set.
     */
    void priority( const rulePriority &p,         /**< [in] the new priority */
                   double              delay = -1 /**< [in] [opt] the message delay, if \< 0 the default is used */
    )
    {
        m_priority = p;

        if( delay < 0 )
        {
            switch( m_priority )
            {
            case rulePriority::info:
                m_messageDelay = default_info_msg_delay;
                break;
            case rulePriority::caution:
                m_messageDelay = default_caution_msg_delay;
                break;
            case rulePriority::warning:
                m_messageDelay = default_warning_msg_delay;
                break;
            case rulePriority::alert:
                m_messageDelay = default_alert_msg_delay;
                break;
            default:
                m_messageDelay = 0;
            }
        }
        else
        {
            m_messageDelay = delay;
        }
    }

    /// Get the rule priority
    /**
     * \returns the current rule priority
     */
    const rulePriority &priority()
    {
        return m_priority;
    }

    /// Set the message
    void message( const std::string &m /**< [in] the new message*/ )
    {
        m_message = m;
    }

    /// Get the message
    /** Optionally sets the message time to now.
     * \returns the current message
     */
    const std::string &message( bool settime = false /**< If true m_lastMsg is set to now */ )
    {
        if( settime )
        {
            if( clock_gettime( CLOCK_ISIO, &m_lastMsg ) < 0 )
            {
                throw mx::exception( mx::errno2error_t( errno ), "getting message time" );
            }
        }

        return m_message;
    }

    const timespec &lastMsg()
    {
        return m_lastMsg;
    }

    /// Get the time since the last message
    double sinceLastMsg()
    {
        timespec ts;
        if( clock_gettime( CLOCK_ISIO, &ts ) < 0 )
        {
            throw mx::exception( mx::errno2error_t( errno ), "getting current time" );
        }

        return ( 1.0 * ts.tv_sec + ts.tv_nsec / 1e9 ) - ( 1.0 * m_lastMsg.tv_sec + m_lastMsg.tv_nsec / 1e9 );
    }

    /// Check if it's time to send a message
    /** If the message delay is \<= 0, this is based on message count (i.e. has it been sent).
     * Otherwise it's based on the time since last sent
     */
    bool timeToSend()
    {
        if( m_messageDelay <= 0 )
        {
            if( m_messageCount == 0 )
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
            if( m_messageCount == 0 || sinceLastMsg() >= m_messageDelay )
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    /// Set the message delay
    void messageDelay( double md /**< [in] the new message delay */ )
    {
        m_messageDelay = md;
    }

    /// Get the message delay
    double messageDelay()
    {
        return m_messageDelay;
    }

    /// Set the message count
    void messageCount( int mc /**< [in] the new message count */ )
    {
        m_messageCount = mc;
    }

    /// Increment the message count
    int incMessageCount()
    {
        ++m_messageCount;
        return m_messageCount;
    }

    /// Get the message count
    int messageCount()
    {
        return m_messageCount;
    }

    /// Set the comparison for this rule
    void comparison( const ruleComparison &c /**< [in] the new comparison*/ )
    {
        m_comparison = c;
    }

    /// Get the rule comparison
    /**
     * \returns the current rule comparison
     *
     */
    const ruleComparison &comparison()
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

    /// Pop one pending runtime diagnostic, if any.
    /**
     * \returns true when a diagnostic string was returned
     * \returns false when no diagnostic was pending
     */
    virtual bool popRuntimeDiagnostic( std::string &diagnostic /**< [out] the next pending diagnostic message */ )
    {
        diagnostic = "";
        return false;
    }

    /// Compare two strings
    /** String comparison can only be Eq or Neq.
     *
     * \returns true if the comparison is true
     * \returns false if the comparison is false
     * \returns std::string with error message if the comparison is not valid
     */
    boolorerr_t compTxt( const std::string &str1, ///< [in] the first string to compare
                         const std::string &str2  ///< [in] the second string to compare
    )
    {
        boolorerr_t rv = false;

        switch( m_comparison )
        {
        case ruleComparison::Eq:
            if( str1 == str2 )
                rv = true;
            break;
        case ruleComparison::Neq:
            if( str1 != str2 )
                rv = true;
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
    boolorerr_t compSw( const pcf::IndiElement::SwitchStateType &sw1, ///< [in] the first switch to compare
                        const pcf::IndiElement::SwitchStateType &sw2  ///< [in] the first switch to compare
    )
    {
        boolorerr_t rv = false;

        switch( m_comparison )
        {
        case ruleComparison::Eq:
            if( sw1 == sw2 )
                rv = true;
            break;
        case ruleComparison::Neq:
            if( sw1 != sw2 )
                rv = true;
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
    boolorerr_t compNum( const double &num1, ///< [in] the first number to compare
                         const double &num2, ///< [in] the second number to compare
                         const double &tol   ///< [in] the tolerance for the comparison
    )
    {
        boolorerr_t rv = false;

        switch( m_comparison )
        {
        case ruleComparison::Eq:
            if( fabs( num1 - num2 ) <= tol )
                rv = true;
            break;
        case ruleComparison::Neq:
            if( fabs( num1 - num2 ) > tol )
                rv = true;
            break;
        case ruleComparison::Lt:
            if( num1 < num2 )
                rv = true;
            break;
        case ruleComparison::Gt:
            if( num1 > num2 )
                rv = true;
            break;
        case ruleComparison::LtEq:
            if( fabs( num1 - num2 ) <= tol )
                rv = true;
            else if( num1 < num2 )
                rv = true;
            break;
        case ruleComparison::GtEq:
            if( fabs( num1 - num2 ) <= tol )
                rv = true;
            else if( num1 > num2 )
                rv = true;
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
    boolorerr_t compBool( const bool &b1, ///< [in] the first bool to compare
                          const bool &b2  ///< [in] the second bool to compare
    )
    {
        boolorerr_t rv = false;

        switch( m_comparison )
        {
        case ruleComparison::Eq:
            if( b1 == b2 )
                rv = true;
            break;
        case ruleComparison::Neq:
            if( b1 != b2 )
                rv = true;
            break;
        case ruleComparison::And:
            if( b1 && b2 )
                rv = true;
            break;
        case ruleComparison::Nand:
            if( !( b1 && b2 ) )
                rv = true;
            break;
        case ruleComparison::Or:
            if( b1 || b2 )
                rv = true;
            break;
        case ruleComparison::Nor:
            if( !b1 && !b2 )
                rv = true;
            break;
        case ruleComparison::Imply:
            // https://en.wikipedia.org/wiki/Material_conditional
            if( !b1 || b2 )
                rv = true;
            break;
        case ruleComparison::Nimply:
            // https://en.wikipedia.org/wiki/Material_nonimplication
            if( b1 && !b2 )
                rv = true;
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

    pcf::IndiProperty *m_property{ nullptr }; ///< Pointer to the property

    std::string m_element; ///< The element name within the property

  public:
    // Default c'tor is deleted, you must supply the property type
    onePropRule() = delete;

    /// Constructor.  You must provide the property type to construct a onePropRule
    explicit onePropRule( int type ) : m_type( type /**< The property type, from pcf::IndiProperty::Type*/ )
    {
    }

    /// Set the property pointer
    /**
     * \throws mx::err::invalidarg if \p property is nullptr
     * \throws mx::err::invalidconfig if the supplied property has the wrong type
     */
    void property( pcf::IndiProperty *property /**< [in] the new property pointer*/ )
    {
        if( property == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "property is nullptr" );
        }

        if( property->getType() != m_type )
        {
            throw mx::exception( mx::error_t::invalidconfig, "property is not correct type" );
        }

        m_property = property;
    }

    /// Get the property pointer
    /**
     * \returns the current value of m_property
     */
    const pcf::IndiProperty *property()
    {
        return m_property;
    }

    /// Set the element name
    void element( const std::string &el /**< [in] the new element name*/ )
    {
        m_element = el;
    }

    /// Get the element name
    /**
     * \returns the current value of m_element
     */
    const std::string &element()
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
        if( m_property == nullptr )
        {
            rv = "property is null";
        }
        else if( !m_property->find( m_element ) )
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

    pcf::IndiProperty *m_property1{ nullptr }; ///< Pointer to the first property

    std::string m_element1; ///< The element name within the first property

    pcf::IndiProperty *m_property2{ nullptr }; ///< Pointer to the second property

    std::string m_element2; ///< The element name within the second property

  public:
    // Default c'tor is deleted, you must supply the property type
    twoPropRule() = delete;

    /// Constructor.  You must provide the property type to construct a twoPropRule
    explicit twoPropRule( int type ) : m_type( type /**< The property type, from pcf::IndiProperty::Type*/ )
    {
    }

    /// Set the first property pointer
    /**
     * \throws mx::err::invalidarg if \p property is nullptr
     * \throws mx::err::invalidconfig if the supplied property has the wrong type
     */
    void property1( pcf::IndiProperty *property /**< [in] the new property pointer*/ )
    {
        if( property == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "property is nullptr" );
        }

        if( property->getType() != m_type )
        {
            throw mx::exception( mx::error_t::invalidconfig, "property is not correct type" );
        }

        m_property1 = property;
    }

    /// Get the first property pointer
    /**
     * \returns the current value of m_property1
     */
    const pcf::IndiProperty *property1()
    {
        return m_property1;
    }

    /// Set the first element name
    void element1( const std::string &el /**< [in] the new element name*/ )
    {
        m_element1 = el;
    }

    /// Get the first element name
    /**
     * \returns the current value of m_element1
     */
    const std::string &element1()
    {
        return m_element1;
    }

    /// Set the second property pointer
    /**
     * \throws mx::err::invalidarg if \p property is nullptr
     * \throws mx::err::invalidconfig if the supplied property has the wrong type
     */
    void property2( pcf::IndiProperty *property /**< [in] the new property pointer*/ )
    {
        if( property == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "property is nullptr" );
        }

        if( property->getType() != m_type )
        {
            throw mx::exception( mx::error_t::invalidconfig, "property is not correct type" );
        }

        m_property2 = property;
    }

    /// Get the second property pointer
    /**
     * \returns the current value of m_property2
     */
    const pcf::IndiProperty *property2()
    {
        return m_property2;
    }

    /// Set the second element name
    void element2( const std::string &el /**< [in] the new element name*/ )
    {
        m_element2 = el;
    }

    /// Get the second element name
    /**
     * \returns the current value of m_element2
     */
    const std::string &element2()
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

        if( m_property1 == nullptr )
        {
            rv = "property1 is null";
            return rv;
        }

        if( !m_property1->find( m_element1 ) )
        {
            rv = "element1 is not found";
            return rv;
        }

        if( m_property2 == nullptr )
        {
            rv = "property2 is null";
            return rv;
        }

        if( !m_property2->find( m_element2 ) )
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
    double m_target{ 0 }; ///< The target value for comparison
    double m_tol{ 1e-6 }; ///< The tolerance for the comparison

  public:
    /// Default c'tor.
    numValRule() : onePropRule( pcf::IndiProperty::Number )
    {
    }

    /// Set the target for the comparison
    void target( const double &tgt /**< [in] The new target*/ )
    {
        m_target = tgt;
    }

    /// Get the target
    /**
     * \returns the current value of m_target
     */
    const double &target()
    {
        return m_target;
    }

    /// Set the tolerance
    /** This is used for equality comparison to allow for floating point precision
     * and text conversions in INDI.  Set to 0 for strict comparison.
     *
     * \throws mx::err:invalidarg if the new value is negative
     */
    void tol( const double &t /**< [in] the new tolerance*/ )
    {
        if( t < 0 )
        {
            throw mx::exception( mx::error_t::invalidarg, "tolerance can't be negative" );
        }

        m_tol = t;
    }

    /// Get the tolerance
    /**
     * \returns the current value of m_tol
     */
    const double &tol()
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        double val = ( *m_property )[m_element].get<double>();

        rv = compNum( val, m_target, m_tol );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
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
    txtValRule() : onePropRule( pcf::IndiProperty::Text )
    {
    }

    /// Set the target for the comparison
    void target( const std::string &target /**< [in] The new target*/ )
    {
        m_target = target;
    }

    /// Get the target
    /**
     * \returns the current value of m_target
     */
    const std::string &target()
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        rv = compTxt( ( *m_property )[m_element].get(), m_target );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
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
    pcf::IndiElement::SwitchStateType m_target{
        pcf::IndiElement::UnknownSwitchState }; ///< The target value for comparison

  public:
    /// Default c'tor.
    swValRule() : onePropRule( pcf::IndiProperty::Switch )
    {
    }

    /// Set the target for the comparison
    void target( const pcf::IndiElement::SwitchStateType &ss /**< [in] The new target*/ )
    {
        m_target = ss;
    }

    /// Set the target for the comparison
    /** This version provided for config file processing.
     *
     * \throws mx::err::invalidarg if switchState is something other than "On" or Off
     */
    void target( const std::string &switchState /**< [in] The new target*/ )
    {
        if( switchState == "On" )
        {
            m_target = pcf::IndiElement::On;
        }
        else if( switchState == "Off" )
        {
            m_target = pcf::IndiElement::Off;
        }
        else
        {
            throw mx::exception( mx::error_t::invalidarg, "invalid switch state" );
        }
    }

    /// Get the target
    /**
     * \returns the current value of m_target
     */
    const pcf::IndiElement::SwitchStateType &target()
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        rv = compSw( ( *m_property )[m_element].getSwitchState(), m_target );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }
};

/// Compare the difference in time between a value and now
/** Now is the time of evaluation of the rule
 */
struct timeDiffRule : public onePropRule
{

  public:
    /// Name of this rule, used by config system
    static constexpr char name[] = "timeDiff";

  protected:
    double m_target{ 0 }; ///< The target value for comparison
    double m_tol{ 1e-6 }; ///< The tolerance for the comparison

  public:
    /// Default c'tor.
    timeDiffRule() : onePropRule( pcf::IndiProperty::Number )
    {
    }

    /// Set the target for the comparison
    void target( const double &tgt /**< [in] The new target*/ )
    {
        m_target = tgt;
    }

    /// Get the target
    /**
     * \returns the current value of m_target
     */
    const double &target()
    {
        return m_target;
    }

    /// Set the tolerance
    /** This is used for equality comparison to allow for floating point precision
     * and text conversions in INDI.  Set to 0 for strict comparison.
     *
     * \throws mx::err:invalidarg if the new value is negative
     */
    void tol( const double &t /**< [in] the new tolerance*/ )
    {
        if( t < 0 )
        {
            throw mx::exception( mx::error_t::invalidarg, "tolerance can't be negative" );
        }

        m_tol = t;
    }

    /// Get the tolerance
    /**
     * \returns the current value of m_tol
     */
    const double &tol()
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        timespec now;
        clock_gettime( CLOCK_ISIO, &now );

        double val = ( 1.0 * now.tv_sec + now.tv_nsec / 1e9 ) - ( *m_property )[m_element].get<double>();

        rv = compNum( val, m_target, m_tol );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }
};

/// Compare two elements based on their numeric values
struct elCompNumRule : public twoPropRule
{

  public:
    /// Name of this rule, used by config system
    static constexpr char name[] = "elCompNum";

  protected:
    double m_tol{ 1e-6 }; ///< The tolerance for the comparison

  public:
    /// Default c'tor.
    elCompNumRule() : twoPropRule( pcf::IndiProperty::Number )
    {
    }

    /// Set the tolerance
    /** This is used for equality comparison to allow for floating point precision
     * and text conversions in INDI.  Set to 0 for strict comparison.
     *
     * \throws mx::err:invalidarg if the new value is negative
     */
    void tol( const double &t /**< [in] the new tolerance*/ )
    {
        if( t < 0 )
        {
            throw mx::exception( mx::error_t::invalidarg, "tolerance can't be negative" );
        }

        m_tol = t;
    }

    /// Get the tolerance
    /**
     * \returns the current value of m_tol
     */
    const double &tol()
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        rv = compNum( ( *m_property1 )[m_element1].get<double>(), ( *m_property2 )[m_element2].get<double>(), m_tol );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }
};

/// Compare two elements based on their text values
struct elCompTxtRule : public twoPropRule
{
  public:
    /// Name of this rule, used by config system
    static constexpr char name[] = "elCompTxt";

    /// Default c'tor.
    elCompTxtRule() : twoPropRule( pcf::IndiProperty::Text )
    {
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        rv = compTxt( ( *m_property1 )[m_element1].get(), ( *m_property2 )[m_element2].get() );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }
};

/// Compare two elements based on their switch values
struct elCompSwRule : public twoPropRule
{

  public:
    /// Name of this rule, used by config system
    static constexpr char name[] = "elCompSw";

    /// Default c'tor.
    elCompSwRule() : twoPropRule( pcf::IndiProperty::Switch )
    {
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        rv = compSw( ( *m_property1 )[m_element1].getSwitchState(), ( *m_property2 )[m_element2].getSwitchState() );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }
};

/// Build and compare a switch-name combination against a target switch vector.
/**
 * This rule reads the currently active element name from each source switch
 * property, combines those names with a literal `{}` placeholder format string,
 * and compares the result against the currently active element name in a target
 * switch property.
 */
struct multiSwitchComboRule : public indiCompRule
{

  public:
    /// Name of this rule, used by config system
    static constexpr char name[] = "multiSwitchCombo";

  protected:
    /// The configuration-section name of this rule, used in diagnostics.
    std::string m_ruleName;

    /// The source switch properties in format-substitution order.
    std::vector<pcf::IndiProperty *> m_properties;

    /// The config keys for the source switch properties, used in diagnostics.
    std::vector<std::string> m_propertyKeys;

    /// Per-source latch state used to avoid repeated multi-On diagnostics.
    std::vector<bool> m_multiOn;

    /// The literal format string used to combine source switch names.
    std::string m_format;

    /// The target switch property whose active element name is compared.
    pcf::IndiProperty *m_targetProperty{ nullptr };

    /// The config key for the target property, used in diagnostics.
    std::string m_targetPropertyKey;

    /// Latch state used to avoid repeated target multi-On diagnostics.
    bool m_targetMultiOn{ false };

    /// Runtime diagnostics that should be logged without failing evaluation.
    std::vector<std::string> m_pendingDiagnostics;

    /// Count plain `{}` placeholders and reject any other brace usage.
    size_t formatPlaceholders( bool &invalidBraces /**< [out] true when unsupported brace syntax is present */ ) const
    {
        invalidBraces = false;

        size_t count = 0;
        for( size_t n = 0; n < m_format.size(); ++n )
        {
            if( m_format[n] == '{' )
            {
                if( n + 1 < m_format.size() && m_format[n + 1] == '}' )
                {
                    ++count;
                    ++n;
                }
                else
                {
                    invalidBraces = true;
                    return count;
                }
            }
            else if( m_format[n] == '}' )
            {
                invalidBraces = true;
                return count;
            }
        }

        return count;
    }

    /// Resolve the active element name for a switch property.
    std::string activeName( pcf::IndiProperty *property,      /**< [in] the property to inspect */
                            const std::string &propertyKey,   /**< [in] the config key for diagnostics */
                            bool              &multiOnLatched /**< [in/out] the multi-On diagnostic latch */
    )
    {
        size_t      onCount = 0;
        std::string active;

        for( auto &&el : property->getElements() )
        {
            if( el.second.getSwitchState() == pcf::IndiElement::On )
            {
                if( onCount == 0 )
                {
                    active = el.first;
                }

                ++onCount;
            }
        }

        if( onCount == 0 )
        {
            multiOnLatched = false;
            return "";
        }

        if( onCount == 1 )
        {
            multiOnLatched = false;
            return active;
        }

        if( !multiOnLatched )
        {
            std::string ruleName = m_ruleName;
            if( ruleName == "" )
            {
                ruleName = "<unnamed>";
            }

            m_pendingDiagnostics.push_back(
                std::format( "multiSwitchCombo rule {} found multiple switch elements On in {}; using empty string",
                             ruleName,
                             propertyKey ) );
        }

        multiOnLatched = true;

        return "";
    }

    /// Apply literal `{}` substitution to build the comparison string.
    std::string
    formatCombo( const std::vector<std::string> &values /**< [in] the source switch names in order */ ) const
    {
        std::string combo;
        size_t      valueIndex = 0;

        for( size_t n = 0; n < m_format.size(); ++n )
        {
            if( m_format[n] == '{' && n + 1 < m_format.size() && m_format[n + 1] == '}' )
            {
                combo += values[valueIndex];
                ++valueIndex;
                ++n;
            }
            else
            {
                combo += m_format[n];
            }
        }

        return combo;
    }

  public:
    /// Default c'tor.
    /** Changes the default comparison to Neq for mismatch detection.
     */
    multiSwitchComboRule()
    {
        comparison( defaultComparison() );
    }

    /// Get the default comparison for this rule type.
    /**
     * \returns `ruleComparison::Neq`
     */
    virtual ruleComparison defaultComparison() const
    {
        return ruleComparison::Neq;
    }

    /// Set the rule name used in diagnostics.
    void ruleName( const std::string &ruleName /**< [in] the config-section name of this rule */ )
    {
        m_ruleName = ruleName;
    }

    /// Get the configured rule name.
    /**
     * \returns the current value of m_ruleName
     */
    const std::string &ruleName()
    {
        return m_ruleName;
    }

    /// Append one source switch property.
    void property( pcf::IndiProperty *property,   /**< [in] the next source switch property */
                   const std::string &propertyKey /**< [in] the config key for the source property */
    )
    {
        if( property == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "property is nullptr" );
        }

        if( property->getType() != pcf::IndiProperty::Switch )
        {
            throw mx::exception( mx::error_t::invalidconfig, "property is not correct type" );
        }

        m_properties.push_back( property );
        m_propertyKeys.push_back( propertyKey );
        m_multiOn.push_back( false );
    }

    /// Get a source switch property by index.
    /**
     * \returns the current source switch property at index `n`
     */
    const pcf::IndiProperty *property( size_t n /**< [in] the zero-based source property index */ )
    {
        if( n >= m_properties.size() )
        {
            throw mx::exception( mx::error_t::invalidarg, "source property index out of range" );
        }

        return m_properties[n];
    }

    /// Get a source property key by index.
    /**
     * \returns the configured property key at index `n`
     */
    const std::string &propertyKey( size_t n /**< [in] the zero-based source property index */ )
    {
        if( n >= m_propertyKeys.size() )
        {
            throw mx::exception( mx::error_t::invalidarg, "source property key index out of range" );
        }

        return m_propertyKeys[n];
    }

    /// Get the number of configured source switch properties.
    /**
     * \returns the current number of source properties
     */
    size_t numSwitches()
    {
        return m_properties.size();
    }

    /// Set the literal format string for the source switch names.
    void format( const std::string &format /**< [in] the literal `{}` placeholder format string */ )
    {
        m_format = format;
    }

    /// Get the literal format string.
    /**
     * \returns the current value of m_format
     */
    const std::string &format()
    {
        return m_format;
    }

    /// Set the target switch property.
    void targetProperty( pcf::IndiProperty *property /**< [in] the target switch property */ )
    {
        if( property == nullptr )
        {
            throw mx::exception( mx::error_t::invalidarg, "targetProperty is nullptr" );
        }

        if( property->getType() != pcf::IndiProperty::Switch )
        {
            throw mx::exception( mx::error_t::invalidconfig, "targetProperty is not correct type" );
        }

        m_targetProperty = property;
    }

    /// Get the target switch property.
    /**
     * \returns the current target switch property
     */
    const pcf::IndiProperty *targetProperty()
    {
        return m_targetProperty;
    }

    /// Set the target property key used in diagnostics.
    void targetPropertyKey( const std::string &propertyKey /**< [in] the config key for the target property */ )
    {
        m_targetPropertyKey = propertyKey;
    }

    /// Get the target property key used in diagnostics.
    /**
     * \returns the current value of m_targetPropertyKey
     */
    const std::string &targetPropertyKey()
    {
        return m_targetPropertyKey;
    }

    /// Check if this rule is valid as configured.
    virtual boolorerr_t valid()
    {
        if( m_properties.size() == 0 )
        {
            return "no source switch properties configured";
        }

        if( m_properties.size() != m_propertyKeys.size() || m_properties.size() != m_multiOn.size() )
        {
            return "source switch configuration is inconsistent";
        }

        for( size_t n = 0; n < m_properties.size(); ++n )
        {
            if( m_properties[n] == nullptr )
            {
                return std::format( "property{} is nullptr", n + 1 );
            }

            if( m_properties[n]->getType() != pcf::IndiProperty::Switch )
            {
                return std::format( "property{} is not a switch property", n + 1 );
            }

            if( m_propertyKeys[n] == "" )
            {
                return std::format( "property{} key is empty", n + 1 );
            }
        }

        if( m_targetProperty == nullptr )
        {
            return "targetProperty is nullptr";
        }

        if( m_targetProperty->getType() != pcf::IndiProperty::Switch )
        {
            return "targetProperty is not a switch property";
        }

        if( m_targetPropertyKey == "" )
        {
            return "targetProperty key is empty";
        }

        bool   invalidBraces = false;
        size_t placeholders  = formatPlaceholders( invalidBraces );
        if( invalidBraces )
        {
            return "format only supports literal {} placeholders";
        }

        if( placeholders != m_properties.size() )
        {
            return std::format(
                "format placeholder count {} does not match numSwitches {}", placeholders, m_properties.size() );
        }

        if( m_comparison != ruleComparison::Eq && m_comparison != ruleComparison::Neq )
        {
            return "operator not valid for multiSwitchComboRule";
        }

        return true;
    }

    /// Get the value of this rule.
    /** First checks if the rule is currently valid. Then performs the combo-name
     * comparison and returns the result.
     *
     * \returns the value of the comparison, true or false
     *
     * \throws mx::err::invalidconfig if the rule is not currently valid
     * \throws mx::err::invalidconfig on an error from the comparison
     */
    virtual bool value()
    {
        boolorerr_t rv = valid();
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        std::vector<std::string> activeNames;
        activeNames.reserve( m_properties.size() );
        for( size_t n = 0; n < m_properties.size(); ++n )
        {
            bool multiOn = m_multiOn[n];
            activeNames.push_back( activeName( m_properties[n], m_propertyKeys[n], multiOn ) );
            m_multiOn[n] = multiOn;
        }

        bool targetMultiOn = m_targetMultiOn;

        std::string comboName  = formatCombo( activeNames );
        std::string targetName = activeName( m_targetProperty, m_targetPropertyKey, targetMultiOn );
        m_targetMultiOn        = targetMultiOn;

        rv = compTxt( comboName, targetName );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }

    /// Pop one pending runtime diagnostic, if any.
    virtual bool popRuntimeDiagnostic( std::string &diagnostic /**< [out] the next pending diagnostic message */ )
    {
        if( m_pendingDiagnostics.size() == 0 )
        {
            diagnostic = "";
            return false;
        }

        diagnostic = m_pendingDiagnostics.front();
        m_pendingDiagnostics.erase( m_pendingDiagnostics.begin() );

        return true;
    }
};

/// A rule to compare two rules
/**
 *
 */
struct ruleCompRule : public indiCompRule
{

  public:
    /// Name of this rule, used by config system
    static constexpr char name[] = "ruleComp";

  protected:
    indiCompRule *m_rule1{ nullptr }; ///< rule one
    indiCompRule *m_rule2{ nullptr }; ///< rule two

  public:
    /// Default c'tor
    /** Changes default comparison to And for ruleCompRule
     */
    ruleCompRule()
    {
        comparison( defaultComparison() );
    }

    /// Get the default comparison for this rule type.
    /**
     * \returns `ruleComparison::And`
     */
    virtual ruleComparison defaultComparison() const
    {
        return ruleComparison::And;
    }

    /// Set the pointer to the first rule
    void rule1( indiCompRule *r /**< [in] the new pointer to rule1*/ )
    {
        m_rule1 = r;
    }

    /// Get the pointer to the first rule
    /**
     * \returns the current value of m_rule1
     */
    const indiCompRule *rule1()
    {
        return m_rule1;
    }

    /// Set the pointer to the second rule
    void rule2( indiCompRule *r /**< [in] the new pointer to rule2*/ )
    {
        m_rule2 = r;
    }

    /// Get the pointer to the first rule
    /**
     * \returns the current value of m_rule2
     */
    const indiCompRule *rule2()
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
        if( m_rule1 == nullptr )
        {
            rv = "rule1 is nullptr";
        }
        else if( m_rule2 == nullptr )
        {
            rv = "rule2 is nullptr";
        }
        else
        {
            rv = m_rule1->valid();
            if( isError( rv ) )
            {
                return rv;
            }

            rv = m_rule2->valid();
            if( isError( rv ) )
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
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        rv = compBool( m_rule1->value(), m_rule2->value() );
        if( isError( rv ) )
        {
            throw mx::exception( mx::error_t::invalidconfig, std::get<std::string>( rv ) );
        }

        return std::get<bool>( rv );
    }

    /// Pop one pending runtime diagnostic from either child rule.
    virtual bool popRuntimeDiagnostic( std::string &diagnostic /**< [out] the next pending diagnostic message */ )
    {
        if( m_rule1 != nullptr && m_rule1->popRuntimeDiagnostic( diagnostic ) )
        {
            return true;
        }

        if( m_rule2 != nullptr && m_rule2->popRuntimeDiagnostic( diagnostic ) )
        {
            return true;
        }

        diagnostic = "";
        return false;
    }
};

#endif // stateRuleEngine_indiCompRules_hpp
