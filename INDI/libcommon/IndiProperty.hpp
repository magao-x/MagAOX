/** \file IndiProperty.hpp
 *
 * Declarations for the IndiProperty class.
 *
 * @author Paul Grenz (@Steward Observatory, original author)
 * @author Jared Males (@Steward Observatory, refactored for MagAO-X)
 */

#ifndef libcommon_IndiProperty_hpp
#define libcommon_IndiProperty_hpp

#include <string>
#include <map>
#include <exception>
#include "TimeStamp.hpp"
#include "IndiElement.hpp"

#include <mutex>
#include <shared_mutex>
/* 2024-05-24 refactor in progress 
 * ToDo:
 * - make sure member data in best order
 * - organize get/set functions
 * - convert from rwLock to shared mutex
 */

namespace pcf
{

class IndiProperty
{

public:

    /// The possible types of INDI properties 
    /** The order and enumeration of this list is important.
      * Do not add or change enumerations here without adjusting
      * the indexing of the 'allowed attributes' list.
      */ 
    enum class Type
    {
        Unknown = 0, ///< Type is not known, generally indicates an error
        BLOB,        ///< The binary large object (BLOB) type
        Light,       ///< The INDI Light type
        Number,      ///< The INDI Number type
        Switch,      ///< The INDI Switch type
        Text,        ///< The INDI Text type
    };

    /// The possible states of an INDI Property
    enum class State
    {
        Unknown = 0, ///< State is not known, generally indicates an error
        Alert = 1,   ///< State Alert
        Busy,        ///< State Busy
        Ok,          ///< State Ok
        Idle         ///< State Idle
    };

    /// The possible INDI property permission states
    enum class Perm
    {
        Unknown = 0,  ///< Unknwon permission type, generally indicates an error
        ReadOnly = 1, ///< Property is read only
        ReadWrite,    ///< Property is read and write
        WriteOnly     ///< Property is write only
    };

    /// The INDI switch property switch-rules
    enum class SwitchRule
    {
        Unknown = 0,   ///< Unknwon switch rule, generally indicates an error
        AnyOfMany = 1, ///< Any of many rule
        AtMostOne,     ///< At most one rule
        OneOfMany      ///< One of many rule
    };

    enum class BLOBEnable
    {
        Unknown = 0,
        Also = 1,
        Only,
        Never
    };

    enum class Error
    {
        None = 0,
        CouldntFindElement = -3,
        ElementAlreadyExists = -5,
        IndexOutOfBounds = -6,
        WrongElementType = -7,
        Undefined = -9999
    };

    

    

    

    

    
public:
    class Excep : public std::exception
    {

    private:
        Excep() {}

    public:
        explicit Excep(const IndiProperty::Error &tCode) : m_tCode(tCode) {}

        ~Excep() throw() {}
        const IndiProperty::Error &getCode() const
        {
            return m_tCode;
        }

        virtual const char *what() const throw()
        {
            return IndiProperty::errorMsg(m_tCode).c_str();
        }

    private:
        IndiProperty::Error m_tCode;
    };

    /** \name Member Data
      * @{
      */
protected:

    /// The type of this object. Set on construction, it cannot be changed except by assignment.
    Type m_type{Type::Unknown};

    /// The INDI device name
    std::string m_device;

    /// The name of this property
    std::string m_name;

    /// The current state of this property
    State m_state{State::Unknown};

    /// The current message associated with this property
    std::string m_message;

    /// The permission setting of this property
    Perm m_perm{Perm::Unknown};

    /// If a switch, the switch-rule associate with this property
    SwitchRule m_rule{SwitchRule::Unknown};

    /// The UI group
    std::string m_group;

    /// The UI label
    std::string m_label;

    /// The time it takes to change this property 
    double m_timeout{0.0f};

    /// The moment when these data were valid
    pcf::TimeStamp m_timeStamp;

    /// The protocol version
    std::string m_version;

    /// A dictionary of elements, indexable by name.
    std::map<std::string, pcf::IndiElement> m_elements;

    /// This can also be the value.
    BLOBEnable m_beValue{BLOBEnable::Unknown};
    
    // A read write lock to protect the internal data.
    mutable std::shared_mutex m_rwData;

    ///@}

    /** \name Construction and Destruction
     *@{
     */
public:
    /// Constructor.
    IndiProperty();

    /// Constructor with a type. This will be used often.
    explicit IndiProperty(const Type & type /**< [in] the INDI property \ref Type*/);

    /// Constructor with a type, device and name. This will be used often.
    IndiProperty( const Type & type,          ///< [in] the INDI property \ref Type
                  const std::string & device, ///< [in] the name of the INDI device which owns this property
                  const std::string & name    ///< [in] the name of this INDI property
                );

    /// Constructor with a type, device, name, state, and perm.
    IndiProperty( const Type & type,                            ///< [in] the INDI property \ref Type
                  const std::string & device,                   ///< [in] the name of the INDI device which owns this property
                  const std::string & name,                     ///< [in] the name of this INDI property
                  const State & state,                          ///< [in] the INDI property \ref State
                  const Perm & perm,                            ///< [in] the INDI property \ref Perm
                  const SwitchRule & rule = SwitchRule::Unknown ///< [in] [optional] the INDI property \ref SwitchRule
                );

    /// Copy constructor.
    IndiProperty( const IndiProperty &ipRhs /**< [in] the IndiProperty to copy */);

    /// Destructor.
    virtual ~IndiProperty();

    ///@}
    
    /** \name Member Data Access 
      * @{
      */

    /// Get the property type
    /** 
      * \note m_type has only a get function, no set, since it can't be changed
      * 
      * \returns the current value of m_type
      */
    const Type &type() const;
    
    /// Set the device name
    void device(const std::string & dev /**< [in] the new device name */);

    /// Get the device name
    /** \returns the current value of m_device
      */
    const std::string & device() const;

    /// Check if the device name is valid
    /** The device is valid if m_device is non-zero size.
      *
      * \returns true if m_device is valid
      * \returns false if m_device is not valid
      */
    bool hasValidDevice() const;

    /// Set the property name
    void name(const std::string & na /**< [in] the new property name */);

    /// Get the property name 
    /** \returns the current value of m_name
      */
    const std::string & name() const;

    /// Check if the name is valid
    /** The name is valid if m_name is non-zero size.
      *
      * \returns true if m_name is valid
      * \returns false if m_name is not valid
      */
    bool hasValidName() const;

    /// Create the unique key for this property based on the device name and the property name. 
    /** A '.' is used as the character to join them together.
      * This key must be unique for all indi devices.
      *
      * \returns the key as "device.name"
      */
    std::string createUniqueKey() const;

    /// Set the
    void state(const State & st /**< [in]  */);

    /// Get the 
    /** \returns the current value of 
      */
    const State & state() const;

    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidState() const;

    /// Set the
    void message(const std::string & msg /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const std::string & message() const;

    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidMessage() const;

    /// Set the
    void perm(const Perm & prm);

    /// Get the  
    /** \returns the current value of
      */
    const Perm & perm() const;

    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidPerm() const;

    /// Set the
    void rule(const SwitchRule & rl /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const SwitchRule & rule() const;

    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidRule() const;

    /// Set the
    void group(const std::string & grp /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const std::string & group() const;
    
    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidGroup() const;

    /// Set the
    void label(const std::string & lbl /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const std::string &label() const;
    
    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidLabel() const;

    /// Set the
    void timeout(const double & tmo /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const double & timeout() const;
    
    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidTimeout() const;

    /// Set the
    void timeStamp(const pcf::TimeStamp &ts /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const pcf::TimeStamp & timeStamp() const;
    
    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidTimeStamp() const;

    /// Set the
    void version(const std::string &vers /**< [in]  */);

    /// Get the  
    /** \returns the current value of
      */
    const std::string & version() const;

    /// Check if the  is valid
    /** The  is valid if m_ is non-zero size.
      *
      * \returns true if m_ is valid
      * \returns false if m_ is not valid
      */
    bool hasValidVersion() const;

//-------------------------------------
    /// Set the entire map of elements.
    void elements(const std::map<std::string, pcf::IndiElement> & els);

    /// Get the entire map of elements.
    const std::map<std::string, pcf::IndiElement> & elements() const;

    unsigned int numElements() const;

    /// Adds a new element.
    /// Throws if the element already exists.
    void add(const pcf::IndiElement & el);

    /// Adds an element if it doesn't exist. If it does exist, this is a no-op.
    void addIfNoExist(const pcf::IndiElement & el);

    /// Removes an element named 'szElementName'.
    /// Throws if the element doesn't exist.
    void remove(const std::string & elName);

    ///  Returns true if the element 'szElementName' exists, false otherwise.
    bool find(const std::string & elName) const;

    /// Updates the value of an element named 'szElementName'.
    /// Throws if the element doesn't exist.
    void update(const std::string & elName,
                const pcf::IndiElement & el);

    /// Updates the value of an element, adds it if it doesn't exist.
    void update(const pcf::IndiElement & el);

    const IndiElement & at(const std::string & elName) const;

    IndiElement & at(const std::string & elName);

    /// Set the
    void beValue(const BLOBEnable & blobe);
    
    const BLOBEnable & beValue() const;

    /// Returns true if this contains a valid BLOB-enable value.
    bool hasValidBeValue() const;
    
    ///@}




    // Operators.
public:
    
    /// Assigns the internal data of this object from an existing one.
    const IndiProperty &operator=(const IndiProperty &ipRhs);

    /// This is an alternate way of calling 'setBLOBEnable'.
    const BLOBEnable &operator=(const BLOBEnable &tValue);
    
    /// Returns true if we have an exact match (value as well).
    bool operator==(const IndiProperty &ipRhs) const;
    
    // Return a reference to an element so it can be modified.
    const IndiElement &operator[](const std::string &szName) const;
    
    IndiElement &operator[](const std::string &szName);
 
    // Return a reference to an element so it can be modified.
    const IndiElement &operator[](const unsigned int &uiIndex) const;
    
    IndiElement &operator[](const unsigned int &uiIndex);

    /** \name General Methods.
      * @{
      */ 

public:
    /// Reset this object.
    void clear();
    
    /// Get the string name of a property \ref Type
    /** 
      * \returns 
      */
    static std::string type2String( const Type & type /**< [in] the \ref Type to convert*/ );

    /// Get the property \ref Type given its string name
    /** 
      * \returns 
      */
    static Type string2Type( const std::string & str /**< [in] the string to convert */ );
    
    /// Get the string name of the given \ref BLOBEnable
    /** 
      * \returns 
      */
    static std::string BLOBEnable2String( const BLOBEnable & type /**<[in] the \ref BLOBEnable to convert */ );

    /// Get the \ref BLOBEnable given its string name.
    /** 
      * \returns 
      */
    static BLOBEnable string2BLOBEnable( const std::string & str /**< [in] the string to convert */ );

    /// Get the string name of the given \ref Perm.
    /** 
      * \returns 
      */
    static std::string permission2String( const Perm & perm /**< [in] the \ref Perm to convert*/ );
    
    /// Get the \ref Perm given the string name.
    /** 
      * \returns 
      */
    static Perm string2Permission( const std::string & str  /**< [in] the string to convert */ );

    /// Get the string name of the state.
    /** 
      * \returns 
      */
    static std::string state2String( const State &state /**< [in] the \ref State to convert*/ );
    
    /// Get the state string name of the \ref State.
    /** 
      * \returns 
      */
    static State string2State( const std::string & str  /**< [in] the string to convert */ );

    /// Get the string name of a \ref SwitchRule.
    /** 
      * \returns 
      */
    static std::string switchRule2String( const SwitchRule & rule /**< [in] the \ref SwitchRule to convert*/ );

    /// Get the \ref SwitchRule given the string name.
    /** 
      * \returns 
      */
    static SwitchRule string2SwitchRule( const std::string & str /**< [in] the string to convert */ );

    /// Get the message concerning the error.
    /** 
      * \returns 
      */
    static std::string errorMsg( const Error & err /**< [in] the \ref Error to convert*/ );

    /// Ensures that a name conforms to the INDI standard and can be used as
    /// an identifier. This means:
    ///     1) No ' ' - these will be converted to '_'
    ///     2) No '.' - these will be converted to '___'
    ///     3) No Unprintable chars - these will be converted to '_'
    static std::string scrubName( const std::string &szName);

    ///@}

    
    
    // Element functions.
public:
    

}; // class IndiProperty

} // namespace pcf

#endif // libcommon_IndiProperty_hpp
