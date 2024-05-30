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
    enum class Error
    {
        None = 0,
        CouldntFindElement = -3,
        ElementAlreadyExists = -5,
        IndexOutOfBounds = -6,
        WrongElementType = -7,
        Undefined = -9999
    };

    enum class BLOBEnable
    {
        Unknown = 0,
        Also = 1,
        Only,
        Never
    };

    enum class State
    {
        Unknown = 0,
        Alert = 1,
        Busy,
        Ok,
        Idle
    };

    enum class SwitchRule
    {
        Unknown = 0,
        AnyOfMany = 1,
        AtMostOne,
        OneOfMany
    };

    enum class Perm
    {
        Unknown = 0,
        ReadOnly = 1,
        ReadWrite,
        WriteOnly
    };

    // These are the types that a property can be.
    // The order and enumeration of this list is important.
    // Do not add or change enumerations here without adjusting
    // the indexing of the 'allowed attributes' list.
    enum class Type
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
    explicit IndiProperty(const Type & type);

    /// Constructor with a type, device and name. This will be used often.
    IndiProperty( const Type & type,
                  const std::string & device,
                  const std::string & name
                );

    /// Constructor with a type, device, name, state, and perm.
    IndiProperty( const Type & type,
                  const std::string & device,
                  const std::string & name,
                  const State & state,
                  const Perm & perm,
                  const SwitchRule & rule = SwitchRule::Unknown
                );

    /// Copy constructor.
    IndiProperty( const IndiProperty &ipRhs );

    /// Destructor.
    virtual ~IndiProperty();

    ///@}
    
    /** \name Member Data Access 
      * @{
      */

    /// There is only a 'getter' for the type, since it can't be changed.
    const Type &type() const;
    
    void device(const std::string & dev);

    const std::string & device() const;

    /// Returns true if this contains a non-empty 'device' attribute.
    bool hasValidDevice() const;

    void name(const std::string & na);

    const std::string & name() const;

    /// Returns true if this contains a non-empty 'name' attribute.
    bool hasValidName() const;

    /// Create the unique key for this property based on the device name and the property name. 
    /** A '.' is used as the character to join them together.
      * This key must be unique for all indi devices.
      *
      * \returns the key as "device.name"
      */
    std::string createUniqueKey() const;

    void state(const State & st);

    const State & state() const;

    /// Returns true if this contains a non-empty 'state' attribute.
    bool hasValidState() const;

    void message(const std::string & msg);

    const std::string & message() const;

    /// Returns true if this contains a non-empty 'message' attribute.
    bool hasValidMessage() const;

    void perm(const Perm & prm);

    const Perm & perm() const;

    /// Returns true if this contains a non-empty 'perm' attribute.
    bool hasValidPerm() const;

    void rule(const SwitchRule & rl);

    const SwitchRule & rule() const;

    /// Returns true if this contains a non-empty 'rule' attribute.
    bool hasValidRule() const;

    void group(const std::string & grp);

    const std::string & group() const;
    
    /// Returns true if this contains a non-empty 'group' attribute.
    bool hasValidGroup() const;

    void label(const std::string & lbl);

    const std::string &label() const;
    
    /// Returns true if this contains a non-empty 'label' attribute.
    bool hasValidLabel() const;

    void timeout(const double & tmo);

    const double & timeout() const;
    
    /// Returns true if this contains a non-empty 'timeout' attribute.
    bool hasValidTimeout() const;

    void timeStamp(const pcf::TimeStamp &ts);

    const pcf::TimeStamp & timeStamp() const;
    
    /// Returns true if this contains a non-empty 'timestamp' attribute.
    bool hasValidTimeStamp() const;

    void version(const std::string &vers);

    const std::string & version() const;

    /// Returns true if this contains a non-empty 'version' attribute.
    bool hasValidVersion() const;

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

    /// All the attribute setters.
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
