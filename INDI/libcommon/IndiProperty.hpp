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
#include "ReadWriteLock.hpp"
#include "TimeStamp.hpp"
#include "IndiElement.hpp"

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

    enum class PropertyState
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

    enum class PropertyPerm
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
            return IndiProperty::getErrorMsg(m_tCode).c_str();
        }

    private:
        IndiProperty::Error m_tCode;
    };

    /** \name Member Data
      * @{
      */
protected:

    std::string m_device;

    std::string m_group;

    std::string m_label;

    std::string m_message;

    std::string m_name;

    PropertyPerm m_perm{PropertyPerm::Unknown};

    SwitchRule m_rule{SwitchRule::Unknown};

    PropertyState m_state{PropertyState::Unknown};

    double m_timeout{0.0f};

    // This is a flag which can be used to show that this property
    // has been requested by a client. This is not managed automatically.
    bool m_requested{false};

    pcf::TimeStamp m_timeStamp;

    std::string m_version;

    /// This can also be the value.
    BLOBEnable m_beValue{BLOBEnable::Unknown};

    /// A dictionary of elements, indexable by name.
    std::map<std::string, pcf::IndiElement> m_mapElements;

    /// The type of this object. It cannot be changed.
    Type m_type{Type::Unknown};

    // A read write lock to protect the internal data.
    mutable pcf::ReadWriteLock m_rwData;


    ///@}

    /** \name Construction and Destruction
     *@{
     */
public:
    /// Constructor.
    IndiProperty();

    /// Constructor with a type. This will be used often.
    explicit IndiProperty(const Type &tType);

    /// Constructor with a type, device and name. This will be used often.
    IndiProperty( const Type &tType,
                  const std::string &szDevice,
                  const std::string &szName);

    /// Constructor with a type, device, name, state, and perm.
    IndiProperty( const Type &tType,
                  const std::string &szDevice,
                  const std::string &szName,
                  const PropertyState &tState,
                  const PropertyPerm &tPerm,
                  const SwitchRule &tRule = SwitchRule::Unknown
                );

    /// Copy constructor.
    IndiProperty( const IndiProperty &ipRhs );

    /// Destructor.
    virtual ~IndiProperty();

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

    // Methods.
public:
    /// Reset this object.
    void clear();
    /// Compares one property with another instance. The values may not match,
    /// but the type must match, as must the device and name. The names of all
    /// the elements must match as well.
    bool compareProperty(const IndiProperty &ipComp) const;
    /// Compares one element value contained in this class with another
    /// instance. The type must match, as must the device and name.
    /// The name of this element must match as well.
    bool compareValue(const IndiProperty &ipComp,
                      const std::string &szElementName) const;
    /// Compares all the element values contained in this class with another
    /// instance. The type must match, as must the device and name. The number
    /// and names of all the elements must match as well.
    bool compareValues(const IndiProperty &ipComp) const;
    /// Returns a string with each attribute enumerated.
    std::string createString() const;
    /// Create a name for this property based on the device name and the
    /// property name. A '.' is used as the character to join them together.
    /// This key should be unique for all indi devices.
    std::string createUniqueKey() const;

    // A getter for blob enable.
    const BLOBEnable &getBLOBEnable() const;

    // A getter for each attribute.
    const std::string &getDevice() const;
    const std::string &getGroup() const;
    const std::string &getLabel() const;
    const std::string &getMessage() const;
    const std::string &getName() const;
    const PropertyPerm &getPerm() const;
    const SwitchRule &getRule() const;

    const PropertyState &getState() const;
    
    const double &getTimeout() const;
    const pcf::TimeStamp &getTimeStamp() const;
    /// There is only a 'getter' for the type, since it can't be changed.
    const Type &getType() const;
    const std::string &getVersion() const;
    /// Compares one element value contained in this class with another
    /// instance. The type must match, as must the device and name.
    /// The name of this element must match as well, and the value must be a
    /// new, non-blank value.
    bool hasNewValue(const IndiProperty &ipComp,
                     const std::string &szElementName) const;
    const bool &isRequested() const;

    /// Returns the string type given the enumerated type.
    /// Throws exception if string is not found.
    static std::string convertTypeToString(const Type &tType);
    /// Returns the enumerated type given the tag.
    static Type convertStringToType(const std::string &szTag);
    /// Returns the string type given the enumerated type.
    static std::string getBLOBEnableString(const BLOBEnable &tType);
    /// Returns the enumerated type given the string type.
    static BLOBEnable getBLOBEnable(const std::string &szType);
    /// Returns the string type given the enumerated type.
    static std::string getPropertyPermString(const PropertyPerm &tType);
    /// Returns the enumerated type given the string type.
    static PropertyPerm getPropertyPerm(const std::string &szType);
    /// Returns the string type given the enumerated type.
    static std::string getPropertyStateString(const PropertyState &tType);
    /// Returns the enumerated type given the string type.
    static PropertyState getPropertyState(const std::string &szType);
    /// Returns the string type given the enumerated type.
    static std::string getSwitchRuleString(const SwitchRule &tType);
    /// Returns the enumerated type given the string type.
    static SwitchRule getSwitchRule(const std::string &szType);
    /// Returns the message concerning the error.
    static std::string getErrorMsg(const Error &nErr);
    /// Ensures that a name conforms to the INDI standard and can be used as
    /// an identifier. This means:
    ///     1) No ' ' - these will be converted to '_'
    ///     2) No '.' - these will be converted to '___'
    ///     3) No Unprintable chars - these will be converted to '_'
    static std::string scrubName(const std::string &szName);

    /// Returns the number of elements.
    unsigned int getNumElements() const;

    /// Returns true if this contains a valid BLOB-enable value.
    bool hasValidBLOBEnable() const;
    /// Returns true if this contains a non-empty 'device' attribute.
    bool hasValidDevice() const;
    /// Returns true if this contains a non-empty 'group' attribute.
    bool hasValidGroup() const;
    /// Returns true if this contains a non-empty 'label' attribute.
    bool hasValidLabel() const;
    /// Returns true if this contains a non-empty 'message' attribute.
    bool hasValidMessage() const;
    /// Returns true if this contains a non-empty 'name' attribute.
    bool hasValidName() const;
    /// Returns true if this contains a non-empty 'perm' attribute.
    bool hasValidPerm() const;
    /// Returns true if this contains a non-empty 'rule' attribute.
    bool hasValidRule() const;
    /// Returns true if this contains a non-empty 'state' attribute.
    bool hasValidState() const;
    /// Returns true if this contains a non-empty 'timeout' attribute.
    bool hasValidTimeout() const;
    /// Returns true if this contains a non-empty 'timestamp' attribute.
    bool hasValidTimeStamp() const;
    /// Returns true if this contains a non-empty 'version' attribute.
    bool hasValidVersion() const;

    /// All the attribute setters.
    void setBLOBEnable(const BLOBEnable &tValue);
    void setDevice(const std::string &szValue);
    void setGroup(const std::string &szValue);
    void setLabel(const std::string &szValue);
    void setMessage(const std::string &szValue);
    void setName(const std::string &szValue);
    void setPerm(const PropertyPerm &tValue);
    void setRequested(const bool &oRequested);
    void setRule(const SwitchRule &tValue);
    void setState(const PropertyState &tValue);
    void setTimeout(const double &xValue);
    void setTimeStamp(const pcf::TimeStamp &tsValue);
    void setVersion(const std::string &szValue);

    // Element functions.
public:
    // Return a reference to an element so it can be modified.
    const IndiElement &at(const std::string &szName) const;
    IndiElement &at(const std::string &szName);
    // Return a reference to an element so it can be modified.
    const IndiElement &at(const unsigned int &uiIndex) const;
    IndiElement &at(const unsigned int &uiIndex);
    /// Adds a new element.
    /// Throws if the element already exists.
    void add(const pcf::IndiElement &ieNew);
    /// Adds an element if it doesn't exist. If it does exist, this is a no-op.
    void addIfNoExist(const pcf::IndiElement &ieNew);
    ///  Returns true if the element 'szElementName' exists, false otherwise.
    bool find(const std::string &szElementName) const;
    /// Get the entire map of elements.
    const std::map<std::string, pcf::IndiElement> &getElements() const;
    /// Removes an element named 'szElementName'.
    /// Throws if the element doesn't exist.
    void remove(const std::string &szElementName);
    /// Set the entire map of elements.
    void setElements(const std::map<std::string, pcf::IndiElement> &mapElements);
    /// Updates the value of an element named 'szElementName'.
    /// Throws if the element doesn't exist.
    void update(const std::string &szElementName,
                const pcf::IndiElement &ieUpdate);
    /// Updates the value of an element, adds it if it doesn't exist.
    void update(const pcf::IndiElement &ieNew);


}; // class IndiProperty

} // namespace pcf

#endif // libcommon_IndiProperty_hpp
