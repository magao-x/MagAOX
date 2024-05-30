/// IndiProperty.cpp
///
/// @author Paul Grenz
///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include "IndiProperty.hpp"

namespace pcf
{

IndiProperty::IndiProperty()
{
}

IndiProperty::IndiProperty(const Type & type) : m_type(type)
{
}

IndiProperty::IndiProperty(const Type & type,
                           const std::string & device,
                           const std::string & name) : m_type(type), m_device(device), m_name(name)
{
}

IndiProperty::IndiProperty(const Type & type,
                           const std::string & device,
                           const std::string & name,
                           const State & state,
                           const Perm & perm,
                           const SwitchRule & rule) : m_type(type), m_device(device), m_name(name), 
                                                       m_state(state), m_perm(perm), m_rule(rule)
{
}

IndiProperty::IndiProperty(const IndiProperty &ipRhs) : m_type(ipRhs.m_type), m_device(ipRhs.m_device), m_name(ipRhs.m_name),
                                                        m_state(ipRhs.m_state), m_message(ipRhs.m_message), m_perm(ipRhs.m_perm),
                                                        m_rule(ipRhs.m_rule), m_group(ipRhs.m_group), m_label(ipRhs.m_label),                                      
                                                        m_timeout(ipRhs.m_timeout),
                                                        /*m_requested(ipRhs.m_requested),*/ m_timeStamp(ipRhs.m_timeStamp),
                                                        m_version(ipRhs.m_version), m_elements(ipRhs.m_elements), m_beValue(ipRhs.m_beValue)
{
}

IndiProperty::~IndiProperty()
{
}

const IndiProperty::Type &IndiProperty::type() const
{
    std::shared_lock rLock(m_rwData);
    return m_type;
}

void IndiProperty::device(const std::string & dev)
{
    std::unique_lock wLock(m_rwData);
    m_device = dev;
}

const std::string &IndiProperty::device() const
{
    std::shared_lock rLock(m_rwData);
    return m_device;
}

bool IndiProperty::hasValidDevice() const
{
    std::shared_lock rLock(m_rwData);
    return (m_device.size() != 0);
}

void IndiProperty::name(const std::string & nm)
{
    std::unique_lock wLock(m_rwData);
    m_name = nm;
}

const std::string &IndiProperty::name() const
{
    std::shared_lock rLock(m_rwData);
    return m_name;
}

bool IndiProperty::hasValidName() const
{
    std::shared_lock rLock(m_rwData);
    return (m_name.size() != 0);
}

std::string IndiProperty::createUniqueKey() const
{
    std::shared_lock rLock(m_rwData);
    return m_device + "." + m_name;
}

void IndiProperty::state(const IndiProperty::State & st)
{
    std::unique_lock wLock(m_rwData);
    m_state = st;
}

const IndiProperty::State &IndiProperty::state() const
{
    std::shared_lock rLock(m_rwData);
    return m_state;
}

bool IndiProperty::hasValidState() const
{
    std::shared_lock rLock(m_rwData);
    return (m_state != State::Unknown);
}

void IndiProperty::message(const std::string & msg)
{
    std::unique_lock wLock(m_rwData);
    m_message = msg;
}

const std::string &IndiProperty::message() const
{
    std::shared_lock rLock(m_rwData);
    return m_message;
}

bool IndiProperty::hasValidMessage() const
{
    std::shared_lock rLock(m_rwData);
    return (m_message.size() != 0);
}

void IndiProperty::perm(const IndiProperty::Perm & prm)
{
    std::unique_lock wLock(m_rwData);
    m_perm = prm;
}

const IndiProperty::Perm & IndiProperty::perm() const
{
    std::shared_lock rLock(m_rwData);
    return m_perm;
}

bool IndiProperty::hasValidPerm() const
{
    std::shared_lock rLock(m_rwData);
    return (m_perm != Perm::Unknown);
}

void IndiProperty::rule(const IndiProperty::SwitchRule & rl)
{
    std::unique_lock wLock(m_rwData);
    m_rule = rl;
}

const IndiProperty::SwitchRule &IndiProperty::rule() const
{
    std::shared_lock rLock(m_rwData);
    return m_rule;
}

bool IndiProperty::hasValidRule() const
{
    std::shared_lock rLock(m_rwData);
    return (m_rule != SwitchRule::Unknown);
}

void IndiProperty::group(const std::string & grp)
{
    std::unique_lock wLock(m_rwData);
    m_group = grp;
}

const std::string &IndiProperty::group() const
{
    std::shared_lock rLock(m_rwData);
    return m_group;
}

bool IndiProperty::hasValidGroup() const
{
    std::shared_lock rLock(m_rwData);
    return (m_group.size() != 0);
}

void IndiProperty::label(const std::string & lbl)
{
    std::unique_lock wLock(m_rwData);
    m_label = lbl;
}

const std::string &IndiProperty::label() const
{
    std::shared_lock rLock(m_rwData);
    return m_label;
}

bool IndiProperty::hasValidLabel() const
{
    std::shared_lock rLock(m_rwData);
    return (m_label.size() != 0);
}

void IndiProperty::timeout(const double & tmo)
{
    std::unique_lock wLock(m_rwData);
    m_timeout = tmo;
}

const double &IndiProperty::timeout() const
{
    std::shared_lock rLock(m_rwData);
    return m_timeout;
}

bool IndiProperty::hasValidTimeout() const
{
    std::shared_lock rLock(m_rwData);
    return (m_timeout != 0.0f);
}

void IndiProperty::timeStamp(const TimeStamp & ts)
{
    std::unique_lock wLock(m_rwData);
    m_timeStamp = ts;
}

const TimeStamp &IndiProperty::timeStamp() const
{
    std::shared_lock rLock(m_rwData);
    return m_timeStamp;
}

bool IndiProperty::hasValidTimeStamp() const
{
    // todo: Timestamp is always valid.... this is a weak point.
    std::shared_lock rLock(m_rwData);
    return true;
}


void IndiProperty::version(const std::string & vers)
{
    std::unique_lock wLock(m_rwData);
    m_version = vers;
}

const std::string &IndiProperty::version() const
{
    std::shared_lock rLock(m_rwData);
    return m_version;
}

bool IndiProperty::hasValidVersion() const
{
    std::shared_lock rLock(m_rwData);
    return (m_version.size() != 0);
}

void IndiProperty::elements(const std::map<std::string, IndiElement> & els)
{
    std::unique_lock wLock(m_rwData);
    m_elements = els;
}

const std::map<std::string, IndiElement> &IndiProperty::elements() const
{
    std::unique_lock wLock(m_rwData);
    return m_elements;
}

unsigned int IndiProperty::numElements() const
{
    std::shared_lock rLock(m_rwData);
    return (m_elements.size());
}

void IndiProperty::add(const IndiElement & el)
{
    std::unique_lock wLock(m_rwData);

    std::map<std::string, IndiElement>::const_iterator itr = m_elements.find(el.name());

    if (itr != m_elements.end())
    {
        throw Excep(Error::ElementAlreadyExists);
    }

    // Actually add it to the map.
    m_elements[el.name()] = el;

    m_timeStamp = TimeStamp::now();
}

void IndiProperty::addIfNoExist(const IndiElement & el)
{
    std::unique_lock wLock(m_rwData);

    std::map<std::string, IndiElement>::const_iterator itr = m_elements.find(el.name());

    if (itr == m_elements.end())
    {
        // Actually add it to the map.
        m_elements[el.name()] = el;
        m_timeStamp = TimeStamp::now();
    }
}

void IndiProperty::remove(const std::string & elName)
{
    std::unique_lock wLock(m_rwData);

    std::map<std::string, IndiElement>::iterator itr = m_elements.find(elName);

    if (itr == m_elements.end())
    {
        throw Excep(Error::CouldntFindElement);
    }

    // Actually delete the element.
    m_elements.erase(itr);
}

bool IndiProperty::find(const std::string & elName) const
{
    std::shared_lock rLock(m_rwData);

    std::map<std::string, IndiElement>::const_iterator itr = m_elements.find(elName);

    return (itr != m_elements.end());
}

void IndiProperty::update(const IndiElement & elName)
{
    std::unique_lock wLock(m_rwData);

    // Actually add it to the map, or update it.
    m_elements[elName.name()] = elName;

    m_timeStamp = TimeStamp::now();
}

void IndiProperty::update(const std::string & elName,
                          const IndiElement & el)
{
    std::unique_lock wLock(m_rwData);

    std::map<std::string, IndiElement>::iterator itr = m_elements.find(elName);

    if (itr == m_elements.end())
    {
        throw Excep(Error::CouldntFindElement);
    }

    itr->second = el;

    m_timeStamp = TimeStamp::now();
}

const IndiElement &IndiProperty::at(const std::string & elName) const
{
    std::shared_lock rLock(m_rwData);

    std::map<std::string, IndiElement>::const_iterator itr = m_elements.find(elName);

    if (itr == m_elements.end())
    {
        throw std::runtime_error(std::string("Element name '") + elName + "' not found.");
    }

    return itr->second;
}

IndiElement &IndiProperty::at(const std::string & elName)
{
    std::unique_lock wLock(m_rwData);

    std::map<std::string, IndiElement>::iterator itr = m_elements.find(elName);

    if (itr == m_elements.end())
    {
        throw std::runtime_error(std::string("Element name '") + elName + "' not found.");
    }

    return itr->second;
}

void IndiProperty::beValue(const BLOBEnable & blobe)
{
    std::unique_lock wLock(m_rwData);
    m_beValue = blobe;
}

const IndiProperty::BLOBEnable &IndiProperty::beValue() const
{
    std::shared_lock rLock(m_rwData);
    return m_beValue;
}

bool IndiProperty::hasValidBeValue() const
{
    std::shared_lock rLock(m_rwData);
    return (m_beValue != BLOBEnable::Unknown);
}




const IndiProperty &IndiProperty::operator=(const IndiProperty &ipRhs)
{
    if (&ipRhs != this)
    {
        std::unique_lock wLock(m_rwData);

        m_device = ipRhs.m_device;
        m_group = ipRhs.m_group;
        m_label = ipRhs.m_label;
        m_message = ipRhs.m_message;
        m_name = ipRhs.m_name;
        m_perm = ipRhs.m_perm;
        //m_requested = ipRhs.m_requested;
        m_rule = ipRhs.m_rule;
        m_state = ipRhs.m_state;
        m_timeout = ipRhs.m_timeout;
        m_timeStamp = ipRhs.m_timeStamp;
        m_version = ipRhs.m_version;
        m_beValue = ipRhs.m_beValue;

        m_elements = ipRhs.m_elements;
        m_type = ipRhs.m_type;
    }
    return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// This is an alternate way of calling 'setBLOBEnable'.

const IndiProperty::BLOBEnable &IndiProperty::operator=(const BLOBEnable &tValue)
{
    std::unique_lock wLock(m_rwData);
    m_beValue = tValue;
    return tValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if we have an exact match (value as well).

bool IndiProperty::operator==(const IndiProperty &ipRhs) const
{
    std::shared_lock rLock(m_rwData);

    // If we are comparing ourself to ourself - easy!
    if (&ipRhs == this)
        return true;

    // If they are different sizes, they are different.
    if (ipRhs.m_elements.size() != m_elements.size())
        return false;

    // We need some iterators for each of the maps.
    std::map<std::string, IndiElement>::const_iterator itrRhs = ipRhs.m_elements.end();
    std::map<std::string, IndiElement>::const_iterator itr = m_elements.begin();
    for (; itr != m_elements.end(); ++itr)
    {
        // Can we find an element of the same name in the other map?
        itrRhs = ipRhs.m_elements.find(itr->first);

        // If we can't find the name, these are different.
        if (itrRhs == ipRhs.m_elements.end())
            return false;

        // If we found it, and they don't match, these are different.
        if (!(itrRhs->second == itr->second))
            return false;
    }

    // Otherwise the maps are identical and it comes down to the
    // attributes here matching.
    return (m_device == ipRhs.m_device &&
            m_group == ipRhs.m_group &&
            m_label == ipRhs.m_label &&
            m_message == ipRhs.m_message &&
            m_name == ipRhs.m_name &&
            m_perm == ipRhs.m_perm &&
            // m_requested == ipRhs.m_requested &&
            m_rule == ipRhs.m_rule &&
            m_state == ipRhs.m_state &&
            // m_timeout == ipRhs.m_timeout &&
            // m_timeStamp ==ipRhs.m_timeStamp &&  // Don't compare!
            m_version == ipRhs.m_version &&
            m_beValue == ipRhs.m_beValue &&
            m_type == ipRhs.m_type);
}

////////////////////////////////////////////////////////////////////////////////
/// Ensures that a name conforms to the INDI standard and can be used as
/// an identifier. This means:
///     1) No ' ' - these will be converted to '_'.
///     2) No '.' - these will be converted to '___'.
///     3) No Unprintable chars - these will be converted to '_'.

std::string IndiProperty::scrubName(const std::string &szName)
{
    std::string szScrubbed(szName);

    // We are replacing one char with multiple chars, so we have to do it
    // the long way first.
    size_t pp = 0;
    size_t rr = 0;
    while ((rr = szScrubbed.find('.', pp)) != std::string::npos)
    {
        szScrubbed.replace(rr, 1, "___");
        pp = rr + 1;
    }

    // These are one-for-one replacements, so we can do them in-place.
    std::replace_if(szScrubbed.begin(), szScrubbed.end(),
                    std::not1(std::ptr_fun(::isalnum)), '_');

    return szScrubbed;
}

////////////////////////////////////////////////////////////////////////////////
/// Compares one property with another instance. The values may not match,
/// but the type must match, as must the device and name. The names of all the
/// elements must match as well.

/*bool IndiProperty::compareProperty(const IndiProperty &ipComp) const
{
    std::shared_lock rLock(m_rwData);

    // If we are comparing ourself to ourself - easy!
    if (&ipComp == this)
        return true;

    if (ipComp.type() != m_type)
        return false;

    if (ipComp.device() != m_device || ipComp.name() != m_name)
        return false;

    // If they are different sizes, they are different.
    if (ipComp.m_elements.size() != m_elements.size())
        return false;

    // We need some iterators for each of the maps.
    std::map<std::string, IndiElement>::const_iterator itrComp = ipComp.m_elements.end();
    std::map<std::string, IndiElement>::const_iterator itr = m_elements.begin();
    for (; itr != m_elements.end(); ++itr)
    {
        // Can we find an element of the same name in the other map?
        itrComp = ipComp.m_elements.find(itr->first);

        // If we can't find the name, these are different.
        if (itrComp == ipComp.m_elements.end())
            return false;
    }

    // If we got here, we are identical.
    return true;
}*/

////////////////////////////////////////////////////////////////////////////////
/// Compares one element value contained in this class with another
/// instance. The type must match, as must the device and name.
/// The name of this element must match as well.

/*bool IndiProperty::compareValue(const IndiProperty &ipComp,
                                const std::string &szElementName) const
{
    std::shared_lock rLock(m_rwData);

    // If we are comparing ourself to ourself - easy!
    if (&ipComp == this)
        return true;

    if (ipComp.type() != m_type)
        return false;

    if (ipComp.device() != m_device || ipComp.name() != m_name)
        return false;

    // Can we find this element in this map? If not, we fail.
    std::map<std::string, IndiElement>::const_iterator itr =
        m_elements.find(szElementName);
    if (itr == m_elements.end())
        return false;

    // Can we find this element in the other map? If not, we fail.
    std::map<std::string, IndiElement>::const_iterator itrComp =
        ipComp.m_elements.find(szElementName);
    if (itrComp == ipComp.m_elements.end())
        return false;

    // If we found it, and the values don't match, these are different.
    if (itr->second.value() != itrComp->second.value())
        return false;

    // If we got here, we are identical.
    return true;
}*/

////////////////////////////////////////////////////////////////////////////////
/// Compares all the element values contained in this class with another
/// instance. The type must match, as must the device and name. The number
/// and names of all the elements must match as well.

/*bool IndiProperty::compareValues(const IndiProperty &ipComp) const
{
    std::shared_lock rLock(m_rwData);

    // If we are comparing ourself to ourself - easy!
    if (&ipComp == this)
        return true;

    if (ipComp.type() != m_type)
        return false;

    if (ipComp.device() != m_device || ipComp.name() != m_name)
        return false;

    // If they are different sizes, they are different.
    if (ipComp.m_elements.size() != m_elements.size())
        return false;

    // We need some iterators for each of the maps.
    std::map<std::string, IndiElement>::const_iterator itrComp = ipComp.m_elements.end();
    std::map<std::string, IndiElement>::const_iterator itr = m_elements.begin();
    for (; itr != m_elements.end(); ++itr)
    {
        // Can we find an element of the same name in the other map?
        itrComp = ipComp.m_elements.find(itr->first);

        // If we can't find the name, these are different.
        if (itrComp == ipComp.m_elements.end())
            return false;

        // If we found it, and the values don't match, these are different.
        if (itrComp->second.value() != itr->second.value())
            return false;
    }

    // If we got here, we are identical.
    return true;
}*/


////////////////////////////////////////////////////////////////////////////////
/// Returns a std::string with each attribute enumerated.

/*std::string IndiProperty::createString() const
{
    std::shared_lock rLock(m_rwData);

    std::stringstream ssOutput;
    ssOutput << "{ "
             << "\"device\" : \"" << m_device << "\" , "
             << "\"name\" : \"" << m_name << "\" , "
             << "\"type\" : \"" << convertTypeToString(m_type) << "\" , "
             << "\"group\" : \"" << m_group << "\" , "
             << "\"label\" : \"" << m_label << "\" , "
             << "\"timeout\" : \"" << m_timeout << "\" , "
             << "\"version\" : \"" << m_version << "\" , "
             << "\"timestamp\" : \"" << m_timeStamp.getFormattedIso8601Str() << "\" , "
             << "\"perm\" : \"" << getPermString(m_perm) << "\" , "
             << "\"rule\" : \"" << getSwitchRuleString(m_rule) << "\" , "
             << "\"state\" : \"" << getStateString(m_state) << "\" , "
             << "\"BLOBenable\" : \"" << getBLOBEnableString(m_beValue) << "\" , "
             << "\"message\" : \"" << m_message << "\" "
             << "\"elements\" : [ \n";

    std::map<std::string, IndiElement>::const_iterator itr = m_elements.begin();
    for (; itr != m_elements.end(); ++itr)
    {
        ssOutput << "    ";
        if (itr != m_elements.begin())
            ssOutput << " , ";
        ssOutput << itr->second.createString();
    }

    ssOutput << "\n] "
             << " } ";

    return ssOutput.str();
}*/

////////////////////////////////////////////////////////////////////////////////
/// Create a name for this property based on the device name and the
/// property name. A '.' is used as the chracter to join them together.
/// This key should be unique for all indi devices.







void IndiProperty::clear()
{
    std::unique_lock wLock(m_rwData);
    m_elements.clear();
}

////////////////////////////////////////////////////////////////////////////////

const IndiElement &IndiProperty::operator[](const std::string &szName) const
{
    std::shared_lock rLock(m_rwData);
    std::map<std::string, IndiElement>::const_iterator itr = m_elements.find(szName);

    if (itr == m_elements.end())
        throw std::runtime_error(std::string("Element name '") + szName + "' not found.");

    return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element named szName.
/// Throws exception if name is not found.

IndiElement &IndiProperty::operator[](const std::string &szName)
{
    std::unique_lock wLock(m_rwData);
    std::map<std::string, IndiElement>::iterator itr = m_elements.find(szName);

    if (itr == m_elements.end())
        throw std::runtime_error(std::string("Element name '") + szName + "' not found.");

    return itr->second;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the element at an index (zero-based).
/// Throws exception if name is not found.

const IndiElement &IndiProperty::operator[](const unsigned int &uiIndex) const
{
    std::shared_lock rLock(m_rwData);

    if (uiIndex > m_elements.size() - 1)
        throw Excep(Error::IndexOutOfBounds);

    std::map<std::string, IndiElement>::const_iterator itr = m_elements.begin();
    std::advance(itr, uiIndex);

    return itr->second;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the element at an index (zero-based).
/// Throws exception if name is not found.

IndiElement &IndiProperty::operator[](const unsigned int &uiIndex)
{
    std::unique_lock wLock(m_rwData);

    if (uiIndex > m_elements.size() - 1)
        throw Excep(Error::IndexOutOfBounds);

    std::map<std::string, IndiElement>::iterator itr = m_elements.begin();
    std::advance(itr, uiIndex);

    return itr->second;
}


std::string IndiProperty::errorMsg(const Error &nErr)
{
    std::string szMsg;
    switch (nErr)
    {
    //  errors defined in this class.
    case Error::None:
        szMsg = "No Error";
        break;
    case Error::CouldntFindElement:
        szMsg = "Could not find element";
        break;
    case Error::ElementAlreadyExists:
        szMsg = "Element already exists";
        break;
    case Error::IndexOutOfBounds:
        szMsg = "Index out of bounds";
        break;
    default:
        szMsg = "Unknown error";
        break;
    }
    return szMsg;
}

IndiProperty::BLOBEnable IndiProperty::string2BLOBEnable( const std::string & str )
{
    BLOBEnable type = BLOBEnable::Unknown;

    if (str == "Never")
    {
        type = BLOBEnable::Never;
    }
    else if (str == "Also")
    {
        type = BLOBEnable::Also;
    }
    else if (str == "Only")
    {
        type = BLOBEnable::Only;
    }

    return type;
}

std::string IndiProperty::BLOBEnable2String( const BLOBEnable & type )
{
    switch (type)
    {
        case BLOBEnable::Unknown:
            return std::string("");
        case BLOBEnable::Never:
            return std::string("Never");
        case BLOBEnable::Also:
            return std::string("Also");
        case BLOBEnable::Only:
            return std::string("Only");
        default:
            return std::string("");
    }

}

IndiProperty::State IndiProperty::string2State(const std::string &str)
{
    if(str == "Idle")
    {
        return State::Idle;
    }
    else if(str == "Ok")
    {
        return State::Ok;
    }
    else if(str == "Busy")
    {
        return State::Busy;
    }
    else if(str == "Alert")
    {
        return State::Alert;
    }
    else 
    {
        return State::Unknown;
    }
}

std::string IndiProperty::state2String(const State & state)
{
    switch (state)
    {
        case State::Unknown:
            return std::string("");
        case State::Idle:
            return std::string("Idle");
        case State::Ok:
            return std::string("Ok");
        case State::Busy:
           return std::string("Busy");
        case State::Alert:
            return std::string("Alert");
        default:
            return std::string("");
    }
}

IndiProperty::SwitchRule IndiProperty::string2SwitchRule(const std::string &str)
{

    if (str == "OneOfMany")
    {
        return SwitchRule::OneOfMany;
    }
    else if (str == "AtMostOne")
    {
        return SwitchRule::AtMostOne;
    }
    else if (str == "AnyOfMany")
    {
        return SwitchRule::AnyOfMany;
    }
    else 
    {
        return SwitchRule::Unknown;
    }

}

std::string IndiProperty::switchRule2String(const SwitchRule & rule)
{
    switch (rule)
    {
        case SwitchRule::OneOfMany:
            return std::string("OneOfMany");
            break;
        case SwitchRule::AtMostOne:
            return std::string("AtMostOne");
            break;
        case SwitchRule::AnyOfMany:
            return std::string("AnyOfMany");
            break;
        case SwitchRule::Unknown:
            return std::string("");
        default:
            return std::string("");
    }
}

IndiProperty::Perm IndiProperty::string2Permission(const std::string &str)
{
    if (str == "ro")
    {
        return Perm::ReadOnly;
    }
    else if (str == "wo")
    {
        return Perm::WriteOnly;
    }
    else if (str == "rw")
    {
        return Perm::ReadWrite;
    }
    else 
    {
        return Perm::Unknown;
    }

}

std::string IndiProperty::permission2String(const Perm & perm)
{
    switch (perm)
    {
        case Perm::Unknown:
            return std::string("");
        case Perm::ReadOnly:
            return std::string("ro");
        case Perm::WriteOnly:
            return std::string("wo");
        case Perm::ReadWrite:
            return std::string("rw");
        default:
            return std::string("");
    }
}

std::string IndiProperty::type2String(const Type &type)
{
    switch (type)
    {
        case Type::Unknown:
            return std::string("");
        case Type::BLOB:
            return std::string("BLOB");
        case Type::Light:
            return std::string ("Light");
        case Type::Number:
            return std::string ("Number");
        case Type::Switch:
            return std::string("Switch");
        case Type::Text:
            return std::string("Text");
        default:
            return std::string("");
    }
}

IndiProperty::Type IndiProperty::string2Type( const std::string & str )
{
    if(str == "BLOB")
    {
        return Type::BLOB;
    }
    else if(str == "Light")
    {
        return Type::Light;
    }
    else if(str == "Number")
    {
        return Type::Number;
    }
    else if(str == "Switch")
    {
        return Type::Switch;
    }
    else if(str == "Text")
    {
        return Type::Text;
    }
    else 
    {
        return Type::Unknown;
    }
}

} //namespace pcf
