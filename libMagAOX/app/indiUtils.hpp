/** \file indiUtils.hpp
  * \brief MagAO-X INDI Utilities
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * History:
  * - 2019-01-03 created by JRM
  * 
  * \ingroup app_files
  */

#ifndef app_indiUtils_hpp
#define app_indiUtils_hpp

#include <iostream>
#include <limits>

#include "../../INDI/libcommon/IndiProperty.hpp"
#include "../../INDI/libcommon/IndiElement.hpp"


namespace MagAOX
{
namespace app
{
namespace indi 
{

#define INDI_IDLE (pcf::IndiProperty::Idle)   
#define INDI_OK (pcf::IndiProperty::Ok)
#define INDI_BUSY (pcf::IndiProperty::Busy)
#define INDI_ALERT (pcf::IndiProperty::Alert)
  

/// Add a standard INDI Text element
/**
  * \returns 0 on success 
  * \returns -1 on error
  */ 
inline
int addTextElement( pcf::IndiProperty & prop, ///< [out] the property to which to add the elemtn
                    const std::string & name,  ///< [in] the name of the element
                    const std::string & label = "" ///< [in] [optional] the GUI label suggestion for this property
                  )                                                
{
   prop.add(pcf::IndiElement(name, 0));
      
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop[name].setLabel(label);
   }
   
   return 0;
}


/// Add a standard INDI Number element
/**
  * \returns 0 on success 
  * \returns -1 on error
  */ 
template<typename T>
int addNumberElement( pcf::IndiProperty & prop, ///< [out] the property to which to add the elemtn
                      const std::string & name,  ///< [in] the name of the element
                      const T & min, ///< [in] the minimum value for the element
                      const T & max, ///< [in] the minimum value for the element
                      const T & step, ///< [in] the step size of the lement
                      const std::string & format,  ///< [in] the _ value for the elements, applied to both target and current.  Set to "" to use the MagAO-X standard for type.
                      const std::string & label = "" ///< [in] [optional] the GUI label suggestion for this property
                    )                                                
{
   prop.add(pcf::IndiElement(name, 0));
   prop[name].setMin(min);
   prop[name].setMax(max);
   prop[name].setStep(step);
   prop[name].setFormat(format);
      
   //Don't set "" just in case libcommon does something with defaults
   if(label != "")
   {
      prop[name].setLabel(label);
   }
   
   return 0;
}

/// Update the value of the INDI element, but only if it has changed.
/** Only sends the set property message if the new value is different.
  * For properties with more than one element that may have changed, you should use the vector version below.
  * 
  * \todo this needs a const char specialization to std::string
  * 
  */  
template<typename T, class indiDriverT>
void updateIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                      const std::string & el,  ///< [in] The element name
                      const T & newVal,        ///< [in] the new value
                      indiDriverT * indiDriver, ///< [in] the MagAOX INDI driver to use
                      pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok
                    )
{
   if( !indiDriver ) return;
   
   try
   {
      //This is same code in IndiElement
      std::stringstream ssValue;
      ssValue.precision( 15 );
      ssValue << std::boolalpha << newVal;
      
      pcf::IndiProperty::PropertyStateType oldState = p.getState();
   
      //Do comparison in string space, not raw value
      if(p[el].getValue() != ssValue.str()|| oldState != newState)
      {
         p[el].set(newVal);
         p.setTimeStamp(pcf::TimeStamp());
         p.setState (newState);
         indiDriver->sendSetProperty (p);
      }
   }
   catch(std::exception & e)
   {
      std::cerr << "Exception caught at " << __FILE__ << " " << __LINE__ << " ";
      std::cerr << "from " << p.getName() << "." << el << ": ";
      std::cerr << e.what() << "\n";
   }
   catch(...)
   {
      std::cerr << "Exception caught at " << __FILE__ << " " << __LINE__ << " ";
      std::cerr << "from " << p.getName() << "." << el << "\n";
   }
   
}

/// Update the elements of an INDI propery, but only if there has been a change in at least one.
/** Only sends the set property message if at least one of the new values is different, or if the state has changed.
  *
  * \todo this needs a const char specialization to std::string
  * 
  */  
template<typename T, class indiDriverT>
void updateIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                      const std::vector<std::string> & els,  ///< [in] The element names
                      const std::vector<T> & newVals,        ///< [in] the new values
                      indiDriverT * indiDriver, ///< [in] the MagAOX INDI driver to use
                      pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok
                    )
{
   if( !indiDriver ) return;
   
   size_t n; //loop index outside so we can use it for error reporting.

   try
   {
      //First we look for any changes
      bool changed = false;
      pcf::IndiProperty::PropertyStateType oldState = p.getState();
      
      if(oldState != newState) changed = true;
      
      for(n=0; n< els.size() && changed != true; ++n)
      {
         //This is same code in IndiElement
         std::stringstream ssValue;
         ssValue.precision( 15 );
         ssValue << std::boolalpha << newVals[n];

         //compare in string space
         if(p[els[n]].getValue() != ssValue.str()) changed = true;
      }
      
      //and if there are changes, we send an update
      if(changed)
      {
         for(n=0; n< els.size(); ++n)
         {
            p[els[n]].set(newVals[n]);
         }
         p.setTimeStamp(pcf::TimeStamp());
         p.setState (newState);
         indiDriver->sendSetProperty (p);
      }
   }
   catch(std::exception & e)
   {
      std::cerr << "Exception caught at " << __FILE__ << " " << __LINE__ << " ";
      std::cerr << "from " << p.getName() << "." << els[n] << ": ";
      std::cerr << e.what() << "\n";
   }
   catch(...)
   {
      std::cerr << "Exception caught at " << __FILE__ << " " << __LINE__ << " ";
      std::cerr << "from " << p.getName() << "." << els[n] << "\n";
   }   
}

/// Update the value of the INDI element, but only if it has changed.
/** Only sends the set property message if the new value is different.
  *
  * \todo investigate how this handles floating point values and string conversions.
  * \todo this needs a const char specialization to std::string
  * 
  */  
template<class indiDriverT>
void updateSwitchIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                            const std::string & el,  ///< [in] The element name
                            const pcf::IndiElement::SwitchStateType & newVal,        ///< [in] the new value
                            indiDriverT * indiDriver, ///< [in] the MagAOX INDI driver to use
                            pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok
                          )
{
   if( !indiDriver ) return;

   try
   {
      pcf::IndiElement::SwitchStateType oldVal = p[el].getSwitchState();

      pcf::IndiProperty::PropertyStateType oldState = p.getState();
   
      if(oldVal != newVal || oldState != newState)
      {
         p[el].setSwitchState(newVal);
         p.setTimeStamp(pcf::TimeStamp());
         p.setState (newState);
         indiDriver->sendSetProperty (p);
      }
   }
   catch(...)
   {
      std::cerr << "INDI Exception at " << __FILE__ << " " << __LINE__ << "\n";
      std::cerr << "from " << p.getName() << "." << el << "\n";
   }
}


/// Update the values of a one-of-many INDI switch vector, but only if it has changed.
/** Only sends the set property message if the new settings are different.
  *
  * 
  */  
template<class indiDriverT>
void updateSelectionSwitchIfChanged( pcf::IndiProperty & p,   ///< [in/out] The property containing the element to possibly update
                                     const std::string & el,  ///< [in] The element name which is now on
                                     indiDriverT * indiDriver, ///< [in] the MagAOX INDI driver to use
                                     pcf::IndiProperty::PropertyStateType newState = pcf::IndiProperty::Ok
                                   )
{
   if( !indiDriver ) return;

   if(!p.find(el)) 
   {
      std::cerr << "INDI error at " << __FILE__ << " " << __LINE__ << "\n";
      std::cerr << p.getName() << " does not have " << el << "\n";
      return;
   }
   
   try
   {
      
   bool changed = false;
   for(auto elit = p.getElements().begin(); elit != p.getElements().end(); ++elit)
   {
      if( elit->first == el )
      {
         if(elit->second.getSwitchState() != pcf::IndiElement::On)
         {
            p[elit->first].setSwitchState(pcf::IndiElement::On);
            changed = true;
         }
      }
      else
      {
         if(elit->second.getSwitchState() != pcf::IndiElement::Off)
         {
            p[elit->first].setSwitchState(pcf::IndiElement::Off);
            changed = true;
         }
      }   
   }  

   pcf::IndiProperty::PropertyStateType oldState = p.getState();
   
   if(changed || oldState != newState)
   {
      p.setState (newState);
      p.setTimeStamp(pcf::TimeStamp());
      indiDriver->sendSetProperty (p);
   }
   
   }
   catch(...)
   {
      std::cerr << "INDI Exception at " << __FILE__ << " " << __LINE__ << "\n";
      std::cerr << "from " << p.getName() << "." << el << "\n";
   }
}

/// Parse an INDI key into the device and property names
/** We often represent an INDI property as a unique key in the form
  * `deviceName.propName`.  This function parses such a key into its
  * parts.
  *
  * \returns 0  on success
  * \returns -1 if the provided key is not at least 3 characters long
  * \returns -2 if no '.' is found
  * \returns -3 if '.' is the first character
  * \returns -4 if '.' is the last character
  */
inline
int parseIndiKey( std::string & devName,  ///< [out] the device name
                  std::string & propName, ///< [out] the property name
                  const std::string & key ///< [in] the key to parse
                )
{
    if(key.size() < 3)
    {
        return -1;
    }

    size_t p = key.find('.');

    if(p == std::string::npos)
    {
        devName = "";
        propName = "";
        return -2;
    }

    if(p == 0)
    {
        devName = "";
        propName = "";
        return -3;
    }

    if(p == key.size()-1)
    {
        devName = "";
        propName = "";
        return -4;
    }

    devName = key.substr(0, p);
    propName = key.substr(p+1);

    return 0;
}

} //namespace indi
} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
