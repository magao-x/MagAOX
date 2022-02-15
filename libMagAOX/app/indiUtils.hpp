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
  * \todo investigate how this handles floating point values and string conversions.
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
      T oldVal = p[el].get<T>();

      pcf::IndiProperty::PropertyStateType oldState = p.getState();
   
      if(oldVal != newVal || oldState != newState)
      {
         p[el].set(newVal);
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
  * \todo investigate how this handles floating point values and string conversions.
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
         T oldVal = p[els[n]].get<T>();

         if(oldVal != newVals[n]) changed = true;
      }
      
      //and if there are changes, we send an update
      if(changed)
      {
         for(n=0; n< els.size(); ++n)
         {
            p[els[n]].set(newVals[n]);
         }
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
      indiDriver->sendSetProperty (p);
   }
   
   }
   catch(...)
   {
      std::cerr << "INDI Exception at " << __FILE__ << " " << __LINE__ << "\n";
      std::cerr << "from " << p.getName() << "." << el << "\n";
   }
}


} //namespace indi
} //namespace app
} //namespace MagAOX

#endif //app_magAOXIndiDriver_hpp
