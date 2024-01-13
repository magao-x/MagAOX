#ifndef selectionSwStatus_hpp
#define selectionSwStatus_hpp

#include <QWidget>

#include "ui_statusDisplay.h"


#include "../xWidgets/statusDisplay.hpp"
#include "../xWidgets/selectionSw.hpp"

namespace xqt 
{
   
class selectionSwStatus : public statusDisplay
{
   Q_OBJECT
   
protected:
   

public:
   selectionSwStatus(  const std::string & device,
                       const std::string & property,
                       const std::string & element,
                       const std::string & label,
                       const std::string & units,
                       QWidget * Parent = 0, 
                       Qt::WindowFlags f = Qt::WindowFlags()
                     );
   
   ~selectionSwStatus();
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   

};
   
selectionSwStatus::selectionSwStatus(  const std::string & device,
                                       const std::string & property,
                                       const std::string & element,
                                       const std::string & label,
                                       const std::string & units,
                                       QWidget * Parent, 
                                       Qt::WindowFlags f
                                     ) : statusDisplay(device, property, element, label, units, Parent, f)
{
   m_ctrlWidget = (xWidget *) (new selectionSw(device, property, this, Qt::Dialog));

}
   
selectionSwStatus::~selectionSwStatus()
{
}
  
void selectionSwStatus::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == m_property)
   {
      std::string newName;
      for(auto it = ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it)
      {
         if(it->second.getSwitchState() == pcf::IndiElement::On)
         {
            if(newName != "")
            {
               std::cerr << "More than one switch selected in " << ipRecv.getDevice() << "." << ipRecv.getName() << "\n";
            }
         
            newName = it->second.name();
            if(newName != m_value) m_valChanged = true;
            m_value = newName; 
         }
      }

      
   }

   updateGUI();
}


} //namespace xqt
   
#include "moc_selectionSwStatus.cpp"

#endif
