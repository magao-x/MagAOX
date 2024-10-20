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

   selectionSwStatus(  QWidget * Parent = 0,
                       Qt::WindowFlags f = Qt::WindowFlags()
                     );

   selectionSwStatus(  const std::string & device,
                       const std::string & property,
                       const std::string & element,
                       const std::string & label,
                       const std::string & units,
                       QWidget * Parent = 0,
                       Qt::WindowFlags f = Qt::WindowFlags()
                     );

   ~selectionSwStatus();

   void setup( const std::string & device,
                       const std::string & property,
                       const std::string & element,
                       const std::string & label,
                       const std::string & units
                     );

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);


};

selectionSwStatus::selectionSwStatus(  QWidget * Parent,
                                       Qt::WindowFlags f
                                     ) : statusDisplay(Parent, f)
{
}

selectionSwStatus::selectionSwStatus(  const std::string & device,
                                       const std::string & property,
                                       const std::string & element,
                                       const std::string & label,
                                       const std::string & units,
                                       QWidget * Parent,
                                       Qt::WindowFlags f
                                     ) : statusDisplay(Parent, f)
{
    setup(device, property, element, label, units);
}

selectionSwStatus::~selectionSwStatus()
{
}

void selectionSwStatus::setup(  const std::string & device,
                                       const std::string & property,
                                       const std::string & element,
                                       const std::string & label,
                                       const std::string & units
                                     )
{
    statusDisplay::setup(device, property, element, label, units);

    m_ctrlWidget = (xWidget *) (new selectionSw(device, property, this, Qt::Dialog));
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

            newName = it->second.getName();
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
