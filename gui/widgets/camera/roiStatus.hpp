#ifndef roiStatus_hpp
#define roiStatus_hpp

#include <QWidget>

#include "ui_statusDisplay.h"


#include "../xWidgets/statusDisplay.hpp"

namespace xqt 
{
   
class roiStatus : public statusDisplay
{
   Q_OBJECT
   
protected:
   
   int m_bin_x_curr {0};
   int m_bin_x_tgt {0};
   
   int m_bin_y_curr {0};
   int m_bin_y_tgt {0};

   float m_cen_x_curr {0};
   float m_cen_x_tgt {0};

   float m_cen_y_curr {0};
   float m_cen_y_tgt {0};

   int m_wid_curr {0};
   int m_wid_tgt {0};

   int m_hgt_curr {0};
   int m_hgt_tgt {0};

public:
   roiStatus( std::string & camName,
              QWidget * Parent = 0, 
              Qt::WindowFlags f = 0
            );
   
   ~roiStatus();
   
   virtual void subscribe();
                
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void updateGUI();

};
   
roiStatus::roiStatus( std::string & camName,
                      QWidget * Parent, 
                      Qt::WindowFlags f) : statusDisplay(camName, "", "", "ROI", "", Parent, f)
{
   m_ctrlWidget = (xWidget *) (new roi(camName, this, Qt::Dialog));
}
   
roiStatus::~roiStatus()
{
}

void roiStatus::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty(this, m_device, "roi_region_bin_x");
   m_parent->addSubscriberProperty(this, m_device, "roi_region_bin_y");
   m_parent->addSubscriberProperty(this, m_device, "roi_region_x");
   m_parent->addSubscriberProperty(this, m_device, "roi_region_y");
   m_parent->addSubscriberProperty(this, m_device, "roi_region_w");
   m_parent->addSubscriberProperty(this, m_device, "roi_region_h");

   statusDisplay::subscribe();

   return;
}
  
void roiStatus::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_device) return;
   
   if(ipRecv.getName() == "roi_region_bin_x")
   {
      if(ipRecv.find("current"))
      {
         int curr = ipRecv["current"].get<int>();
         if(curr != m_bin_x_curr) m_valChanged = true;
         m_bin_x_curr = curr;
      }

      if(ipRecv.find("target"))
      {
         m_bin_x_tgt = ipRecv["target"].get<int>();
      }
   }
   else if(ipRecv.getName() == "roi_region_bin_y")
   {
      if(ipRecv.find("current"))
      {
         int curr = ipRecv["current"].get<int>();
         if(curr != m_bin_y_curr) m_valChanged = true;
         m_bin_y_curr = curr;
      }

      if(ipRecv.find("target"))
      {
         m_bin_y_tgt = ipRecv["target"].get<int>();
      }
   }
   else if(ipRecv.getName() == "roi_region_x")
   {
      if(ipRecv.find("current"))
      {
         float curr = ipRecv["current"].get<float>();
         if(curr != m_cen_x_curr) m_valChanged = true;
         m_cen_x_curr = curr;
      }

      if(ipRecv.find("target"))
      {
         m_cen_x_tgt = ipRecv["target"].get<float>();
      }
   }
   else if(ipRecv.getName() == "roi_region_y")
   {
      if(ipRecv.find("current"))
      {
         float curr = ipRecv["current"].get<float>();
         if(curr != m_cen_y_curr) m_valChanged = true;
         m_cen_y_curr = curr;
      }

      if(ipRecv.find("target"))
      {
         m_cen_y_tgt = ipRecv["target"].get<float>();
      }
   }
   else if(ipRecv.getName() == "roi_region_w")
   {
      if(ipRecv.find("current"))
      {
         int curr = ipRecv["current"].get<int>();
         if(curr != m_wid_curr) m_valChanged = true;
         m_wid_curr = curr;
      }

      if(ipRecv.find("target"))
      {
         m_wid_tgt = ipRecv["target"].get<int>();
      }
   }
   else if(ipRecv.getName() == "roi_region_h")
   {
      if(ipRecv.find("current"))
      {
         int curr = ipRecv["current"].get<int>();
         if(curr != m_hgt_curr) m_valChanged = true;
         m_hgt_curr = curr;
      }

      if(ipRecv.find("target"))
      {
         m_hgt_tgt = ipRecv["target"].get<int>();
      }
   }

   updateGUI();
}

void roiStatus::updateGUI()
{
   if(isEnabled())
   {
      if(m_valChanged)
      {
         char stat[64];
         snprintf(stat, sizeof(stat), "%d x %d [%d x %d]", m_wid_curr, m_hgt_curr, m_bin_x_curr, m_bin_y_curr);
      
         ui.status->setTextChanged(stat);  
         m_valChanged = false;
 
      }
   }


} //updateGUI()

} //namespace xqt
   
#include "moc_roiStatus.cpp"

#endif
