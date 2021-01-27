
#include "cameraStatus.hpp"

cameraStatus::cameraStatus() : rtimvOverlayInterface()
{
}

cameraStatus::~cameraStatus()
{
}

int cameraStatus::attachOverlay( rtimvOverlayAccess & roa,
                                 mx::app::appConfigurator & config
                               )
{   
   m_roa = roa;
   m_qgs = roa.m_graphicsView->scene();
   
   
   config.configUnused(m_deviceName, mx::app::iniFile::makeKey("camera", "name"));
   
   if(m_deviceName == "")
   {
      m_enableable = false;
      disableOverlay();
      return 0;
   }
   
   m_enableable = true;
   m_enabled = true;
   
   config.configUnused(m_filterDeviceName, mx::app::iniFile::makeKey("camera", "filterDevice"));
   
   
   if(m_enabled) enableOverlay();
   else disableOverlay();
   
   return 0;
}
      
int cameraStatus::updateOverlay()
{
   if(!m_enabled) return 0;
   
   if(m_roa.m_dictionary == nullptr) return 0;
   
   if(m_roa.m_graphicsView == nullptr) return 0;
   
   size_t n = 0;
   char * str;
   char tstr[128];
   std::string sstr;
   
   if( m_roa.m_dictionary->count(m_deviceName + ".temp_ccd.current") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_deviceName + ".temp_ccd.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "%0.1f C", strtod(str,0));
      m_roa.m_graphicsView->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
   //Get curr size
   if( m_roa.m_dictionary->count(m_deviceName + ".fg_frameSize.width") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_deviceName + ".fg_frameSize.width"].m_blob;
      m_width = atoi(str);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".fg_frameSize.height") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_deviceName + ".fg_frameSize.height"].m_blob;
      m_height = atoi(str);
      
   }
   //Get full ROI size
   if( m_roa.m_dictionary->count(m_deviceName + ".roi_full_region.x") > 0 && m_roa.m_dictionary->count(m_deviceName + ".roi_full_region.y") > 0 && 
         m_roa.m_dictionary->count(m_deviceName + ".roi_full_region.w") > 0 && m_roa.m_dictionary->count(m_deviceName + ".roi_full_region.h") > 0 )
   {
      m_fullROI_x = strtod((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_full_region.x"].m_blob,NULL);
      m_fullROI_y = strtod((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_full_region.y"].m_blob,NULL);
      m_fullROI_w = atoi((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_full_region.w"].m_blob);
      m_fullROI_h = atoi((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_full_region.h"].m_blob);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".roi_region_w.current") > 0 && m_roa.m_dictionary->count(m_deviceName + ".roi_region_h.current") > 0
        && m_roa.m_dictionary->count(m_deviceName + ".roi_region_x.current") > 0 && m_roa.m_dictionary->count(m_deviceName + ".roi_region_y.current") > 0   )
   {
      char * strw = (char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_w.current"].m_blob;
      char * strh = (char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_h.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "%s x %s", strw, strh );
      m_roa.m_graphicsView->statusTextText(n, tstr);
      ++n;
    
      int w = atoi(strw);
      int h = atoi(strh);
      float x = strtod((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_x.current"].m_blob, NULL);
      float y = strtod((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_y.current"].m_blob, NULL);
         
      if( m_roa.m_dictionary->count(m_deviceName + ".roi_region_w.target") > 0 && m_roa.m_dictionary->count(m_deviceName + ".roi_region_h.target") > 0
            && m_roa.m_dictionary->count(m_deviceName + ".roi_region_x.target") > 0 && m_roa.m_dictionary->count(m_deviceName + ".roi_region_y.target") > 0)
      {
         
         int wt = atoi((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_w.target"].m_blob);
         int ht = atoi((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_h.target"].m_blob);
         float xt = strtod((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_x.target"].m_blob, NULL);
         float yt = strtod((char *)(*m_roa.m_dictionary)[m_deviceName + ".roi_region_y.target"].m_blob, NULL);
         
         if(wt != w || ht != h || xt != x || yt != y)
         {
            if(!m_roiBox)
            {
               m_roiBox = new StretchBox(xt-0.5*(wt-1),yt-0.5*(ht-1),wt,ht);
               m_roiBox->setPenColor(Qt::magenta);
               m_roiBox->setPenWidth(1);
               m_roiBox->setVisible(true);
               m_roiBox->setStretchable(false);
               m_roiBox->setRemovable(false);
               //m_roa.m_userBoxes->insert(m_roiBox);
               emit newStretchBox(m_roiBox);
            }
            
            float xc = xt-x + 0.5*((float)m_width-1);
            
            float yc = m_height - ( yt-y + 0.5*((float)m_height-1));
            
            m_roiBox->setRect(m_roiBox->mapRectFromScene(xc-0.5*(wt-1.0),yc-0.5*(ht-1.0),wt,ht));
            m_roiBox->setVisible(true);
            
         }
         else
         {
            if(m_roiBox) m_roiBox->setVisible(false);
         }
      }
      
      if(w != m_fullROI_w || h != m_fullROI_h)
      {
         if(!m_roiFullBox)
         {
            m_roiFullBox = new StretchBox(0,0,16,16);
            m_roiFullBox->setPenColor(Qt::magenta);
            m_roiFullBox->setPenWidth(1);
            m_roiFullBox->setVisible(true);
            m_roiFullBox->setStretchable(false);
            m_roiFullBox->setRemovable(false);
            m_roa.m_userBoxes->insert(m_roiFullBox);
            emit newStretchBox(m_roiFullBox);
         }
         
         float xc = m_fullROI_x-x + 0.5*((float)m_width-1);
         
         float yc = m_height - ( m_fullROI_y-y + 0.5*((float)m_height-1));
         
         m_roiFullBox->setRect(m_roiFullBox->mapRectFromScene(xc-0.5*(m_fullROI_w-1.0),yc-0.5*(m_fullROI_h-1.0),m_fullROI_w,m_fullROI_h));
         m_roiFullBox->setVisible(true);
      }
      else
      {
         if(m_roiFullBox) m_roiFullBox->setVisible(false);
      }
      
   }
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
   if( m_roa.m_dictionary->count(m_deviceName + ".exptime.current") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_deviceName + ".exptime.current"].m_blob;
      double et = strtod(str,0);
      if(et >= 100)
      {
        snprintf(tstr, sizeof(tstr), "%0.1f s", et);
      }
      else if(et >= 10)
      {
         snprintf(tstr, sizeof(tstr), "%0.2f s", et);
      }
      else if(et >= 0.1)
      {
         snprintf(tstr, sizeof(tstr), "%0.3f s", et);
      }
      else
      {
         if(et >= 0.00009999)
         {
            snprintf(tstr, sizeof(tstr), "%0.2f ms", et*1000.0);
         }
         else
         {
            snprintf(tstr, sizeof(tstr), "%0.2f us", et*1000000.0);
         }
      }
      m_roa.m_graphicsView->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
   if( m_roa.m_dictionary->count(m_deviceName + ".fps.current") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_deviceName + ".fps.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "%0.1f FPS", strtod(str,0));
      m_roa.m_graphicsView->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
   if( m_roa.m_dictionary->count(m_deviceName + ".emgain.current") > 0)
   {
      str = (char *)(*m_roa.m_dictionary)[m_deviceName + ".emgain.current"].m_blob;
      snprintf(tstr, sizeof(tstr), "EMG: %d", atoi(str));
      m_roa.m_graphicsView->statusTextText(n, tstr);
      ++n;
   }
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;
   
   if( m_roa.m_dictionary->count(m_deviceName + ".shutter_status.status") > 0)
   {
      sstr = std::string((char *)(*m_roa.m_dictionary)[m_deviceName + ".shutter_status.status"].m_blob);
      
      if(sstr == "READY")
      {
         if( m_roa.m_dictionary->count(m_deviceName + ".shutter.toggle") > 0)
         {
            sstr = std::string((char *)(*m_roa.m_dictionary)[m_deviceName + ".shutter.toggle"].m_blob);
            if(sstr == "on") sstr = "SHUT";
            else sstr = "OPEN";
            m_roa.m_graphicsView->statusTextText(n, sstr.c_str());
            ++n;
         }
      }
      else
      {
         sstr = "sh " + sstr;
         m_roa.m_graphicsView->statusTextText(n, sstr.c_str());
         ++n;
      }
   }
         
   if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;   
   
   if(m_filterDeviceName != "")
   {
      if( m_roa.m_dictionary->count(m_filterDeviceName + ".fsm.state") > 0)
      {
         std::string fwstate = std::string((char *)(*m_roa.m_dictionary)[m_filterDeviceName + ".fsm.state"].m_blob);
         
         if(fwstate == "READY" || fwstate == "OPERATING")
         {
            dictionaryIteratorT start = m_roa.m_dictionary->lower_bound(m_filterDeviceName + ".filterName");
            dictionaryIteratorT end = m_roa.m_dictionary->upper_bound(m_filterDeviceName + ".filterNameZ");
            
            std::string fkey;
            while(start != end)
            {
               if(std::string((char*)start->second.m_blob) == "on")
               {
                  fkey = start->first;
                  break;
               }
               ++start;
            }
            
            size_t pp = fkey.rfind('.');
            if(pp != std::string::npos && pp < fkey.size()-1) //need to be able to add 1
            {
               m_roa.m_graphicsView->statusTextText(n, fkey.substr(pp+1).c_str());
            }
            else
            {
               m_roa.m_graphicsView->statusTextText(n, "filt unk");
            }
         }
         else
         {
            fwstate = "f/w " + fwstate;
            m_roa.m_graphicsView->statusTextText(n, fwstate.c_str());
            ++n;
         }
      }
      if(n > m_roa.m_graphicsView->statusTextNo()-1) return 0;   
   }
   
   return 0;
}

void cameraStatus::keyPressEvent( QKeyEvent * ke)
{
   char key = ke->text()[0].toLatin1();
   
   if(key == 'C')
   {
      if(m_enabled) disableOverlay();
      else enableOverlay();
   }
}

bool cameraStatus::overlayEnabled()
{
   return m_enabled;
}

void cameraStatus::enableOverlay()
{
   if(m_enableable == false) return;
   
   m_enabled = true;
}

void cameraStatus::disableOverlay()
{
   for(size_t n=0; n<m_roa.m_graphicsView->statusTextNo();++n)
   {
      m_roa.m_graphicsView->statusTextText(n, "");
   }
   
   m_enabled = false;
}
