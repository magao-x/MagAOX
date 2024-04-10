
#include "pwfsAlignment.hpp"

#define errPrint(expl) std::cerr << "pwfsAlignment: " << __FILE__ << " " << __LINE__ << " " << expl << std::endl;

pwfsAlignment::pwfsAlignment() : rtimvOverlayInterface()
{
}

pwfsAlignment::~pwfsAlignment()
{
}

int pwfsAlignment::attachOverlay( rtimvOverlayAccess & roa,
                                  mx::app::appConfigurator & config
                                )
{
   
    config.configUnused(m_deviceName, mx::app::iniFile::makeKey("pwfsAlignment", "name"));
   
    if(m_deviceName == "") return 1; //Tell rtimv we can't be used.
   
    m_roa = roa;
   
    m_enabled = false;
   
   
    if(m_roa.m_dictionary != nullptr)
    {
        //Register these
        (*m_roa.m_dictionary)[m_deviceName + ".numPupils"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".sm_frameSize"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".quadrant1"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".quadrant2"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".quadrant3"].setBlob(nullptr, 0);
        (*m_roa.m_dictionary)[m_deviceName + ".quadrant4"].setBlob(nullptr, 0);
        //(*m_roa.m_dictionary)[m_deviceName + "."].setBlob(nullptr, 0);

    }

    if(m_enabled) enableOverlay();
    else disableOverlay();

    return 0;
}
      
int pwfsAlignment::updateOverlay()
{
   if(!m_enabled) return 0;
   
   if(m_roa.m_graphicsView == nullptr) return 0;
   
   static int initialized = false;

   if( m_roa.m_dictionary->count(m_deviceName + ".numPupils.value") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".numPupils.value"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_numPupils = atoi(m_blob);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".sm_frameSize.width") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".sm_frameSize.width"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_width = atoi(m_blob);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".sm_frameSize.height") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".sm_frameSize.height"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_height = atoi(m_blob);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant1.x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant1.x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_1x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant1.y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant1.y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_1y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant1.D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant1.D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_1D = strtod(m_blob,0);
   }
   //std::cerr << "1: " << m_1x << " " << m_1y << " " << m_1D << "\n";
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant1.set-x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant1.set-x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set1x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant1.set-y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant1.set-y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set1y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant1.set-D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant1.set-D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set1D = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant2.x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant2.x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_2x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant2.y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant2.y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_2y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant2.D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant2.D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_2D = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant2.set-x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant2.set-x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set2x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant2.set-y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant2.set-y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set2y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant2.set-D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant2.set-D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set2D = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant3.x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant3.x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_3x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant3.y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant3.y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_3y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant3.D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant3.D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_3D = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant3.set-x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant3.set-x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set3x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant3.set-y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant3.set-y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set3y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant3.set-D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant3.set-D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set3D = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant4.x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant4.x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_4x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant4.y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant4.y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_4y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant4.D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant4.D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_4D = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant4.set-x") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant4.set-x"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set4x = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant4.set-y") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant4.set-y"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set4y = strtod(m_blob,0);
   }
   
   if( m_roa.m_dictionary->count(m_deviceName + ".quadrant4.set-D") > 0)
   {
      if( ( (*m_roa.m_dictionary)[m_deviceName + ".quadrant4.set-D"].getBlobStr(m_blob, sizeof(m_blob))) == sizeof(m_blob) ) return -1; //Don't trust this as a string.
      m_set4D = strtod(m_blob,0);
   }
   
   if(!initialized)
   {
      if(m_numPupils == 0) return 0; //This means we haven't received this property yet.
      if(m_width == 0) return 0; //This means we haven't received this property yet.
      if(m_height == 0) return 0; //This means we haven't received this property yet.
      
      float dQw = 0.5*(0.5*m_width - 1.0);
      
      float dQh = 0.5*(0.5*m_height - 1.0);
      
      if(m_numPupils == 3)
      {
         m_1to2 = m_roa.m_graphicsView->scene()->addLine(QLineF(dQw,(m_height-1)-dQh, 0.5*m_width+dQw, (m_height-1)-dQh ), QColor("red"));
         m_1to3 = m_roa.m_graphicsView->scene()->addLine(QLineF(dQw,(m_height-1)-dQh, 0.5*m_height, dQh ), QColor("red"));
         m_3to4 = m_roa.m_graphicsView->scene()->addLine(QLineF(0.5*m_height, dQh, 0.5*m_width+dQw, (m_height-1)-dQh), QColor("red"));
         
         m_1to2s = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-29.5, 89.5, 119-29.5 ), QPen(QBrush(QColor("lime")), 0.1));
         m_1to3s = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-29.5, 29.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
         m_3to4s = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-89.5, 89.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
         
         m_c1 = m_roa.m_graphicsView->scene()->addEllipse(dQw-14,(m_height-1)-dQh-14, 28.0, 28.0, QColor("red"));
         m_c2 = m_roa.m_graphicsView->scene()->addEllipse(0.5*m_width+dQw-14, (m_height-1)-dQh-14, 28.0, 28.0, QColor("red"));
         m_c3 = m_roa.m_graphicsView->scene()->addEllipse(0.5*m_height-14, dQh-14, 28.0, 28.0, QColor("red"));

         m_c1s = m_roa.m_graphicsView->scene()->addEllipse(29.5-28.0, 119-29.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
         m_c2s = m_roa.m_graphicsView->scene()->addEllipse(89.5-28.0, 119-29.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
         m_c3s = m_roa.m_graphicsView->scene()->addEllipse(29.5-28.0, 119-89.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
      }
      else
      {
         m_1to2 = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-29.5, 89.5, 119-29.5 ), QColor("red"));
         m_1to3 = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-29.5, 29.5, 119-89.5 ), QColor("red"));
         m_3to4 = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-89.5, 89.5, 119-89.5 ), QColor("red"));
         m_2to4 = m_roa.m_graphicsView->scene()->addLine(QLineF(89.5,119-29.5, 89.5, 119-89.5 ), QColor("red"));
         
         m_1to2s = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-29.5, 89.5, 119-29.5 ), QPen(QBrush(QColor("lime")), 0.1));
         m_1to3s = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-29.5, 29.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
         m_3to4s = m_roa.m_graphicsView->scene()->addLine(QLineF(29.5,119-89.5, 89.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
         m_2to4s = m_roa.m_graphicsView->scene()->addLine(QLineF(89.5,119-29.5, 89.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
         
         m_c1 = m_roa.m_graphicsView->scene()->addEllipse(29.5-28.0, 119-29.5-28.0, 56.0, 56.0, QColor("red"));
         m_c2 = m_roa.m_graphicsView->scene()->addEllipse(89.5-28.0, 119-29.5-28.0, 56.0, 56.0, QColor("red"));
         m_c3 = m_roa.m_graphicsView->scene()->addEllipse(29.5-28.0, 119-89.5-28.0, 56.0, 56.0, QColor("red"));
         m_c4 = m_roa.m_graphicsView->scene()->addEllipse(89.5-28.0, 119-89.5-28.0, 56.0, 56.0, QColor("red"));
         
         m_c1s = m_roa.m_graphicsView->scene()->addEllipse(29.5-28.0, 119-29.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
         m_c2s = m_roa.m_graphicsView->scene()->addEllipse(89.5-28.0, 119-29.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
         m_c3s = m_roa.m_graphicsView->scene()->addEllipse(29.5-28.0, 119-89.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
         m_c4s = m_roa.m_graphicsView->scene()->addEllipse(89.5-28.0, 119-89.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
      }
      
      initialized = true;
   }
   
   if(m_numPupils == 3)
   {
      m_1to2->setLine(QLineF(m_1x,m_height-1-m_1y, m_2x, m_height-1-m_2y ));
      m_1to3->setLine(QLineF(m_1x,m_height-1-m_1y, m_3x, m_height-1-m_3y ));
      m_3to4->setLine(QLineF(m_3x,m_height-1-m_3y, m_2x, m_height-1-m_2y ));
      
      m_1to2s->setLine(QLineF(m_set1x,m_height-1-m_set1y, m_set2x, m_height-1-m_set2y ));
      m_1to3s->setLine(QLineF(m_set1x,m_height-1-m_set1y, m_set3x, m_height-1-m_set3y ));
      m_3to4s->setLine(QLineF(m_set3x,m_height-1-m_set3y, m_set2x, m_height-1-m_set2y ));
      
      m_c1->setRect(m_1x-0.5*m_1D, m_height-1-m_1y-0.5*m_1D, m_1D, m_1D);
      m_c2->setRect(m_2x-0.5*m_2D, m_height-1-m_2y-0.5*m_2D, m_2D, m_2D);
      m_c3->setRect(m_3x-0.5*m_3D, m_height-1-m_3y-0.5*m_3D, m_3D, m_3D);
      
      m_c1s->setRect(m_set1x-0.5*m_set1D, m_height-1-m_set1y-0.5*m_set1D, m_set1D, m_set1D);
      m_c2s->setRect(m_set2x-0.5*m_set2D, m_height-1-m_set2y-0.5*m_set2D, m_set2D, m_set2D);
      m_c3s->setRect(m_set3x-0.5*m_set3D, m_height-1-m_set3y-0.5*m_set3D, m_set3D, m_set3D);
   }
   else
   {
      m_1to2->setLine(QLineF(m_1x,m_height-1-m_1y, m_2x, m_height-1-m_2y ));
      m_1to3->setLine(QLineF(m_1x,m_height-1-m_1y, m_3x, m_height-1-m_3y ));
      m_2to4->setLine(QLineF(m_2x,m_height-1-m_2y, m_4x, m_height-1-m_4y ));
      m_3to4->setLine(QLineF(m_3x,m_height-1-m_3y, m_4x, m_height-1-m_4y ));
   
      m_1to2s->setLine(QLineF(m_set1x,m_height-1-m_set1y, m_set2x, m_height-1-m_set2y ));
      m_1to3s->setLine(QLineF(m_set1x,m_height-1-m_set1y, m_set3x, m_height-1-m_set3y ));
      m_2to4s->setLine(QLineF(m_set2x,m_height-1-m_set2y, m_set4x, m_height-1-m_set4y ));
      m_3to4s->setLine(QLineF(m_set3x,m_height-1-m_set3y, m_set4x, m_height-1-m_set4y ));
      
      m_c1->setRect(m_1x-0.5*m_1D, m_height-1-m_1y-0.5*m_1D, m_1D, m_1D);
      m_c2->setRect(m_2x-0.5*m_2D, m_height-1-m_2y-0.5*m_2D, m_2D, m_2D);
      m_c3->setRect(m_3x-0.5*m_3D, m_height-1-m_3y-0.5*m_3D, m_3D, m_3D);
      m_c4->setRect(m_4x-0.5*m_4D, m_height-1-m_4y-0.5*m_4D, m_4D, m_4D);
      
      m_c1s->setRect(m_set1x-0.5*m_set1D, m_height-1-m_set1y-0.5*m_set1D, m_set1D, m_set1D);
      m_c2s->setRect(m_set2x-0.5*m_set2D, m_height-1-m_set2y-0.5*m_set2D, m_set2D, m_set2D);
      m_c3s->setRect(m_set3x-0.5*m_set3D, m_height-1-m_set3y-0.5*m_set3D, m_set3D, m_set3D);
      m_c4s->setRect(m_set4x-0.5*m_set4D, m_height-1-m_set4y-0.5*m_set4D, m_set4D, m_set4D);
   }
   
   return 0;
}

void pwfsAlignment::keyPressEvent( QKeyEvent * ke)
{
   char key = ke->text()[0].toLatin1();
   
   if(key == 'P')
   {
      if(m_enabled) disableOverlay();
      else enableOverlay();
   }
}

bool pwfsAlignment::overlayEnabled()
{
   return m_enabled;
}

void pwfsAlignment::enableOverlay()
{
   if(m_1to2) m_1to2->setVisible(true); 
   if(m_1to3) m_1to3->setVisible(true); 
   if(m_3to4) m_3to4->setVisible(true);
   if(m_2to4) m_2to4->setVisible(true); 
   
   if(m_1to2s) m_1to2s->setVisible(true);
   if(m_1to3s) m_1to3s->setVisible(true);
   if(m_3to4s) m_3to4s->setVisible(true);
   if(m_2to4s) m_2to4s->setVisible(true);
   
   if(m_c1) m_c1->setVisible(true);
   if(m_c2) m_c2->setVisible(true);
   if(m_c3) m_c3->setVisible(true);
   if(m_c4) m_c4->setVisible(true);
   
   if(m_c1s) m_c1s->setVisible(true);
   if(m_c2s) m_c2s->setVisible(true);
   if(m_c3s) m_c3s->setVisible(true);
   if(m_c4s) m_c4s->setVisible(true);
 
   m_enabled = true;
}

void pwfsAlignment::disableOverlay()
{
   if(m_1to2) m_1to2->setVisible(false); 
   if(m_1to3) m_1to3->setVisible(false); 
   if(m_3to4) m_3to4->setVisible(false);
   if(m_2to4) m_2to4->setVisible(false); 
   
   if(m_1to2s) m_1to2s->setVisible(false);
   if(m_1to3s) m_1to3s->setVisible(false);
   if(m_3to4s) m_3to4s->setVisible(false);
   if(m_2to4s) m_2to4s->setVisible(false);
   
   if(m_c1) m_c1->setVisible(false);
   if(m_c2) m_c2->setVisible(false);
   if(m_c3) m_c3->setVisible(false);
   if(m_c4) m_c4->setVisible(false);
   
   if(m_c1s) m_c1s->setVisible(false);
   if(m_c2s) m_c2s->setVisible(false);
   if(m_c3s) m_c3s->setVisible(false);
   if(m_c4s) m_c4s->setVisible(false);
   
   m_enabled = false;
}

std::vector<std::string> pwfsAlignment::info()
{
    std::vector<std::string> vinfo;
    vinfo.push_back("PWFS Alignment overlay");
    
    return vinfo;
}