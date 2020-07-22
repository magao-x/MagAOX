
#include "pwfsAlignment.hpp"

pwfsAlignment::pwfsAlignment() : rtimvOverlayInterface()
{
}

pwfsAlignment::~pwfsAlignment()
{
}

int pwfsAlignment::attachOverlay( rtimvGraphicsView* gv, 
                                  std::unordered_map<std::string, rtimvDictBlob> * dict,
                                  mx::app::appConfigurator & config
                                )
{
   std::cerr << "pwfsAlignment attached -- w config\n";
   
   QGraphicsScene * qgs = gv->scene();
   
   m_dict = dict;
   
   m_1to2 = qgs->addLine(QLineF(29.5,119-29.5, 89.5, 119-29.5 ), QColor("red"));
   m_1to3 = qgs->addLine(QLineF(29.5,119-29.5, 29.5, 119-89.5 ), QColor("red"));
   m_3to4 = qgs->addLine(QLineF(29.5,119-89.5, 89.5, 119-89.5 ), QColor("red"));
   m_2to4 = qgs->addLine(QLineF(89.5,119-29.5, 89.5, 119-89.5 ), QColor("red"));
   
   m_1to2s = qgs->addLine(QLineF(29.5,119-29.5, 89.5, 119-29.5 ), QPen(QBrush(QColor("lime")), 0.1));
   m_1to3s = qgs->addLine(QLineF(29.5,119-29.5, 29.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
   m_3to4s = qgs->addLine(QLineF(29.5,119-89.5, 89.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
   m_2to4s = qgs->addLine(QLineF(89.5,119-29.5, 89.5, 119-89.5 ), QPen(QBrush(QColor("lime")), 0.1));
   
   m_c1 = qgs->addEllipse(29.5-28.0, 119-29.5-28.0, 56.0, 56.0, QColor("red"));
   m_c2 = qgs->addEllipse(89.5-28.0, 119-29.5-28.0, 56.0, 56.0, QColor("red"));
   m_c3 = qgs->addEllipse(29.5-28.0, 119-89.5-28.0, 56.0, 56.0, QColor("red"));
   m_c4 = qgs->addEllipse(89.5-28.0, 119-89.5-28.0, 56.0, 56.0, QColor("red"));
   
   m_c1s = qgs->addEllipse(29.5-28.0, 119-29.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
   m_c2s = qgs->addEllipse(89.5-28.0, 119-29.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
   m_c3s = qgs->addEllipse(29.5-28.0, 119-89.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
   m_c4s = qgs->addEllipse(89.5-28.0, 119-89.5-28.0, 56.0, 56.0, QPen(QBrush(QColor("lime")), 0.1));
   
   if(m_enabled) enableOverlay();
   else disableOverlay();
   
   return 0;
}
      
int pwfsAlignment::updateOverlay()
{
   if(!m_enabled) return 0;
   
   if(m_dict == nullptr) return 0;
   
   char * str;
   if( m_dict->count("camwfs-fit.quadrant1.x") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant1.x"].m_blob;
      m_1x = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant1.y") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant1.y"].m_blob;
      m_1y = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant1.D") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant1.D"].m_blob;
      m_1D = strtod(str,0);
   }
   //std::cerr << "1: " << m_1x << " " << m_1y << " " << m_1D << "\n";
   
   
   if( m_dict->count("camwfs-fit.quadrant2.x") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant2.x"].m_blob;
      m_2x = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant2.y") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant2.y"].m_blob;
      m_2y = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant2.D") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant2.D"].m_blob;
      m_2D = strtod(str,0);
   }
   //std::cerr << "2: " <<  m_2x << " " << m_2y << " " << m_2D << "\n";
   
   if( m_dict->count("camwfs-fit.quadrant3.x") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant3.x"].m_blob;
      m_3x = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant3.y") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant3.y"].m_blob;
      m_3y = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant3.D") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant3.D"].m_blob;
      m_3D = strtod(str,0);
   }
   //std::cerr << "3: " <<  m_3x << " " << m_3y << " " << m_3D << "\n";
   
   
   if( m_dict->count("camwfs-fit.quadrant4.x") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant4.x"].m_blob;
      m_4x = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant4.y") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant4.y"].m_blob;
      m_4y = strtod(str,0);
   }
   
   if( m_dict->count("camwfs-fit.quadrant4.D") > 0)
   {
      str = (char *)(*m_dict)["camwfs-fit.quadrant4.D"].m_blob;
      m_4D = strtod(str,0);
   }
   //std::cerr << "4: " <<  m_4x << " " << m_4y << " " << m_4D << "\n\n";
   
   m_1to2->setLine(QLineF(m_1x,119-m_1y, m_2x, 119-m_2y ));
   m_1to3->setLine(QLineF(m_1x,119-m_1y, m_3x, 119-m_3y ));
   m_2to4->setLine(QLineF(m_2x,119-m_2y, m_4x, 119-m_4y ));
   m_3to4->setLine(QLineF(m_3x,119-m_3y, m_4x, 119-m_4y ));
   
   m_c1->setRect(m_1x-28.0, 119-m_1y-28.0, 56.0, 56.0);
   m_c2->setRect(m_2x-28.0, 119-m_2y-28.0, 56.0, 56.0);
   m_c3->setRect(m_3x-28.0, 119-m_3y-28.0, 56.0, 56.0);
   m_c4->setRect(m_4x-28.0, 119-m_4y-28.0, 56.0, 56.0);
   
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
   std::cerr << "pwfsAlignment enabled\n";
   
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
   std::cerr << "pwfsAlignment disabled\n";

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
