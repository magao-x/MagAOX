/** \file xWidget.hpp
  * \brief Virtual class combining QWidget and multiIndiSubscriber
  * \author Jared R. Males
  */

#ifndef widget_hpp
#define widget_hpp

#include <QWidget>

#include "../../lib/multiIndiSubscriber.hpp"

#define XW_FONT_SIZE (15)

namespace xqt 
{

/// A virtual class combining QWidget and multiIndiSubscriber.
/** This is the base class of most MagAO-X INDI connected GUIs.
  * The virtual functions and event handlers of QWidget and be implemented in base classes. 
  * The virtual functions of multiIndiSubscriber should be implemented to enable INDI communications.
  */ 
class xWidget : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT
   
public:
   xWidget( QWidget * Parent = 0, 
            Qt::WindowFlags f = 0
          ) : QWidget(Parent, f)
   {
   }
   
   ~xWidget() noexcept
   {
   }
};
   
} //namespace xqt

#endif
