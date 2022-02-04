#ifndef statusLabel_hpp
#define statusLabel_hpp

#include <iostream>

#include <QLabel>
#include <QTimer>
#include <QStyle>

namespace xqt
{

/// Implements an updating status display from QLabel
/** This widget has three modes: status; status changed; and editing. They are managed according to:
  * - Status is the normal mode, when not being edited or not changed, with text color set by the style sheet.
  * - When the text is changed via setTextChanged the widget adopts the statusChanged style (border highlight) for the change timeout (default 1.5 sec).
  * - Once this widget gets focus, the style is changed to the default text color and enters editing mode.
  * - When in editing mode (has focus), text updates via setTextChanged() will only occur in the background without changing the displayed text.
  * - If the user presses ESC editing is canceled and the widget returns to status mode.
  * - If the user pauses editing for more than the edit timeout (default 10 sec), this returns to status mode with the current value
  * - If focus then returns, the last edited value is loaded
  * - After the stale timeout (default 60 sec), the edited value is cleared so that subsequent edits start with the current value
  * - onReturnPressed should normally be used to signal a new value to set, rather than editingFinished.
  * 
  * To trigger the statusChanged style, the member function setTextChanged() must be called instead of setText().  This will also prevent interrupting
  * current editing when the widget has focus.
  * 
  * Ref: https://doc.qt.io/qt-5/qlabel.html
  */ 
class statusLabel : public QLabel
{
   Q_OBJECT
   
public:

   enum valchanges{NOTCHANGED, CHANGED, CHANGED_TIMEOUT};

protected:

   QString m_currText; ///< The current text, set with setTextChanged.

   bool m_highlightChanges {true};

   int m_valChanged {NOTCHANGED}; ///< Whether or not the value has changed, can take the values in enum statusLabel::valchanges.

   /// The timer for restoring status style from the statusChanged style
   /** This is started for m_changeTimeout msecs after a setTextChanged() is called.
     * 
     * Ref: https://doc.qt.io/qt-5/qtimer.html
     */
   QTimer * m_changeTimer {nullptr};

   std::chrono::milliseconds m_changeTimeout {1500}; ///< The timeout for m_changeTimer, default is 1,500 msec.

public:

   /// Default constructor.
   statusLabel(QWidget *parent = nullptr);

   /// Set the highlight changes flag
   /** If set to false, the widget will not change to statusChanged.
     * The default is true.
     * 
     * This sets m_highlightChanges
     */
   void highlightChanges(bool hc);

   /// Get the value of the highlight changes flag
   /**
     * \returns the current value of m_highlightChanges
     */
   bool highlightChanges();

   /// Set the change timeout
   /** The change timeout (m_changeTimeout) is the duration for which the 
     * statusChanged CSS style is applied after a value update.
     */ 
   void changeTimeout( std::chrono::milliseconds & cto /**< [in] the new change timeout in msec */);

   /// Get the change timeout
   /** The change timeout (m_changeTimeout) is the duration for which the 
     * statusChanged CSS style is applied after a value update.
     */
   std::chrono::milliseconds changeTimeout();

   /// Adopt the statusChanged CSS style for the duration of m_changeTimeout
   void setTextChanged(const QString & text /**< [in] The new text */);

protected:

   /// The paintEvent causes changes in the style of this widget based on status/statusChanged/editing mode
   /**
     * \override
     */
   virtual void paintEvent(QPaintEvent * e);

protected slots:

   void changeTimerOut();

};

statusLabel::statusLabel( QWidget *parent ) : QLabel(parent)
{
   setProperty("isStatus", true);
   m_changeTimer = new QTimer(this);
   connect(m_changeTimer, SIGNAL(timeout()), this, SLOT(changeTimerOut()));

   QFont qf = font();
   qf.setPixelSize(XW_FONT_SIZE);
   setFont(qf);
}

void statusLabel::highlightChanges(bool hc)
{
   m_highlightChanges = hc;
}

bool statusLabel::highlightChanges()
{
   return m_highlightChanges;
}

void statusLabel::changeTimeout( std::chrono::milliseconds & cto)
{
   m_changeTimeout = cto;
}

std::chrono::milliseconds statusLabel::changeTimeout()
{
   return m_changeTimeout;
}


void statusLabel::setTextChanged(const QString & text)
{
   m_currText = text;
   setText(m_currText);

   m_valChanged = CHANGED;
}

void statusLabel::paintEvent(QPaintEvent * e)
{
   if(m_valChanged == CHANGED)
   {
      setProperty("isStatus", true);
      if(m_highlightChanges)
      {
         setFrameShape(QFrame::Box); 
         setProperty("isStatusChanged", true);
      }
      style()->unpolish(this);
      m_changeTimer->start(m_changeTimeout);
      
      m_valChanged = 0;
   }
   else if(m_valChanged == CHANGED_TIMEOUT)
   {
      m_changeTimer->stop();
      setFrameShape(QFrame::NoFrame);
      setProperty("isStatusChanged",false);
      style()->unpolish(this);
      
      m_valChanged = 0;
   }

   QLabel::paintEvent(e);
}

void statusLabel::changeTimerOut()
{
   m_valChanged = CHANGED_TIMEOUT;
   if(isEnabled()) update();   
}

} //namespace xqt

#include "moc_statusLabel.cpp"

#endif //statusLabel_hpp
