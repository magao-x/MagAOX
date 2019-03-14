
#ifndef xqt_pwrChannel_hpp
#define xqt_pwrChannel_hpp

#include <string>

#include <QWidget>
#include <QSlider>
#include <QTimer>

#include <qwt_text_label.h>

namespace xqt 
{

/// A single power channel control widget
/** Contains the text label and the slider bar control for a single power channel.
  * These widgets are themselves intended to be added to a grid layout -- the 
  * pwrChannel widget does not actually manage them.
  */
class pwrChannel : public QWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_channelName; ///< The name of this channel
    
   QwtTextLabel* m_channelNameLabel {nullptr}; ///< The widget to display the channel name
   
   QSlider * m_channelSwitch {nullptr}; ///< The widget providing user control
   
   int m_setSwitchState {0}; ///< The last state set by the user.
   
   std::vector<int> m_outlets; ///< The outlets controlled by this channel.
   
   double m_onDelay {0}; ///< The total turn-on delay for this channel (between outlets)
   
   double m_onTimeout {10000}; ///< The turn-ontimeout for this channel, the time to wait for device to update the status before re-enabling the switch.
   
   double m_offDelay {0}; ///> The turn-off delay for this channel (between outlets)
   
   double m_offTimeout {10000}; ///< The turn-off timeout for this channel, the time to wait for device to update the status before re-enabling the switch.
   
   QTimer * m_timer {nullptr}; ///< Timer for tracking timeouts on channel state changes
   
public:

   ///Constructor
   /** Constructs the m_channelNameLabel and m_channelSwitch widgets, sets the palette of m_channelSwitch, an connects
     * the m_channelSwitch sliderReleased signal to the sliderRelased slot.
     */ 
   pwrChannel( QWidget * parent = nullptr, 
               Qt::WindowFlags flags = 0
             );

   ///Destructor
   virtual ~pwrChannel();
   
   /// Get the channel name
   /**
     * \returns the current value of m_channelName
     */ 
   std::string channelName();
   
   /// Set the channel name
   /** Sets m_channelName.
     */ 
   void channelName( const std::string & nname /**< [in] the new channel name*/);
   
   int switchState();
   
   void switchState( int swstate);

   QwtTextLabel * channelNameLabel();
   
   QSlider * channelSwitch();
   
   void outlets( const std::vector<int> & outs );
   
   void onDelay( double onD );
   
   void offDelay (double offD);
   
   void calcOnTimeout();
   
   void calcOffTimeout();
   
public slots:
   
   void sliderReleased();
   
   void noTimeOut();
   
   void timeOut();
   
signals:
   
   void switchOn( const std::string & channelName );
   
   void switchOff( const std::string & channelName );
   
   void switchTargetReached();
   
};
   
}//namespace xqt
#endif //xqt_pwrChannel_hpp
