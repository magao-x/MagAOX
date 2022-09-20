#include <mutex>

#include <QWidget>
#include <QPainter>
#include <qwt_dial_needle.h>

#include <xqwt_multi_dial.h>

#include "ui_pwr.h"

#include "../../lib/multiIndiSubscriber.hpp"

#include "../../widgets/pwr/pwrDevice.hpp"
#include "../../widgets/pwr/pwrChannel.hpp"


namespace xqt 
{
   
class pwrGUI : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::vector<xqt::pwrDevice *> m_devices; ///< The power devices detected.
   
   ///Mutex for locking INDI communications.
   std::mutex m_addMutex;
   
public:
   pwrGUI( QWidget * Parent = 0, Qt::WindowFlags f = Qt::WindowFlags());
   
   virtual ~pwrGUI() noexcept;
   
   /// Called by the parent once the parent is disconnected.
   /** If this is reimplemented, you should call pwrGUI::onDisconnect() to ensure children are notified.
     *
     */ 
   virtual void onDisconnect();
   
   /// Callback for a `defProperty` message notifying us that the propery has changed.
   /** This is called by the publisher which is subscribed to.
     * 
     */
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has been defined*/);
   
   /// Callback for a `delProperty` message notifying us that the propery has changed.
   /** This is called by the publisher which is subscribed to.
     * 
     * Derived classes shall implement this.
     */
   virtual void handleDelProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has been deleted*/);

   /// Callback for a SET PROPERTY message notifying us that the propery has changed.
   /** This is called by the publisher which is subscribed to.
     * 
     */
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   /// Populate the switch grid
   /** First erases the existing grid.
     * Then adds each device and its switches
     */  
   void populateGrid();

   /// Erase the switch grid 
   /** Removes all widgets from the grid, re-parenting them to nullptr.
     * This does not delete the widgets themselves.
     */
   void eraseGrid();
   
public slots:

   /// Add a device to the grid.
   /** Erases the grid, then sorts m_devices, and adds each device to the grid.
    */
   void addDevice( std::string * devName, ///< [in] The name of the device.  This pointer is deleted by this slot.
                   std::vector<std::string> * channels
                 );
   
   /// Remove a device from the gui.
   /** First erases the grid, then disconnects the devices signals and then unsubcribes from its properties.
     * Finally, re-populates the grid.
     */ 
   void removeDevice(std::string * devName /**< [in] The name of the device.  This pointer is deleted by this slot*/);

   /// A channel state change has been requested.
   /** Sends the newProperty request.
     */
   void chChange( pcf::IndiProperty & ip /**< [in] the INDI property to send in the newProperty*/);

   /// Update the gauges when values have changed
   void updateGauges();
   
   /// Causes a disconnect, which will then trigger a full reconnect.
   void on_buttonReconnect_pressed();

signals:

   /// Issued when a new device is detected in handleDefProperties
   void gotNewDevice( std::string * devName,              ///< [out] the device name
                      std::vector<std::string> * channels ///< [out] the devices channels
                    );
   
   /// Issued when a device delettion is detected in handleDelProperties
   void gotDeleteDevice( std::string * devName /**< [out] the device name*/);

private:
      
   Ui::pwr ui;
};

} //namespace xqt
   
