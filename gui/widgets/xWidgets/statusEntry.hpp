/** \file statusEntry.hpp
 * \brief Editable INDI status and target entry widget.
 */
#ifndef statusEntry_hpp
#define statusEntry_hpp

#include <QWidget>

#include <cmath>
#include "statusLineEdit.hpp"

#include "ui_statusEntry.h"

#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt
{

/// Display an INDI current value and accept edits to its target.
class statusEntry : public QWidget, public multiIndiSubscriber
{
   Q_OBJECT

public:
  /// Display formats supported by the entry.
  enum types
  {
      STRING,
      INT,
      FLOAT
  };

protected:
  std::string m_device;              ///< INDI device owning the property.
  std::string m_property;            ///< INDI property displayed and edited by this entry.
  std::string m_label;               ///< Label displayed beside the entry.
  std::string m_units;               ///< Optional units appended to the label.
  std::string m_currEl{ "current" }; ///< Name of the element supplying the displayed value.
  std::string m_targEl{ "target" };  ///< Name of the element receiving edits.

  bool m_readOnly{ false };        ///< Whether edits are disabled.
  bool m_highlightChanges{ true }; ///< Whether to highlight updates to the displayed value.

  std::string m_format; ///< Display format for numeric values.

  int m_type{ FLOAT }; ///< Display type used to interpret the current value.

  pcf::IndiProperty::Type m_propertyType{ pcf::IndiProperty::Unknown }; ///< Last reported INDI property type.

  bool m_valChanged{ false }; ///< Whether a new current value awaits a display refresh.

  std::string m_current; ///< Last received value of the current element.
  std::string m_target;  ///< Last received value of the target element.

  QTimer *m_updateTimer{ nullptr }; ///< Timer for refreshing the displayed value.

public:
  /// Construct an unconfigured entry.
  statusEntry( QWidget        *Parent = 0 /**< [in] Optional Qt parent. */,
               Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */
  );

  /// Construct an entry bound to an INDI property.
  statusEntry( const std::string &device /**< [in] INDI device name. */,
               const std::string &property /**< [in] INDI property name. */,
               int                type /**< [in] Display type from types. */,
               const std::string &label /**< [in] Entry label. */,
               const std::string &units /**< [in] Optional units for the label. */,
               QWidget           *Parent = 0 /**< [in] Optional Qt parent. */,
               Qt::WindowFlags    f      = Qt::WindowFlags() /**< [in] Qt window flags. */
  );

  /// Destroy the entry.
  ~statusEntry() noexcept;

  /// Initialize the designer controls and display refresh timer.
  void construct();

  /// Bind the entry to an INDI property and configure its display.
  void setup( const std::string &device /**< [in] INDI device name. */,
              const std::string &property /**< [in] INDI property name. */,
              int                type /**< [in] Display type from types. */,
              const std::string &label /**< [in] Entry label. */,
              const std::string &units /**< [in] Optional units for the label. */
  );

  /// Set the name of the element displayed by the entry.
  void currEl( const std::string &cel /**< [in] Current element name. */ )
  {
      m_currEl = cel;
  }

  /// Set the name of the element written by the entry.
  void targEl( const std::string &tel /**< [in] Target element name. */ )
  {
      m_targEl = tel;
   }

   /// Restore the default format for the configured display type.
   void defaultFormat();

   /// Format the last reported value for display.
   QString formattedValue();

   /// Return the text shown by the editor, including an active edit.
   QString inputText() const;

   /// Set the read only flag
   /** If set to true, the widget will not accept focus and will not enter edit mode.
     * The default is false.
     *
     * This sets m_readOnly
     */
   void readOnly( bool ro /**< [in] Whether to disable editing. */ );

   /// Get the value of the read only flag
   /**
     * \returns the current value of m_readOnly
     */
   bool readOnly();

   /// Choose whether to highlight a changed current value.
   void highlightChanges( bool hc /**< [in] Whether to highlight changes. */ );

   /// Return whether current-value changes are highlighted.
   bool highlightChanges();

   /// Subscribe to the configured INDI property.
   virtual void subscribe();

   /// Request a display refresh after connecting.
   virtual void onConnect();

   /// Clear the cached values and reset the display on disconnect.
   virtual void onDisconnect();

   /// Initialize the entry from an INDI property definition.
   void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Property definition. */ );

   /// Clear the display if the configured property is deleted.
   void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Deleted property. */ );

   /// Cache the current and target elements from an INDI update.
   void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Updated property. */ );

   /// Set the stretch of the horizontal layout
   void setStretch( int s0 /**< [in] Stretch of the spacer, or zero to remove it. */,
                    int s1 /**< [in] Stretch of the label. */,
                    int s2 /**< [in] Stretch of the value. */
   );

   /// Set the format string
   void format( const std::string &f /**< [in] Format for displaying the current value. */ );

 protected:
   /// Release keyboard focus from the value editor.
   virtual void clearFocus();

public slots:
  /// Display the current value if it has changed.
  void updateGUI();

  /// Send the editor contents as a target value.
  void on_value_returnPressed();

  /// Set the value editor text and start editing.
  void setEditText( const QString &qs /**< [in] Text to edit. */ );

  /// End editing and restore the current value display.
  void stopEditing();

signals:

  /// Request a refresh of the displayed value.
  void doUpdateGUI();

private:
  Ui::statusEntry ui; ///< Qt Designer controls for this widget.
};

statusEntry::statusEntry( QWidget * Parent,
                          Qt::WindowFlags f) : QWidget(Parent, f)
{
   construct();
}

statusEntry::statusEntry( const std::string & device,
                          const std::string & property,
                          int type,
                          const std::string & label,
                          const std::string & units,
                          QWidget * Parent,
                          Qt::WindowFlags f) : QWidget(Parent, f)
{
   construct();
   setup(device, property, type, label, units);
}

statusEntry::~statusEntry() noexcept
{
}

void statusEntry::construct()
{
    ui.setupUi(this);
    ui.value->setProperty("isStatus", true);

    QFont qf = ui.label->font();
    qf.setPixelSize(XW_FONT_SIZE);
    ui.label->setFont(qf);

    qf = ui.value->font();
    qf.setPixelSize(XW_FONT_SIZE);
    ui.value->setFont(qf);

    m_updateTimer = new QTimer;

    connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));

    m_updateTimer->start(250);

    connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));
}

void statusEntry::setup( const std::string & device,
                         const std::string & property,
                         int type,
                         const std::string & label,
                         const std::string & units
                       )
{
   m_device = device;
   m_property = property;

   m_type = type;
   defaultFormat();

   m_label = label;
   m_units = units;

   std::string lab = m_label;
   if(m_units != "") lab += " [" + m_units + "]";
   ui.label->setText(lab.c_str());

   onDisconnect();
}

void statusEntry::defaultFormat()
{
   switch(m_type)
   {
      case STRING:
         m_format = "";
         break;
      case INT:
         m_format = "%d";
         break;
      case FLOAT:
         m_format = "auto";
         break;
      default:
         m_format = "%0.2f";
         break;
   }
}

/// Format a floating-point value using precision based on its magnitude.
QString floatAutoFormat( const std::string &value /**< [in] Numeric text to format. */ )
{
   double v;
   try
   {
      v = std::stod(value);
   }
   catch(...)
   {
      return QString("");
   }

   std::string format = "%f";

   double av = fabs(v);
   if(av >= 10000)
   {
      format = "%0.3g";
   }
   else if(av >= 100)
   {
      format = "%0.1f";
   }
   else if(av >= 0.1)
   {
      format = "%0.2f";
   }
   else if(av >= 0.01)
   {
      format = "%0.3f";
   }
   else if(av >= 0.001)
   {
      format = "%0.4f";
   }
   else
   {
      format = "%0.3g";
   }

   char str[16];
   snprintf(str, sizeof(str), format.c_str(), v);
   return QString(str);
}

QString statusEntry::formattedValue()
{
   char str[64];
   switch(m_type)
   {
      case STRING:
         return QString(m_current.c_str());
      case INT:
         try
         {
            snprintf(str, sizeof(str), m_format.c_str(), std::stoi(m_current));
            return QString(str);
         }
         catch(...)
         {
            return QString("");
         }

      case FLOAT:
         if(m_format == "auto") return floatAutoFormat(m_current);

         try
         {
            snprintf(str, sizeof(str), m_format.c_str(), std::stof(m_current));
            return QString(str);
         }
         catch(...)
         {
            return QString("");
         }


      default:
         return QString(m_current.c_str());
   }
}

QString statusEntry::inputText() const
{
    return ui.value->text();
}

void statusEntry::readOnly(bool ro)
{
   m_readOnly = ro;
   ui.value->readOnly(ro);
}

bool statusEntry::readOnly()
{
   return ui.value->readOnly();
}

void statusEntry::highlightChanges(bool hc)
{
   m_highlightChanges = hc;
   ui.value->highlightChanges(hc);
}

bool statusEntry::highlightChanges()
{
   return ui.value->highlightChanges();
}

void statusEntry::subscribe()
{
    if(!m_parent) return ;

    if(m_property != "") m_parent->addSubscriberProperty(this, m_device, m_property);

    return;
}

void statusEntry::onConnect()
{
    m_valChanged = true;
}

void statusEntry::onDisconnect()
{
    m_propertyType = pcf::IndiProperty::Unknown;
    m_current.clear();
    m_target.clear();
    m_valChanged = false;

    ui.value->setText( "---" );
}

void statusEntry::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void statusEntry::handleDelProperty( const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getDevice() != m_device) return;

   if(ipRecv.getName() == m_property)
   {
      onDisconnect();
   }
}

void statusEntry::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
    {
        return;
    }

    if( ipRecv.getName() == m_property )
    {
        m_propertyType = ipRecv.getType();

        if( ipRecv.find( m_currEl ) )
        {
            const std::string current = ipRecv[m_currEl].get();

            if( current != m_current )
            {
                m_valChanged = true;
            }

            m_current = current;
        }

        if( ipRecv.find( m_targEl ) )
        {
            m_target = ipRecv[m_targEl].get();
        }
    }

    emit doUpdateGUI();
}

void statusEntry::setStretch( int s0,
                              int s1,
                              int s2
                            )
{
   //Have to do this b/c hlayout is not exposed for whatever reason.
   QHBoxLayout * qhbl = findChild<QHBoxLayout*>("hlayout");
   if(qhbl)
   {
      qhbl->setStretch(0,s0);
      qhbl->setStretch(1,s1);
      qhbl->setStretch(2,s2);
      if(s0 == 0) //remove the spacer
      {
         QLayoutItem * qli = qhbl->itemAt(0);
         if(qli) qhbl->removeItem(qli);
      }
   }
}

void statusEntry::format( const std::string & f)
{
   m_format = f;
}

void statusEntry::clearFocus()
{
   ui.value->clearFocus();
}

void statusEntry::on_value_returnPressed()
{
    const std::string value = ui.value->text().toStdString();
    ui.value->clearFocus();

    const auto type = m_propertyType == pcf::IndiProperty::Number ? pcf::IndiProperty::Number : pcf::IndiProperty::Text;
    pcf::IndiProperty ip( type, m_device, m_property );
    ip.add( pcf::IndiElement( m_targEl, value ) );

    sendNewProperty( ip );
}

void statusEntry::updateGUI()
{
    if(isEnabled())
    {
        if(m_valChanged)
        {
            ui.value->setTextChanged(formattedValue());
            m_valChanged = false;
        }
    }

} //updateGUI()

void statusEntry::setEditText(const QString & qs)
{
   ui.value->setEditText(qs);
}

void statusEntry::stopEditing()
{
   ui.value->stopEditing();
}



} //namespace xqt

#include "moc_statusEntry.cpp"

#endif
