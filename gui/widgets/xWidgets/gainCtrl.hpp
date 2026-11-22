/** \file gainCtrl.hpp
 * \brief INDI widget for a gain or multiplier with numeric and slider controls.
 */
#ifndef gainCtrl_hpp
#define gainCtrl_hpp

#include "ui_gainCtrl.h"

#include "xWidget.hpp"
#include "statusEntry.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

#define GAIN (0)
#define MULTCOEFF (1)
namespace xqt
{

/// Control an INDI gain or multiplier.
class gainCtrl : public xWidget
{
   Q_OBJECT

protected:
  std::string m_device;   ///< INDI device owning the controlled property.
  std::string m_property; ///< Name of the controlled INDI number property.
  std::string m_label;    ///< Display label, populated from INDI if initially empty.

  int m_modes{ 0 };      ///< Number of modes in this block, or -1 for a loop-wide control.
  int m_modesTotal{ 0 }; ///< Cumulative mode count through this block, or -1 for a loop-wide control.

  float m_current{ 0 }; ///< Current control value, including optimistic updates from local controls.
  float m_target{ 0 };  ///< Last received target value.

  bool m_valChanged{ false }; ///< Whether the reported current value has changed.
  bool m_newSent{ false };    ///< Whether the next property update follows a locally sent command.

  float m_scale{ 0.05 }; ///< Step used by the plus and minus buttons.

  int   m_ctrlType{ GAIN };    ///< Whether this widget is configured for a gain or multiplier.
  float m_maxVal{ 1.5 };       ///< Value at the top of the slider range.
  bool  m_enforceMax{ false }; ///< Whether button and slider commands are capped at the slider maximum.
public:
  /// Construct an unconfigured gain control.
  gainCtrl( QWidget        *Parent = 0 /**< [in] Optional Qt parent. */,
            Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */
  );

  /// Construct a control bound to an INDI gain property.
  gainCtrl( const std::string &device /**< [in] INDI device name. */,
            const std::string &property /**< [in] INDI property name. */,
            const std::string &label /**< [in] Display label, or empty to use the property label. */,
            int                modes /**< [in] Modes in the block, or -1 to hide the count. */,
            int                modesTotal /**< [in] Cumulative modes through the block, or -1 to hide the count. */,
            QWidget           *Parent = 0 /**< [in] Optional Qt parent. */,
            Qt::WindowFlags    f      = Qt::WindowFlags() /**< [in] Qt window flags. */
  );

  /// Destroy the gain control.
  ~gainCtrl();

  /// Bind the widget and numeric entry to an INDI gain property.
  void setup( const std::string &device /**< [in] INDI device name. */,
              const std::string &property /**< [in] INDI property name. */,
              const std::string &label /**< [in] Display label, or empty to use the property label. */,
              int                modes /**< [in] Modes in the block, or -1 to hide the count. */,
              int                modesTotal /**< [in] Cumulative modes through the block, or -1 to hide the count. */
  );

  /// Set the display and control steps for gain values.
  void makeGainCtrl();

  /// Set the display and control steps for multiplier values.
  void makeMultCoeffCtrl();

  /// Subscribe the slider and numeric entry to this INDI property.
  virtual void subscribe();

  /// Notify the numeric entry that the INDI connection opened.
  virtual void onConnect();

  /// Reset the slider and numeric entry when the INDI connection closes.
  virtual void onDisconnect();

  /// Initialize the controls from a property definition.
  virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Property definition. */ );

  /// Reset the controls when their property is deleted.
  virtual void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Deleted property. */ );

  /// Refresh the controls from an INDI property update.
  virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Updated property. */ );

  /// Refresh the slider position from the cached current value.
  virtual void updateGUI();

  /// Send a numeric target using the current control limits.
  void setGain( float g /**< [in] Requested gain or multiplier. */ );

  /// Return the text in the numeric entry, including an active edit.
  QString inputText() const;

public slots:

  /// Increase the value by one step.
  void on_button_plus_pressed();

  /// Cycle the adjustment step.
  void on_button_scale_pressed();

  /// Decrease the value by one step.
  void on_button_minus_pressed();

  /// Reset the gain to zero or the multiplier to one.
  void on_button_zero_pressed();

  /// Preview a slider position in the numeric entry.
  void on_slider_sliderMoved( int s /**< [in] New slider position. */ );

  /// Send the value selected with the slider.
  void on_slider_sliderReleased();

protected:
  Ui::gainCtrl ui; ///< Qt Designer controls for this widget.
};

gainCtrl::gainCtrl( QWidget * Parent,
                    Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
}

gainCtrl::gainCtrl( const std::string & device,
                    const std::string & property,
                    const std::string & label,
                    int modes,
                    int modesTotal,
                    QWidget * Parent,
                    Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);

   setup(device, property, label, modes, modesTotal);
}

gainCtrl::~gainCtrl()
{
}

void gainCtrl::setup( const std::string & device,
                      const std::string & property,
                      const std::string & label,
                      int modes,
                      int modesTotal
                    )
{
   m_device = device;
   m_property = property;
   m_label = label;
   m_modes = modes;
   m_modesTotal = modesTotal;

   ui.status->setup(m_device, m_property, statusEntry::FLOAT, "", "");
   ui.status->setStretch(0,0,1);


   setXwFont(ui.button_plus);
   setXwFont(ui.button_scale);
   setXwFont(ui.button_minus);
   setXwFont(ui.button_zero);
   setXwFont(ui.label);

   ui.label->setText(m_label.c_str());

   if(m_modes == -1 || m_modesTotal == -1) ui.label_nummodes->setText("");
   else
   {
      std::string nnn = std::to_string(m_modes);
      nnn += " / ";
      nnn += std::to_string(m_modesTotal);
      ui.label_nummodes->setText(nnn.c_str());
   }

   makeGainCtrl();

   onDisconnect();
}

void gainCtrl::makeGainCtrl()
{
   m_ctrlType = GAIN;
   ui.status->format("%0.2f");
   m_maxVal = 1.5;
   ui.button_zero->setText("0");
   m_enforceMax = false;

   m_scale = 0.05;
   char sstr[16];
   snprintf(sstr, sizeof(sstr), "%0.2f", m_scale);
   ui.button_scale->setText(sstr);
}

void gainCtrl::makeMultCoeffCtrl()
{
   m_ctrlType = MULTCOEFF;
   ui.status->format("%0.3f");
   m_maxVal = 1.0;
   ui.button_zero->setText("1");
   m_enforceMax = true;

   m_scale = 0.005;
   char sstr[16];
   snprintf(sstr, sizeof(sstr), "%0.3f", m_scale);
   ui.button_scale->setText(sstr);
}

void gainCtrl::subscribe()
{
    if(!m_parent) return;

    if(m_property != "")
    {
        m_parent->addSubscriberProperty(this, m_device, m_property);

    }

    m_parent->addSubscriber(ui.status);

   return;
}

void gainCtrl::onConnect()
{
   ui.status->onConnect();
}

void gainCtrl::onDisconnect()
{
   m_current = 0;
   m_target = 0;
   m_valChanged = false;
   m_newSent = false;
   ui.slider->setValue(0);
   ui.status->onDisconnect();
}

void gainCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void gainCtrl::handleDelProperty( const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getDevice() != m_device) return;

   if(ipRecv.getName() == m_property)
   {
      onDisconnect();
   }
}

void gainCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   if(ipRecv.getDevice() != m_device) return;

   if(ipRecv.getName() == m_property)
   {
      if(m_newSent)
      {
         m_newSent=false;
      }
      else
      {
         if(ipRecv.find("current"))
         {
            float current = ipRecv["current"].get<float>();
            if(current != m_current)
            {
               m_valChanged = true;
               m_current = current;
            }
         }

         if(ipRecv.find("target"))
         {
            m_target = ipRecv["target"].get<float>();
         }

         if(m_label == "")
         {
            m_label = ipRecv.getLabel();

            size_t ssp = m_label.find(" Gain");
            if(ssp == std::string::npos)
            {
                ssp = m_label.find(" Mult");
            }
            if(ssp == std::string::npos)
            {
                ssp = m_label.find(" Limit");
            }

            if(ssp == std::string::npos)
            {
                ui.label->setText(m_label.c_str());
            }
            else
            {
                ui.label->setText(m_label.substr(0,ssp).c_str());
            }
         }
      }
   }

   updateGUI();

   ui.status->handleSetProperty(ipRecv);
}

void gainCtrl::updateGUI()
{
   if(isEnabled())
   {
      int slv = m_current/m_maxVal * (1.0*ui.slider->maximum()) + 0.5;
      if(slv < 0 ) slv = 0;
      else if(slv > ui.slider->maximum()) slv = ui.slider->maximum();

      ui.slider->setValue(slv);
   }

} //updateGUI()

void gainCtrl::setGain( float g )
{
   if(g < 0 ) g = 0;
   if(m_enforceMax && g > m_maxVal) g = m_maxVal;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   ip.setDevice(m_device);
   ip.setName(m_property);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = g;

   sendNewProperty(ip);
   m_newSent = true;

   m_current = g; //do this so multiple pushes update fast

}

QString gainCtrl::inputText() const
{
    return ui.status->inputText();
}

void gainCtrl::on_button_plus_pressed()
{
   setGain(m_current + m_scale);
}

void gainCtrl::on_button_scale_pressed()
{
   if(m_ctrlType == GAIN)
   {
      if( (int) (m_scale * 100 + 0.5) <= 1 )
      {
         m_scale = 0.1;
      }
      else if( (int) (m_scale * 100 + 0.5) >= 10 )
      {
         m_scale = 0.05;
      }
      else
      {
         m_scale = 0.01;
      }

      char sstr[16];
      snprintf(sstr, sizeof(sstr), "%0.2f", m_scale);
      ui.button_scale->setText(sstr);
   }
   else if(m_ctrlType == MULTCOEFF)
   {
      if( (int) (m_scale * 1000 + 0.5) <= 1 )
      {
         m_scale = 0.01;
      }
      else if( (int) (m_scale * 1000 + 0.5) >= 10 )
      {
         m_scale = 0.005;
      }
      else
      {
         m_scale = 0.001;
      }

      char sstr[16];
      snprintf(sstr, sizeof(sstr), "%0.3f", m_scale);
      ui.button_scale->setText(sstr);
   }
}

void gainCtrl::on_button_minus_pressed()
{
   float ng = m_current - m_scale;
   if(ng < 0) ng = 0;
   setGain(ng);
}

void gainCtrl::on_button_zero_pressed()
{
   if(m_ctrlType == GAIN) setGain(0.0);
   if(m_ctrlType == MULTCOEFF) setGain(1.0);
}

void gainCtrl::on_slider_sliderMoved(int s)
{
   double epos = (1.0*s)/ui.slider->maximum() * m_maxVal;

   ui.status->setEditText(QString::number(epos));
}

void gainCtrl::on_slider_sliderReleased()
{
   ui.status->stopEditing();

   float newg = 1.0*ui.slider->value() / (1.0*ui.slider->maximum())*m_maxVal;
   setGain(newg);
}

} //namespace xqt

#include "moc_gainCtrl.cpp"

#endif
