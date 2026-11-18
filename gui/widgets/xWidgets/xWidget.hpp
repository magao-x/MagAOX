/** \file xWidget.hpp
 * \brief Virtual class combining QWidget and multiIndiSubscriber
 * \author Jared R. Males
 */

#ifndef widget_hpp
#define widget_hpp

#include <algorithm>

#include <QAbstractItemModel>
#include <QAbstractItemView>
#include <QComboBox>
#include <QEvent>
#include <QScrollBar>
#include <QVariant>
#include <QWidget>

#include "../../lib/multiIndiSubscriber.hpp"

#define XW_FONT_SIZE ( 15 )

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
    /// Construct the shared MagAO-X widget base.
    xWidget( QWidget        *Parent = 0,                /**< [in] Optional Qt parent widget. */
             Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */
             )
        : QWidget( Parent, f )
    {
    }

    /// Destroy the shared MagAO-X widget base.
    ~xWidget() noexcept
    {
    }

  protected:
    /// Allow the base class to configure shared widget behavior after children are added.
    bool event( QEvent *event /**< [in] Qt event being processed. */ ) override;

    /// Find descendant combo boxes and enable shared popup sizing for them.
    void manageComboBoxPopups();
};

/// Update a combo-box popup to fit its widest current entry without widening the widget itself.
void updateXwComboBoxPopupWidth( QComboBox *combo /**< [in] Combo box whose popup width should be updated. */ );

/// Ensure a combo box uses shared popup sizing and keeps it updated as its model changes.
void manageXwComboBoxPopupWidth( QComboBox *combo /**< [in] Combo box whose popup width should be managed. */ );

template <class qT>
/// Apply the standard MagAO-X pixel font size to a widget.
void setXwFont( qT   *widg,       /**< [in] Widget receiving the standard MagAO-X font. */
                float scale = 1.0 /**< [in] Optional scale factor applied to the standard pixel size. */
)
{
    QFont qf = widg->font();
    qf.setPixelSize( XW_FONT_SIZE * scale + 0.5 );
    widg->setFont( qf );
}

inline bool xWidget::event( QEvent *event )
{
    bool handled = QWidget::event( event );

    if( event != nullptr &&
        ( event->type() == QEvent::ChildAdded || event->type() == QEvent::Polish || event->type() == QEvent::Show ) )
    {
        manageComboBoxPopups();
    }

    return handled;
}

inline void xWidget::manageComboBoxPopups()
{
    const auto combos = findChildren<QComboBox *>();
    for( auto *combo : combos )
    {
        manageXwComboBoxPopupWidth( combo );
    }
}

inline void updateXwComboBoxPopupWidth( QComboBox *combo )
{
    if( combo == nullptr || combo->view() == nullptr )
    {
        return;
    }

    QAbstractItemView *view = combo->view();
    view->setTextElideMode( Qt::ElideNone );

    int popupWidth   = std::max( combo->width(), combo->minimumSizeHint().width() );
    int contentWidth = view->sizeHintForColumn( 0 );
    if( contentWidth < 0 )
    {
        contentWidth = combo->minimumSizeHint().width();
    }

    int scrollBarWidth = 0;
    if( view->verticalScrollBar() != nullptr )
    {
        scrollBarWidth = view->verticalScrollBar()->sizeHint().width();
    }

    popupWidth = std::max( popupWidth, contentWidth + scrollBarWidth + 2 * view->frameWidth() + 16 );

    view->setMinimumWidth( popupWidth );
    if( view->window() != nullptr )
    {
        view->window()->setMinimumWidth( popupWidth );
    }
}

inline void manageXwComboBoxPopupWidth( QComboBox *combo )
{
    if( combo == nullptr )
    {
        return;
    }

    updateXwComboBoxPopupWidth( combo );

    QAbstractItemModel *model = combo->model();
    if( model == nullptr )
    {
        return;
    }

    QObject *managedModel = combo->property( "xwComboPopupModel" ).value<QObject *>();
    if( managedModel == model )
    {
        return;
    }

    combo->setProperty( "xwComboPopupModel", QVariant::fromValue( static_cast<QObject *>( model ) ) );

    QObject::connect( model,
                      &QAbstractItemModel::rowsInserted,
                      combo,
                      [combo]( const QModelIndex &, int, int ) { updateXwComboBoxPopupWidth( combo ); } );

    QObject::connect( model,
                      &QAbstractItemModel::rowsRemoved,
                      combo,
                      [combo]( const QModelIndex &, int, int ) { updateXwComboBoxPopupWidth( combo ); } );

    QObject::connect(
        model, &QAbstractItemModel::modelReset, combo, [combo]() { updateXwComboBoxPopupWidth( combo ); } );

    QObject::connect( model,
                      &QAbstractItemModel::dataChanged,
                      combo,
                      [combo]( const QModelIndex &, const QModelIndex &, const QVector<int> & )
                      { updateXwComboBoxPopupWidth( combo ); } );
}

} // namespace xqt

#endif
