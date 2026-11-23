/** \file pwfsAlignment.hpp
 * \brief Declare the measured and target PWFS pupil overlay.
 */
#ifndef pwfsAlignment_hpp
#define pwfsAlignment_hpp

#include <rtimv/rtimvInterfaces.hpp>

#include <QGraphicsScene>
#include <QGraphicsWidget>
#include <QPointer>
#include <array>

/// Display the fitter's three- or four-pupil geometry in image coordinates.
class pwfsAlignment : public rtimvOverlayInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA( IID "rtimv.overlayInterface/1.4" )
    Q_INTERFACES( rtimvOverlayInterface )

  protected:
    /// Scene items for one measured or target constellation, owned by the common graphics parent.
    struct drawing
    {
        /// Pupil outlines indexed by fitter quadrant.
        std::array<QGraphicsEllipseItem *, 4> m_circles{};

        /// Lines between adjacent pupils.
        std::array<QGraphicsLineItem *, 4> m_edges{};
    };

    /// Borrowed dictionary registered by the host.
    dictionaryT *m_dictionary{ nullptr };

    /// Scene to receive this instance's items.
    QPointer<QGraphicsScene> m_scene;

    /// Scene-owned graphics parent, also removed on detach.
    QPointer<QGraphicsWidget> m_items;

    /// Red outlines of the measured pupil fits.
    drawing m_measured;

    /// Lime outlines of the commanded set points.
    drawing m_target;

    /// INDI device publishing the pupil fits.
    std::string m_deviceName;

    /// User's display preference independent of fitter availability.
    bool m_enabled{ false };

    /// Read one finite fitter element, returning false if missing or invalidated.
    bool number( const std::string &key,  /**< [in] Fully qualified INDI element. */
                 double            &value /**< [out] Decoded number on success. */
    ) const;

    /// Allocate one set of passive outlines beneath the shared graphics parent.
    void createDrawing( drawing    &items, /**< [out] Child item pointers. */
                        const QPen &stroke /**< [in] Measured or target line style. */
    );

    /// Update available circles and connecting lines in the current layout.
    void updateDrawing( drawing           &items,  /**< [in,out] Outlines to update. */
                        const std::string &prefix, /**< [in] Empty for measurements or set- for target values. */
                        int                count,  /**< [in] Number of fitted pupils. */
                        double             height  /**< [in] Detector height in pixels. */
    );

  public:
    /// Construct a disabled overlay without scene items.
    pwfsAlignment() = default;

    /// Remove the instance's graphics before releasing its subscriptions.
    ~pwfsAlignment() override;

    /// Attach the configured fitter and register its property subscriptions.
    int attachOverlay( rtimvOverlayAccess       &access, /**< [in] Viewer scene and shared dictionary. */
                       mx::app::appConfigurator &config  /**< [in] Configuration containing pwfsAlignment.name. */
                       ) override;

    /// Remove this instance's scene items and release borrowed state.
    void detachOverlay();

    /// Redraw measured and target geometry from the dictionary.
    int updateOverlay() override;

    /// Toggle the overlay on uppercase P.
    void keyPressEvent( QKeyEvent *event /**< [in] Key pressed in the viewer. */ ) override;

    /// Return whether the user has requested the overlay.
    bool overlayEnabled() override;

    /// Request display when geometry is available.
    void enableOverlay() override;

    /// Hide all pupil geometry.
    void disableOverlay() override;

    /// Identify the overlay in the viewer's information panel.
    std::vector<std::string> info() override;
};

#endif // pwfsAlignment_hpp
