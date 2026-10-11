/** \file indiDictionary.hpp
 * \brief Declares the rtimv INDI-backed dictionary plugin.
 *
 * \author Jared Males
 */

#ifndef indiDictionary_hpp
#define indiDictionary_hpp

#include <rtimv/rtimvInterfaces.hpp>
#include <rtimv/rtimvCommandInterface.hpp>

#include <mutex>

#include <QObject>
#include <QtPlugin>
#include <QTimer>

#include <iostream>

#include <IndiClient.hpp>

// Forward decl:
class rtimvIndiClient;

/// rtimv dictionary using the INDI protocol
/**
 *
 */
class indiDictionary : public rtimvDictionaryInterface, public rtimvCommandInterface
{
    Q_OBJECT
#ifndef RTIMV_PLUGIN_TEST
    Q_PLUGIN_METADATA( IID "rtimv.dictionaryInterface/1.4" )
#endif
    Q_INTERFACES( rtimvDictionaryInterface rtimvCommandInterface )

  protected:
    std::string m_ipAddress{ "" };      ///< The IP address of the INDI server.
    int         m_port{ 0 };            ///< The port of the INDI server.
    int         m_checkTimeout{ 1000 }; ///< The timeout for checking the INDI connection, in msec.

    dictionaryT *m_dict{ nullptr }; ///< Shared rtimv INDI dictionary backing overlay state.

    std::mutex m_dictMutex; ///< Protects dictionary structure while threads add or snapshot keys.

    rtimvIndiClient *m_client{ nullptr }; ///< INDI client, recreated when the connection is lost.

    std::mutex m_clientMutex; ///< Protects access to the INDI client pointer and its lifecycle.

    bool m_enabled{ false }; ///< Whether or not this plugin is enabled.

    QTimer m_connTimer; ///< Timer used to periodically check the INDI connection.

    QMap<QString, rtimvPropertySnapshot> m_properties; ///< GUI-thread cache of complete defined properties.
    quint64 m_generation{0}; ///< Connection generation invalidating queued old replies.
    quint64 m_revision{0}; ///< Monotonic property receive order.

    /// Invalidate incoming state when a transport is lost; caller owns the client lock.
    void invalidateConnection();

    /// Check the live transport; overridden only by offline tests.
    virtual bool commandConnected() const;

    /// Submit a property to the live client; caller owns the client lock.
    virtual void writeCommand(const pcf::IndiProperty &property /**< [in] Typed outgoing request. */);

  public:
    /// Construct the plugin.
    indiDictionary();

    /// Destroy the plugin and shut down the INDI client.
    virtual ~indiDictionary();

    /// Attach this plugin to rtimv.
    virtual int
    attachDictionary( dictionaryT              *dict,  ///< [in] pointer to the rtimv dictionary, a std::map
                      mx::app::appConfigurator &config ///< [in] app configurator from which to config the connection
    );

    /// Identify the protocol supplied by this optional provider.
    QString commandProtocol() const override;

    /// Validate all vectors before submitting them once in order.
    bool sendCommands(const QVector<rtimvCommand> &commands, /**< [in] Ordered property vectors. */
                      QString &error /**< [out] Rejection or transport failure. */) override;

    /// Copy one complete cached property, valid only on the current connection.
    rtimvPropertySnapshot propertySnapshot(const QString &device, /**< [in] Source device. */
                                           const QString &property /**< [in] Source property. */) const override;

    /// Process a queued complete INDI message on the GUI thread.
    void receiveProperty(const pcf::IndiProperty &property, /**< [in] Incoming vector. */
                         quint64 generation, /**< [in] Connection that received the message. */
                         bool definition, /**< [in] True for a property definition. */
                         bool removed = false /**< [in] True for a property/device deletion. */);

  public slots:

    /// Check the status of the INDI connection.
    void checkConnection();

  public:
    /// Report plugin status for display in rtimv.
    virtual std::vector<std::string> info();
};

#endif // indiDictionary_hpp
