/** \file indiDictionary.hpp
 * \brief Declares the rtimv INDI-backed dictionary plugin.
 *
 * \author Jared Males
 */

#ifndef indiDictionary_hpp
#define indiDictionary_hpp

#include <rtimv/rtimvInterfaces.hpp>

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
class indiDictionary : public rtimvDictionaryInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA( IID "rtimv.dictionaryInterface/1.4" )
    Q_INTERFACES( rtimvDictionaryInterface )

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

  public slots:

    /// Check the status of the INDI connection.
    void checkConnection();

  public:
    /// Report plugin status for display in rtimv.
    virtual std::vector<std::string> info();
};

#endif // indiDictionary_hpp
