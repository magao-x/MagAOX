/** \file cursesINDI.hpp
 * \brief A curses-based terminal client for browsing and editing INDI properties.
 */

#include <array>
#include <cctype>
#include <climits>
#include <filesystem>
#include <format>
#include <fstream>

#include <thread>
#include <mutex>
#include <set>
#include <string_view>

#include "../../INDI/libcommon/IndiClient.hpp"
#include "cursesTableGrid.hpp"

///Simple utility to get the display string of a property
/**
 * \returns the properly formatted element value
 * \returns empty string on error
 */
std::string displayProperty( pcf::IndiProperty &ip /**< [in] the INDI property */ )
{
    std::string str = ip.getName();
    str += " [";

    std::string tstr = "n";
    if( ip.getType() == pcf::IndiProperty::Text )
    {
        tstr = "t";
    }
    if( ip.getType() == pcf::IndiProperty::Switch )
    {
        tstr = "s";
    }
    if( ip.getType() == pcf::IndiProperty::Light )
    {
        tstr = "l";
    }

    str += tstr;

    str += "]";

    if( ip.getState() == pcf::IndiProperty::Idle )
    {
        str += "~";
    }
    if( ip.getState() == pcf::IndiProperty::Ok )
    {
        str += "-";
    }
    if( ip.getState() == pcf::IndiProperty::Busy )
    {
        str += "*";
    }
    if( ip.getState() == pcf::IndiProperty::Alert )
    {
        str += "!";
    }

    return str;
}

///Simple utility to get the display value of an element
/**
 * \returns the properly formatted element value
 * \returns empty string if element is not in property
 */
std::string displayValue( pcf::IndiProperty &ip, /**< [in] the INDI property */
                          std::string       &el  /**< [in] the name of the element */
)
{
    if( !ip.find( el ) )
    {
        return "";
    }

    if( ip.getType() == pcf::IndiProperty::Switch )
    {
        if( ip[el].getSwitchState() == pcf::IndiElement::Off )
        {
            return "|O|";
        }
        if( ip[el].getSwitchState() == pcf::IndiElement::On )
        {
            return "|X|";
        }
        return pcf::IndiElement::getSwitchStateString( ip[el].getSwitchState() );
    }
    else
    {
        return ip[el].getValue();
    }
}

/// Terminal INDI client which displays every known element in a curses table.
/** Connects to the INDI server on localhost:7624, tracks all defined properties,
 * and lets the user navigate, search device.property.element names, and edit writable elements.
 * INDI messages are written to a log file which can be followed in another terminal.
 */
class cursesINDI : public pcf::IndiClient, public cursesTableGrid
{

  public:
    /// Count of pending full-table redraws, incremented by INDI callbacks and consumed by the draw thread.
    int m_redraw{ 0 };

    /// Count of pending in-place value updates, incremented by INDI callbacks and consumed by the draw thread.
    int m_update{ 0 };

    /// Current curses cursor visibility, as passed to curs_set.
    int m_cursStat{ 1 };

    /// Window for user prompts and input (search, edit, confirm).  Owned by this class.
    WINDOW *w_interactWin{ nullptr };

    /// Window showing the full key and value of the currently selected element.  Owned by this class.
    WINDOW *w_curvalWin{ nullptr };

    /// Window showing FSM alert notices.  Owned by this class.
    WINDOW *w_attentionWin{ nullptr };

    /// Window below the table showing the count of displayed elements.  Owned by this class.
    WINDOW *w_countWin{ nullptr };

    /// Set to request the draw thread to exit.
    bool m_shutdown{ false };

    /// Set when the INDI connection is lost rather than closed by the user.
    bool m_connectionLost{ false };

    /// The thread which redraws and updates the table.
    std::thread m_drawThread;

    /// Protects the property and element maps and all curses drawing.
    std::mutex m_drawMutex;

    /// Path of the file to which INDI messages are written.
    std::string m_msgFile{ "/tmp/cursesINDI_logs.txt" };

    /// Output stream for INDI messages, opened on m_msgFile.
    std::ofstream m_msgout;

    /// Number of messages written since m_msgFile was last truncated.
    int m_msgsPrinted{ 0 };

    /// Maximum number of messages written before m_msgFile is truncated.
    int m_msgsMax{ 10000 };

    /// Constructor, which connects to the INDI server on localhost and opens the message file.
    /** \throws std::filesystem::filesystem_error if the message file cannot be opened
     */
    cursesINDI( const std::string &szName,           /**< [in] the client name */
                const std::string &szVersion,        /**< [in] the client version */
                const std::string &szProtocolVersion /**< [in] the INDI protocol version */
    );

    /// Destructor, which deletes the curses windows.
    ~cursesINDI();

    /// Map of INDI properties keyed by their unique device.name key.
    typedef std::map<std::string, pcf::IndiProperty> propMapT;

    /// Value type of propMapT.
    typedef propMapT::value_type propMapValueT;

    /// Iterator type of propMapT.
    typedef propMapT::iterator propMapIteratorT;

    /// All properties received from the server.
    propMapT knownProps;

    /// Identifies a single INDI element and its row in the table.
    struct elementSpec
    {
        std::string propKey;        ///< The unique key of the parent property in knownProps
        std::string device;         ///< The device name
        std::string propertyName;   ///< The property name
        std::string name;           ///< The element name
        int         tableRow{ -1 }; ///< The row of this element in the table, -1 if not displayed
    };

    /// Compare dotted element keys by device, property, then element.
    struct elementKeyLess
    {
        /// Order each field lexicographically, keeping a name before names extending it.
        bool operator()( std::string_view left, /**< [in] first device.property.element key */
                         std::string_view right /**< [in] second device.property.element key */ ) const;
    };

    /// Map of elements keyed by device.property.element and ordered by its separate fields.
    typedef std::map<std::string, elementSpec, elementKeyLess> elementMapT;

    /// Value type of elementMapT.
    typedef elementMapT::value_type elementMapValueT;

    /// Iterator type of elementMapT.
    typedef elementMapT::iterator elementMapIteratorT;

    /// All elements of the properties in knownProps, in table order.
    elementMapT knownElements;

    /// Identity of the selected element, retained when table rows are reordered or deleted.
    std::string m_selectedKey;

    /// True while search owns keyboard input, including after Ctrl+L from another column.
    bool m_deviceSearching{ false };

    /// The dot-separated device, property, and element query typed so far.
    std::string m_deviceTarget;

    /// Store a newly defined or updated property and its elements, and schedule a redraw or update.
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Remove a deleted property, or all properties of a deleted device, and schedule a redraw.
    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] the property or device to delete */ );

    /// Write an INDI message, color coded by priority, to the message file.
    virtual void handleMessage( const pcf::IndiProperty &ipRecv /**< [in] the property containing the message */ );

    /// Handle a property update the same as a definition.
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the received property */ );

    /// Process incoming INDI requests.
    virtual void execute();

    /// Set the curses cursor visibility.
    void cursStat( int cs /**< [in] the new visibility, as passed to curs_set */ );

    /// Get the curses cursor visibility.
    /**
     * \returns the current cursor visibility
     */
    int cursStat();

    /// Create the auxiliary windows and start the draw thread.
    void startUp();

    /// Stop the INDI client and the draw thread, and delete the windows.
    void shutDown();

    /// Thread entry point which calls drawThreadExec.
    static void _drawThreadStart( cursesINDI *c /**< [in] the cursesINDI instance to run */ );

    /// Start the draw thread.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int drawThreadStart();

    /// Execute the draw thread, servicing pending redraws and updates until shutdown or disconnect.
    void drawThreadExec();

    /// Rebuild the table contents from knownElements and redraw it.
    void redrawTable();

    /// Update changed property states and element values in place.
    void updateTable();

    /// Update the current-value window.
    void updateCurVal();

    /// Move the selected cell, locking the draw mutex.
    void moveCurrent( int nextY, /**< [in] the new row */
                      int nextX  /**< [in] the new column */
    );

    /// Move the selected cell and update the prompt windows.  The caller must hold m_drawMutex.
    void _moveCurrent( int nextY, /**< [in] the new row */
                       int nextX  /**< [in] the new column */
    );

    /// Handle a key press from the user.
    void keyPressed( int ch /**< [in] the key code */ );

    /// Rank a field against one case-folded search segment without crossing field boundaries.
    /** \returns 0 for a prefix (including an empty segment), 1 for a substring,
     * 2 for a subsequence, or 3 for no match
     */
    static int searchSegmentRank( std::string      field, /**< [in] device, property, or element name */
                                  std::string_view query /**< [in] lowercase query segment */ );

    /// Find the first displayed row matching a case-insensitive device.property.element query.
    /** Rank prefixes, substrings, and subsequences in device, property, then element order.
     * Empty or omitted segments match any name; ties retain table order.
     * \returns -1 for an empty query, more than three segments, or no displayed match
     */
    static int findSearchRow( const elementMapT &elements, /**< [in] elements in table order */
                              std::string        query /**< [in] dot-separated search text */ );

    /// Update a dot-separated search and jump to its best match while preserving table order.
    void deviceSearch( int ch /**< [in] the key code */ );

    /// Recreate the element count window after the table is drawn.
    /**
     * \returns the result of postPrint
     */
    virtual int postDraw();

    /// Print the count of displayed elements in the count window.
    /**
     * \returns 0
     */
    virtual int postPrint();
};

inline bool cursesINDI::elementKeyLess::operator()( std::string_view left, std::string_view right ) const
{
    for( int field = 0; field < 2; ++field )
    {
        auto leftPeriod  = left.find( '.' );
        auto rightPeriod = right.find( '.' );
        int  comparison  = left.substr( 0, leftPeriod ).compare( right.substr( 0, rightPeriod ) );

        if( comparison != 0 )
        {
            return comparison < 0;
        }

        if( leftPeriod == std::string_view::npos || rightPeriod == std::string_view::npos )
        {
            return leftPeriod == std::string_view::npos && rightPeriod != std::string_view::npos;
        }

        left.remove_prefix( leftPeriod + 1 );
        right.remove_prefix( rightPeriod + 1 );
    }

    return left < right;
}

cursesINDI::cursesINDI( const std::string &szName, const std::string &szVersion, const std::string &szProtocolVersion )
    : pcf::IndiClient( szName, szVersion, szProtocolVersion, "127.0.0.1", 7624 )
{
    m_yTop = 6;
    colWidth( { 4, 19, 18, 18, 18 } );

    m_yBot = 1;

    m_msgout.open( m_msgFile );

    if( !m_msgout )
    {
        throw std::filesystem::filesystem_error(
            std::format( "can't open log file.  you probably need to delete {}", m_msgFile ),
            std::error_code( EPERM, std::system_category() ) );
    }
}

cursesINDI::~cursesINDI()
{
    if( w_interactWin )
    {
        delwin( w_interactWin );
    }
    if( w_curvalWin )
    {
        delwin( w_curvalWin );
    }
    if( w_attentionWin )
    {
        delwin( w_attentionWin );
    }
    if( w_countWin )
    {
        delwin( w_countWin );
    }
}

void cursesINDI::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    if( !ipRecv.hasValidDevice() && !ipRecv.hasValidName() )
    {
        return;
    }

    std::pair<propMapIteratorT, bool> result;

    std::lock_guard<std::mutex> lock( m_drawMutex );

    result = knownProps.insert( propMapValueT( ipRecv.createUniqueKey(), ipRecv ) );

    if( result.second == false )
    {
        result.first->second = ipRecv; //We already have it, so we're already registered
    }
    else
    {
        sendGetProperties( ipRecv ); //Otherwise register for it
    }

    auto elIt = ipRecv.getElements().begin();

    while( elIt != ipRecv.getElements().end() )
    {
        elementSpec es;
        es.propKey      = ipRecv.createUniqueKey();
        es.device       = ipRecv.getDevice();
        es.propertyName = ipRecv.getName();
        es.name         = elIt->second.getName();

        std::string key      = es.propKey + "." + es.name;
        auto        elResult = knownElements.insert( elementMapValueT( key, es ) );

        if( elResult.second == true )
        {
            //If result is new insert, add to TUI table if filter requires
            ++m_redraw;
        }
        else
        {
            //Or just update the table.
            //Should check if this element actually changed....
            ++m_update;
        }
        ++elIt;
    }
}

void cursesINDI::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> lock( m_drawMutex );

    //if(fpout) *fpout << "got delete property" << std::endl;

    if( ipRecv.hasValidDevice() )
    {
        if( !ipRecv.hasValidName() )
        {
            //if(fpout) *fpout << "will delete: " << ipRecv.getDevice() << "\n";

            for( elementMapIteratorT elIt = knownElements.begin(); elIt != knownElements.end(); )
            {
                if( elIt->second.device == ipRecv.getDevice() )
                {
                    elIt = knownElements.erase( elIt );
                }
                else
                {
                    ++elIt;
                }
            }

            for( propMapIteratorT pIt = knownProps.begin(); pIt != knownProps.end(); )
            {
                if( pIt->first == ipRecv.createUniqueKey() )
                {
                    pIt = knownProps.erase( pIt );
                }
                else
                {
                    ++pIt;
                }
            }
        }
        else
        {
            //if(fpout) *fpout << "will delete: " << ipRecv.createUniqueKey() << "\n";

            for( elementMapIteratorT elIt = knownElements.begin(); elIt != knownElements.end(); )
            {
                if( elIt->second.propKey == ipRecv.createUniqueKey() )
                {
                    elIt = knownElements.erase( elIt );
                }
                else
                {
                    ++elIt;
                }
            }

            knownProps.erase( ipRecv.createUniqueKey() );
        }
    }

    ++m_redraw;
}

void cursesINDI::handleMessage( const pcf::IndiProperty &ipRecv )
{
    tm     bdt; //broken down time
    time_t tt = ipRecv.getTimeStamp().getTimeVal().tv_sec;
    gmtime_r( &tt, &bdt );

    char tstr1[25];
    strftime( tstr1, sizeof( tstr1 ), "%H:%M:%S", &bdt );
    char tstr2[11];
    snprintf( tstr2,
              sizeof( tstr2 ),
              ".%06i",
              static_cast<int>( ipRecv.getTimeStamp().getTimeVal().tv_usec ) ); //casting in case we switch to int64_t

    std::string msg = ipRecv.getMessage();

    if( msg.size() > 4 )
    {
        std::string prio = msg.substr( 0, 4 );
        if( prio == "NOTE" )
        {
            m_msgout << "\033[1m";
        }
        else if( prio == "CAUT" ) //given by stateRuleEngine
        {
            m_msgout << "\033[93m\033[1m";
        }
        else if( prio == "WARN" )
        {
            m_msgout << "\033[93m\033[1m";
        }
        else if( prio == "ERR " )
        {
            m_msgout << "\033[91m\033[1m";
        }
        else if( prio == "CRIT" )
        {
            m_msgout << "\033[41m\033[1m";
        }
        else if( prio == "ALRT" || prio == "ALER" )
        {
            m_msgout << "\033[41m\033[1m";
        }
        else if( prio == "EMER" )
        {
            m_msgout << "\033[41m\033[1m";
        }
    }
    m_msgout << std::string( tstr1 ) << std::string( tstr2 ) << " [" << ipRecv.getDevice() << "] " << msg;

    m_msgout << "\033[0m";
    m_msgout << std::endl;

    ++m_msgsPrinted;
    if( m_msgsPrinted > m_msgsMax )
    {
        m_msgout.close();
        m_msgout.open( m_msgFile );
        m_msgsPrinted = 0;
    }
}

void cursesINDI::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    handleDefProperty( ipRecv );
}

void cursesINDI::execute()
{
    processIndiRequests( false );
}

void cursesINDI::cursStat( int cs )
{
    m_cursStat = cs;
    curs_set( m_cursStat );
}

int cursesINDI::cursStat()
{
    return m_cursStat;
}

void cursesINDI::startUp()
{
    if( w_interactWin == nullptr )
    {
        w_interactWin = newwin( 1, m_minWidth, m_yTop - 3, m_xLeft );
    }

    if( w_curvalWin == nullptr )
    {
        w_curvalWin = newwin( 1, m_minWidth, m_yTop - 2, m_xLeft );
    }

    if( w_attentionWin == nullptr )
    {
        w_attentionWin = newwin( 1, m_minWidth, m_yTop - 4, m_xLeft );
    }

    keypad( w_interactWin, TRUE );

    m_shutdown = false;
    drawThreadStart();
}

void cursesINDI::shutDown()
{
    if( getQuitProcess() && !m_shutdown )
    {
        m_connectionLost = true;
    }

    m_shutdown = true;

    quitProcess();
    deactivate();

    m_drawThread.join();

    m_cellContents.clear();

    if( w_interactWin )
    {
        delwin( w_interactWin );
    }
    w_interactWin = nullptr;

    if( w_countWin )
    {
        delwin( w_countWin );
    }
    w_countWin = nullptr;
}

inline void cursesINDI::_drawThreadStart( cursesINDI *c )
{
    c->drawThreadExec();
}

inline int cursesINDI::drawThreadStart()
{
    try
    {
        m_drawThread = std::thread( _drawThreadStart, this );
    }
    catch( const std::exception &e )
    {
        return -1;
    }
    catch( ... )
    {
        return -1;
    }

    if( !m_drawThread.joinable() )
    {
        return -1;
    }

    return 0;
}

inline void cursesINDI::drawThreadExec()
{
    while( !m_shutdown && !getQuitProcess() )
    {
        ////if(fpout) *fpout << "draw thread . . ." << std::endl;
        if( m_redraw > 0 )
        {
            redrawTable();
        }

        if( m_update > 0 )
        {
            updateTable();
        }

        std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>( 250000000 ) );
    }

    if( getQuitProcess() && !m_shutdown )
    {
        m_connectionLost = true;

        m_cellContents.clear();
        redrawTable();
        m_shutdown = true;
    }
}

void cursesINDI::redrawTable()
{
    std::lock_guard<std::mutex> lock( m_drawMutex );

    int start_redraw = m_redraw;

    //if(fpout) *fpout << "redrawTable: " << m_redraw << std::endl;

    m_cellContents.clear();

    bool fsmAlerts = false;

    std::set<std::string> alertDev;

    for( elementMapIteratorT es = knownElements.begin(); es != knownElements.end(); ++es )
    {
        //if(fpout) *fpout << knownProps[es->second.propKey].getName() << " " <<
        //knownProps[es->second.propKey].getState() << "\n";

        if( knownProps[es->second.propKey].getName() == "fsm" &&
            knownProps[es->second.propKey].getState() == pcf::IndiProperty::Alert )
        {
            fsmAlerts = true;
            alertDev.insert( knownProps[es->second.propKey].getDevice() );
        }
        std::vector<std::string> s;

        s.resize( m_colFraction.size() );

        s[0] = std::to_string( m_cellContents.size() + 1 );
        s[1] = knownProps[es->second.propKey].getDevice();
        s[2] = displayProperty( knownProps[es->second.propKey] );

        s[3] = es->second.name;

        s[4] = displayValue( knownProps[es->second.propKey], es->second.name );

        m_cellContents.push_back( s );
        es->second.tableRow = m_cellContents.size() - 1;
    }

    if( knownElements.empty() )
    {
        m_startRow = 0;
        m_currY    = 0;
    }
    else
    {
        // Keep the selected identity; if removed, use its successor or the last remaining row.
        auto selected = knownElements.lower_bound( m_selectedKey );

        if( selected == knownElements.end() )
        {
            --selected;
        }

        size_t row = selected->second.tableRow;
        m_startRow = row > m_currY ? row - m_currY : 0;
        m_currY    = row - m_startRow;
    }

    draw();

    //if(fpout) *fpout << "fsmAlerts: " << fsmAlerts << "\n";

    int cx, cy;
    getyx( w_interactWin, cy, cx );
    int cs = cursStat();
    cursStat( 0 );

    wclear( w_attentionWin );
    if( fsmAlerts )
    {
        std::string alrt = "!! FSM alert: " + *alertDev.begin();
        if( alertDev.size() > 1 )
        {
            alrt += " (+" + std::to_string( alertDev.size() - 1 ) + ")";
        }

        wprintw( w_attentionWin, "%s", alrt.c_str() );
    }
    wrefresh( w_attentionWin );

    wmove( w_interactWin, cy, cx );
    cursStat( cs );
    wrefresh( w_interactWin );

    m_redraw -= start_redraw;
    if( m_redraw < 0 )
    {
        m_redraw = 0;
    }

    _moveCurrent( m_currY, m_currX );
}

void cursesINDI::updateTable()
{
    if( m_redraw )
    {
        return; //Pending redraw, so we skip it and let that take care of it.
    }

    std::lock_guard<std::mutex> lock( m_drawMutex );

    int start_update = m_update;

    //if(fpout) *fpout << "updateTable: " << m_update << std::endl;
    int cx, cy;

    getyx( w_interactWin, cy, cx );
    int cs = cursStat();

    updateCurVal();

    bool                  fsmAlerts{ false };
    std::set<std::string> alertDev;

    for( auto it = knownElements.begin(); it != knownElements.end(); ++it )
    {
        //if(fpout) *fpout << knownProps[it->second.propKey].getName() << " " <<
        //knownProps[it->second.propKey].getState() << "\n";

        if( knownProps[it->second.propKey].getName() == "fsm" &&
            knownProps[it->second.propKey].getState() == pcf::IndiProperty::Alert )
        {
            fsmAlerts = true;
            alertDev.insert( knownProps[it->second.propKey].getDevice() );
        }

        if( it->second.tableRow == -1 )
        {
            continue;
        }

        if( m_cellContents[it->second.tableRow][2] != displayProperty( knownProps[it->second.propKey] ) )
        {
            m_cellContents[it->second.tableRow][2] = displayProperty( knownProps[it->second.propKey] );

            if( it->second.tableRow - m_startRow < (size_t)tabHeight() ) //It's currently displayed
            {
                cursStat( 0 );
                wclear( m_gridWin[it->second.tableRow - m_startRow][2] );
                if( hasContent( it->second.tableRow, 2 ) )
                {
                    wprintw( m_gridWin[it->second.tableRow - m_startRow][2],
                             "%s",
                             m_cellContents[it->second.tableRow][2].c_str() );
                }
                wrefresh( m_gridWin[it->second.tableRow - m_startRow][2] );
                wmove( w_interactWin, cy, cx );
                cursStat( cs );
                wrefresh( w_interactWin );
            }
        }

        if( m_cellContents[it->second.tableRow][4] !=
            displayValue( knownProps[it->second.propKey], it->second.name ) ) //.getValue())
        {
            m_cellContents[it->second.tableRow][4] =
                displayValue( knownProps[it->second.propKey],
                              it->second.name ); //knownProps[it->second.propKey][it->second.name].getValue();

            if( it->second.tableRow - m_startRow < (size_t)tabHeight() ) //It's currently displayed
            {
                cursStat( 0 );
                wclear( m_gridWin[it->second.tableRow - m_startRow][4] );
                if( hasContent( it->second.tableRow, 4 ) )
                {
                    wprintw( m_gridWin[it->second.tableRow - m_startRow][4],
                             "%s",
                             m_cellContents[it->second.tableRow][4].c_str() );
                }
                wrefresh( m_gridWin[it->second.tableRow - m_startRow][4] );
                wmove( w_interactWin, cy, cx );
                cursStat( cs );
                wrefresh( w_interactWin );
            }
        };
        //updateContents( it->second.tableRow, 4,  knownProps[it->second.propKey][it->second.name].getValue());
    }

    wattron( m_gridWin[m_currY][m_currX], A_REVERSE );

    //if(fpout) *fpout << "fsmAlerts: " << fsmAlerts << "\n";

    getyx( w_interactWin, cy, cx );
    cs = cursStat();
    cursStat( 0 );
    wclear( w_attentionWin );
    if( fsmAlerts )
    {
        std::string alrt = "!! FSM alert: " + *alertDev.begin();
        if( alertDev.size() > 1 )
        {
            alrt += " (+" + std::to_string( alertDev.size() - 1 ) + ")";
        }

        wprintw( w_attentionWin, "%s", alrt.c_str() );
    }
    wrefresh( w_attentionWin );
    wmove( w_interactWin, cy, cx );
    cursStat( cs );
    wrefresh( w_interactWin );

    //print();

    //    wmove(w_interactWin,cy,cx);
    //    cursStat(cs);
    //    wrefresh(w_interactWin);

    m_update -= start_update;
    if( m_update < 0 )
    {
        m_update = 0;
    }
}

void cursesINDI::updateCurVal()
{
    int cx, cy;
    getyx( w_interactWin, cy, cx );
    int cs = cursStat();

    cursStat( 0 );
    wclear( w_curvalWin );

    auto it = knownElements.find( m_selectedKey );

    if( it == knownElements.end() )
    {
        wrefresh( w_curvalWin );
        wrefresh( w_interactWin );
        cursStat( cs );
        return;
    }

    std::string cval = "> " + it->second.propKey + "." + it->second.name + " = ";

    cval += displayValue( knownProps[it->second.propKey], it->second.name );

    wprintw( w_curvalWin, "%s", cval.c_str() );
    wrefresh( w_curvalWin );

    wmove( w_interactWin, cy, cx );
    cursStat( cs );
    wrefresh( w_interactWin );

    return;
}

void cursesINDI::moveCurrent( int nextY, int nextX )
{
    std::lock_guard<std::mutex> lock( m_drawMutex );

    _moveCurrent( nextY, nextX );
}

void cursesINDI::_moveCurrent( int nextY, int nextX )
{
    int         currX       = m_currX;
    int         currY       = m_currY;
    std::string selectedKey = m_selectedKey;

    moveSelected( nextY, nextX );

    m_selectedKey.clear();

    for( const auto &[key, element] : knownElements )
    {
        if( element.tableRow >= 0 && static_cast<size_t>( element.tableRow ) == m_startRow + m_currY )
        {
            m_selectedKey = key;
            break;
        }
    }

    //if(fpout) *fpout << "moved: " << nextX << " " << nextY << " " << currX << "->" << m_currX << " " << currY << "->"
    //<< m_currY << std::endl;

    if( m_deviceSearching && nextX != 1 && ( nextX != currX || nextY != currY ) )
    {
        wclear( w_interactWin );
        wrefresh( w_interactWin );
        m_deviceSearching = false;
    }

    updateCurVal();

    // Background redraws must preserve search focus without moving the selected cell.
    if( m_deviceSearching && nextX != 1 )
    {
        return;
    }

    if( nextX == 1 )
    {
        if( currX != nextX )
        {
            wclear( w_interactWin );
            wprintw( w_interactWin, "search: %s", m_deviceSearching ? m_deviceTarget.c_str() : "" );
            wrefresh( w_interactWin );
        }
    }
    else if( currY != nextY || currX == 1 || selectedKey != m_selectedKey )
    {
        wclear( w_interactWin );
        if( m_currY + m_startRow >= knownElements.size() )
        {
            wrefresh( w_interactWin );
            return;
        }

        auto it = knownElements.begin();
        while( it != knownElements.end() )
        {
            if( (size_t)it->second.tableRow == m_currY + m_startRow )
            {
                break;
            }
            ++it;
        }

        if( it == knownElements.end() )
        {
            wrefresh( w_interactWin );
            return;
        }

        if( knownProps[it->second.propKey].getPerm() != pcf::IndiProperty::ReadWrite )
        {
            wrefresh( w_interactWin );
            return;
        }

        if( knownProps[it->second.propKey].getType() == pcf::IndiProperty::Text )
        {
            wprintw( w_interactWin, "(e)dit this text" );
        }
        else if( knownProps[it->second.propKey].getType() == pcf::IndiProperty::Number )
        {
            wprintw( w_interactWin, "(e)dit this number" );
        }
        else if( knownProps[it->second.propKey].getType() == pcf::IndiProperty::Switch )
        {
            wprintw( w_interactWin, "(p)ress or (t)oggle this switch" );
        }

        wrefresh( w_interactWin );
    }
}

inline int cursesINDI::searchSegmentRank( std::string field, std::string_view query )
{
    for( char &ch : field )
    {
        ch = std::tolower( static_cast<unsigned char>( ch ) );
    }

    auto position = field.find( query );

    if( position == 0 )
    {
        return 0;
    }

    if( position != std::string::npos )
    {
        return 1;
    }

    size_t matched = 0;

    for( char ch : field )
    {
        if( ch == query[matched] )
        {
            ++matched;

            if( matched == query.size() )
            {
                return 2;
            }
        }
    }

    return 3;
}

inline int cursesINDI::findSearchRow( const elementMapT &elements, std::string query )
{
    if( query.empty() )
    {
        return -1;
    }

    for( char &ch : query )
    {
        ch = std::tolower( static_cast<unsigned char>( ch ) );
    }

    std::array<std::string_view, 3> segments{};
    std::string_view                remaining = query;

    for( size_t i = 0; i < segments.size(); ++i )
    {
        auto period = remaining.find( '.' );
        segments[i] = remaining.substr( 0, period );

        if( period == std::string_view::npos )
        {
            break;
        }

        if( i == segments.size() - 1 )
        {
            return -1;
        }

        remaining.remove_prefix( period + 1 );
    }

    std::array<int, 3> bestRank{ 3, 3, 3 };
    int                bestRow = -1;

    for( const auto &[key, element] : elements )
    {
        if( element.tableRow < 0 )
        {
            continue;
        }

        std::array<int, 3> rank{ searchSegmentRank( element.device, segments[0] ),
                                 searchSegmentRank( element.propertyName, segments[1] ),
                                 searchSegmentRank( element.name, segments[2] ) };

        if( rank[0] == 3 || rank[1] == 3 || rank[2] == 3 )
        {
            continue;
        }

        if( rank < bestRank )
        {
            bestRank = rank;
            bestRow  = element.tableRow;
        }
    }

    return bestRow;
}

void cursesINDI::deviceSearch( int ch )
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_drawMutex );

        if( ch == KEY_BACKSPACE && m_deviceSearching && !m_deviceTarget.empty() )
        {
            m_deviceTarget.pop_back();
        }
        else if( ch >= 0 && ch <= UCHAR_MAX && std::isprint( static_cast<unsigned char>( ch ) ) )
        {
            if( !m_deviceSearching )
            {
                m_deviceTarget.clear();
            }

            m_deviceTarget += static_cast<char>( ch );
        }
        else
        {
            return;
        }

        m_deviceSearching = true;
        wclear( w_interactWin );
        wprintw( w_interactWin, "search: %s", m_deviceTarget.c_str() );
        wrefresh( w_interactWin );

        int row = findSearchRow( knownElements, m_deviceTarget );

        if( row < 0 )
        {
            return;
        }

        m_startRow = row;
        _moveCurrent( 0, 1 );
    }

    redrawTable();
}

void cursesINDI::keyPressed( int ch )
{
    if( ch == ( 0x1f & 'l' ) )
    {
        std::lock_guard<std::mutex> lock( m_drawMutex );

        m_deviceTarget.clear();
        m_deviceSearching = true;
        wclear( w_interactWin );
        wprintw( w_interactWin, "search: " );
        wrefresh( w_interactWin );
        return;
    }

    // Search accepts input from the device column or after an explicit Ctrl+L focus.
    if( m_currX == 1 || m_deviceSearching )
    {
        deviceSearch( ch );
        return;
    }

    switch( ch )
    {
    case 'e':
    {
        if( m_currY + m_startRow >= knownElements.size() )
        {
            break;
        }
        auto it = knownElements.begin();
        while( it != knownElements.end() )
        {
            if( (size_t)it->second.tableRow == m_currY + m_startRow )
            {
                break;
            }
            ++it;
        }

        if( it == knownElements.end() )
        {
            break;
        }

        //Can't edit a switch
        if( knownProps[it->second.propKey].getType() != pcf::IndiProperty::Text &&
            knownProps[it->second.propKey].getType() != pcf::IndiProperty::Number )
        {
            break;
        }

        cursStat( 1 );

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wprintw( w_interactWin, "set %s.%s=", it->second.propKey.c_str(), it->second.name.c_str() );
            wrefresh( w_interactWin );
        }

        bool        escape = false;
        std::string newStr;
        int         nch;
        while( ( nch = wgetch( w_interactWin ) ) != '\n' )
        {
            if( nch == ERR )
            {
                if( getQuitProcess() )
                {
                    //If the IndiConnection has set 'quitProces' but no other shutdown
                    //has been issued then we record this as a lost connection.
                    if( !m_shutdown )
                    {
                        m_connectionLost = true;
                    }
                    break;
                }
                else
                {
                    continue;
                }
            }
            else if( nch == 3 )
            {
                m_shutdown = true;
                break;
            }

            cursStat( 1 );

            if( nch == 27 )
            {
                std::lock_guard<std::mutex> lock( m_drawMutex );
                wclear( w_interactWin );
                wrefresh( w_interactWin );
                escape = true;
                break;
            }
            if( nch == KEY_BACKSPACE )
            {
                if( newStr.size() > 0 )
                {
                    std::lock_guard<std::mutex> lock( m_drawMutex );
                    newStr.erase( newStr.size() - 1, 1 );
                    wprintw( w_interactWin, "\b \b" );
                    wrefresh( w_interactWin );
                }
            }
            else if( std::isprint( nch ) )
            {
                std::lock_guard<std::mutex> lock( m_drawMutex );
                wprintw( w_interactWin, "%c", nch );
                wrefresh( w_interactWin );

                newStr += (char)nch;
            }
        }
        if( escape )
        {
            break;
        }
        if( m_shutdown )
        {
            break;
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wprintw( w_interactWin,
                     "send %s.%s=%s? y/n [n]",
                     it->second.propKey.c_str(),
                     it->second.name.c_str(),
                     newStr.c_str() );
            wrefresh( w_interactWin );
        }

        nch = 0;
        while( ( nch = wgetch( w_interactWin ) ) == ERR )
        {
        }

        if( nch == 3 )
        {
            m_shutdown = true;
            break;
        }

        if( nch == 'y' )
        {
            pcf::IndiProperty ipSend( knownProps[it->second.propKey].getType() );

            ipSend.setDevice( knownProps[it->second.propKey].getDevice() );
            ipSend.setName( knownProps[it->second.propKey].getName() );
            ipSend.add( pcf::IndiElement( it->second.name ) );
            //if(fpout) *fpout << "newStr: " << newStr << std::endl;

            ipSend[it->second.name].setValue( newStr );
            sendNewProperty( ipSend );
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wrefresh( w_interactWin );
        }

        break;
    } //case 'e'
    case 't':
    {
        if( m_currY + m_startRow >= knownElements.size() )
        {
            break;
        }
        auto it = knownElements.begin();
        while( it != knownElements.end() )
        {
            if( (size_t)it->second.tableRow == m_currY + m_startRow )
            {
                break;
            }
            ++it;
        }

        if( it == knownElements.end() )
        {
            break;
        }

        if( !knownProps[it->second.propKey].find( it->second.name ) )
        {
            break; //Just a check.
        }

        if( knownProps[it->second.propKey].getType() != pcf::IndiProperty::Switch )
        {
            break;
        }

        std::string                       toggleString;
        pcf::IndiElement::SwitchStateType toggleState;
        if( knownProps[it->second.propKey][it->second.name].getSwitchState() == pcf::IndiElement::Off )
        {
            toggleString = "On";
            toggleState  = pcf::IndiElement::On;
        }
        else if( knownProps[it->second.propKey][it->second.name].getSwitchState() == pcf::IndiElement::On )
        {
            toggleString = "Off";
            toggleState  = pcf::IndiElement::Off;
        }
        else
        {
            break; //would happen fo state unknown
        }

        cursStat( 1 );

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wprintw( w_interactWin,
                     "toggle %s.%s to %s?",
                     it->second.propKey.c_str(),
                     it->second.name.c_str(),
                     toggleString.c_str() );
            wrefresh( w_interactWin );
        }

        int nch = 0;
        while( ( nch = wgetch( w_interactWin ) ) == ERR )
        {
        }

        if( nch == 3 )
        {
            m_shutdown = true;
            break;
        }

        if( nch == 'y' )
        {
            pcf::IndiProperty ipSend( knownProps[it->second.propKey].getType() );

            ipSend.setDevice( knownProps[it->second.propKey].getDevice() );
            ipSend.setName( knownProps[it->second.propKey].getName() );
            ipSend.add( pcf::IndiElement( it->second.name ) );
            ipSend[it->second.name].setSwitchState( toggleState );
            sendNewProperty( ipSend );
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wrefresh( w_interactWin );
        }

        break;
    } //case 't'
    case 'p':
    {
        if( m_currY + m_startRow >= knownElements.size() )
        {
            break;
        }
        auto it = knownElements.begin();
        while( it != knownElements.end() )
        {
            if( (size_t)it->second.tableRow == m_currY + m_startRow )
            {
                break;
            }
            ++it;
        }

        if( it == knownElements.end() )
        {
            break;
        }

        if( !knownProps[it->second.propKey].find( it->second.name ) )
        {
            break; //Just a check.
        }

        if( knownProps[it->second.propKey].getType() != pcf::IndiProperty::Switch )
        {
            break;
        }
        cursStat( 1 );

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wprintw( w_interactWin, "press switch %s.%s?", it->second.propKey.c_str(), it->second.name.c_str() );
            wrefresh( w_interactWin );
        }

        int nch = 0;
        while( ( nch = wgetch( w_interactWin ) ) == ERR )
        {
        }

        if( nch == 3 )
        {
            m_shutdown = true;
            break;
        }

        if( nch == 'y' )
        {
            pcf::IndiProperty ipSend( knownProps[it->second.propKey].getType() );

            ipSend.setDevice( knownProps[it->second.propKey].getDevice() );
            ipSend.setName( knownProps[it->second.propKey].getName() );

            //Must add all elements
            for( auto elit = knownProps[it->second.propKey].getElements().begin();
                 elit != knownProps[it->second.propKey].getElements().end();
                 ++elit )
            {
                ipSend.add( elit->second );
                if( knownProps[it->second.propKey].getRule() != pcf::IndiProperty::AnyOfMany )
                {
                    ipSend[elit->first].setSwitchState( pcf::IndiElement::Off );
                }
            }

            ipSend[it->second.name].setSwitchState( pcf::IndiElement::On );
            sendNewProperty( ipSend );
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_drawMutex );
            wclear( w_interactWin );
            wrefresh( w_interactWin );
        }

        break;
    } //case 'p'
    default:
        return; //break;
    }

    std::lock_guard<std::mutex> lock( m_drawMutex );
    cursStat( 0 );
    wrefresh( w_interactWin );
}

inline int cursesINDI::postDraw()
{
    //if(fpout) *fpout << "post draw" << std::endl;

    if( w_countWin )
    {
        wclear( w_countWin );
        delwin( w_countWin );
    }
    w_countWin = newwin( 1, m_minWidth, m_yTop + tabHeight() + 1, m_xLeft );

    return postPrint();
}

inline int cursesINDI::postPrint()
{
    if( !w_countWin )
    {
        return 0;
    }

    int shown = tabHeight();
    if( m_cellContents.size() - m_startRow < (size_t)shown )
    {
        shown = m_cellContents.size() - m_startRow;
    }

    wclear( w_countWin );
    wprintw( w_countWin, "%i/%zu elements shown.", shown, knownElements.size() );
    wrefresh( w_countWin );

    return 0;
}
