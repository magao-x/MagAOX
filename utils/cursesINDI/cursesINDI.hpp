
#include "../../INDI/libcommon/IndiClient.hpp"
#include "cursesTable.hpp"


class cursesINDI : public pcf::IndiClient, public cursesTable
{

public:
   std::vector<int> m_cx {0, 5, 20, 35, 50};

   size_t m_currY {0};
   size_t m_currX {1};

   int m_redraw {0};

   int m_update {0};

   int m_cursStat {1};

   WINDOW * w_interactWin {nullptr};
   WINDOW * w_countWin
   {nullptr};

   bool m_shutdown {false};
   bool m_connectionLost{false};

   std::thread m_drawThread;
   std::mutex m_drawMutex;

   cursesINDI( const std::string &szName,
               const std::string &szVersion,
               const std::string &szProtocolVersion
             );

   ~cursesINDI();

   typedef std::map< std::string, pcf::IndiProperty> propMapT;
   typedef propMapT::value_type propMapValueT;
   typedef propMapT::iterator propMapIteratorT;

   propMapT knownProps;

   struct elementSpec
   {
      std::string propKey;
      std::string name;
      int tableRow {-1};
   };

   typedef std::map< std::string, elementSpec> elementMapT;
   typedef elementMapT::value_type elementMapValueT;
   typedef elementMapT::iterator elementMapIteratorT;

   elementMapT knownElements;

   virtual void handleDefProperty( const pcf::IndiProperty &ipRecv );

   virtual void handleDelProperty( const pcf::IndiProperty &ipRecv );

   virtual void handleMessage( const pcf::IndiProperty &ipRecv );

   virtual void handleSetProperty( const pcf::IndiProperty &ipRecv );

   virtual void execute();

   void cursStat(int cs);

   int cursStat();

   void startUp();

   void shutDown();

   static void _drawThreadStart( cursesINDI * c /**< [in]  */);

   /// Start the draw thread.
   int drawThreadStart();

   /// Execute the draw thread.
   void drawThreadExec();


   void redrawTable();

   void updateTable();

   void moveCurrent( int nextY,
                     int nextX
                   );

   void _moveCurrent( int nextY,
                     int nextX
                   );

   void keyPressed( int ch );

   

   

};

cursesINDI::cursesINDI( const std::string &szName,
                        const std::string &szVersion,
                        const std::string &szProtocolVersion
                     ) : pcf::IndiClient(szName, szVersion, szProtocolVersion, "127.0.0.1", 7624)
{
   m_tabY = 5;
   m_tabX = 0;
   m_tabHeight = LINES-5;
   m_tabWidth = COLS;

}

cursesINDI::~cursesINDI()
{
   if( w_interactWin ) delwin(w_interactWin);
}

void cursesINDI::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
   if(!ipRecv.hasValidDevice() && !ipRecv.hasValidName())
   {
      return;
   }

   std::pair<propMapIteratorT, bool> result;

   std::lock_guard<std::mutex> lock(m_drawMutex);

   result = knownProps.insert(propMapValueT( ipRecv.createUniqueKey(), ipRecv ));


   if(result.second == false)
   {
      result.first->second = ipRecv;
   }
   else
   {
      sendGetProperties(ipRecv);
   }

   auto elIt = ipRecv.getElements().begin();

   while(elIt != ipRecv.getElements().end())
   {
      elementSpec es;
      es.propKey = ipRecv.createUniqueKey();
      es.name = elIt->second.getName();

      std::string key = es.propKey + "." + es.name;
      auto elResult = knownElements.insert( elementMapValueT(key, es));

      if(elResult.second == true)
      {
         //If result is new insert, add to TUI table if filter requires
         ++m_redraw;
      }
      else
      {
         ++m_update;
      }
      ++elIt;
   }

}

void cursesINDI::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
   static_cast<void>(ipRecv);
}

void cursesINDI::handleMessage( const pcf::IndiProperty &ipRecv )
{
   static_cast<void>(ipRecv);
}

void cursesINDI::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
   handleDefProperty(ipRecv);
}

void cursesINDI::execute()
{
   processIndiRequests(false);
}

void cursesINDI::cursStat(int cs)
{
   m_cursStat = cs;
   curs_set(m_cursStat);

}

int cursesINDI::cursStat()
{
   return m_cursStat;
}

void cursesINDI::startUp()
{
   if(w_interactWin == nullptr)
   {
      w_interactWin = newwin( 1, m_tabWidth, m_tabY-2, m_tabX);
   }

   keypad(w_interactWin, TRUE);

   if(w_countWin == nullptr)
   {
      w_countWin = newwin( 1, m_tabWidth, m_tabY+m_tabHeight+1, m_tabX);
   }

   wprintw(w_countWin, "elements shown.");
   wrefresh(w_countWin);

   m_shutdown = false;
   drawThreadStart();
}

void cursesINDI::shutDown()
{
   m_shutdown = true;

   quitProcess();
   deactivate();

   m_drawThread.join();

   rows.clear();

   if(w_interactWin) delwin(w_interactWin);
   w_interactWin = nullptr;

   if(w_countWin) delwin(w_countWin);
   w_countWin = nullptr;

}

inline
void cursesINDI::_drawThreadStart( cursesINDI * c)
{
   c->drawThreadExec();
}

inline
int cursesINDI::drawThreadStart()
{
   try
   {
      m_drawThread = std::thread( _drawThreadStart, this);
   }
   catch( const std::exception & e )
   {
      return -1;
   }
   catch( ... )
   {
      return -1;
   }

   if(!m_drawThread.joinable())
   {
      return -1;
   }

   return 0;
}

inline
void cursesINDI::drawThreadExec()
{
   while(!m_shutdown && !getQuitProcess())
   {
      if(m_redraw > 0)
      {
         redrawTable();
      }

      if(m_update > 0)
      {
         updateTable();
      }

      std::this_thread::sleep_for( std::chrono::duration<unsigned long, std::nano>(250000000));
   }

   if(getQuitProcess() && !m_shutdown)
   {
      m_connectionLost = true;

      rows.clear();
      redrawTable();
      m_shutdown = true;
   }

}

void cursesINDI::redrawTable()
{
   std::lock_guard<std::mutex> lock(m_drawMutex);

   int start_redraw = m_redraw;

   rows.clear();

   for( elementMapIteratorT es = knownElements.begin(); es != knownElements.end(); ++es)
   {
      es->second.tableRow = addRow(m_cx);

      std::vector<std::string> s;

      s.resize( m_cx.size() );
      s[0] = std::to_string(es->second.tableRow+1);
      s[1] = knownProps[es->second.propKey].getDevice();
      s[2] = knownProps[es->second.propKey].getName();
      s[3] = es->second.name;
      s[4] = knownProps[es->second.propKey][es->second.name].getValue();

      updateContents(es->second.tableRow, s);
   }

   m_redraw -= start_redraw;
   if(m_redraw <0) m_redraw = 0;


   wclear(w_countWin);

   int shown = m_tabHeight;
   if(rows.size() < (size_t) m_tabHeight ) shown = rows.size();
   wprintw(w_countWin, "%i/%i elements shown.", shown, knownElements.size());
   wrefresh(w_countWin);

   _moveCurrent(m_currY, m_currX);
}

void cursesINDI::updateTable()
{
   if(m_redraw) return; //Pending redraw, so we skip it and let that take care of it.

   std::lock_guard<std::mutex> lock(m_drawMutex);

   int start_update = m_update;

   int cx, cy;

   getyx(w_interactWin, cy, cx);
   int cs = cursStat();
   cursStat(0);

   for(auto it = knownElements.begin(); it != knownElements.end(); ++it)
   {
      if(it->second.tableRow == -1) continue;

      updateContents( it->second.tableRow, 4,  knownProps[it->second.propKey][it->second.name].getValue());
   }

   wmove(w_interactWin,cy,cx);
   cursStat(cs);
   wrefresh(w_interactWin);

   m_update -= start_update;
   if(m_update <0) m_update = 0;
}

void cursesINDI::moveCurrent( int nextY,
                              int nextX
                            )
{
   std::lock_guard<std::mutex> lock(m_drawMutex);
   _moveCurrent(nextY, nextX);
}

void cursesINDI::_moveCurrent( int nextY,
                               int nextX
                             )
{
   //Do some bounds checks
   if(rows.size() == 0 || m_currY >= rows.size()) return;
   if(m_currX >= rows[m_currY].m_cellWin.size()) return;

   //Now turn off the reverse
   wattroff(rows[m_currY].m_cellWin[m_currX], A_REVERSE);
   rows[m_currY].updateContents( m_currX, rows[m_currY].m_cellContents[m_currX], true);

   //Move the cursor position
   if(nextY >= 0 && (size_t) nextY < rows.size() && nextX >= 1 && nextX <= 4)
   {
      m_currY = nextY;
      m_currX = nextX;
   }

   //Turn it back on
   wattron(rows[m_currY].m_cellWin[m_currX], A_REVERSE);
   rows[m_currY].updateContents( m_currX, rows[m_currY].m_cellContents[m_currX], true);

   wattroff(rows[m_currY].m_cellWin[m_currX], A_REVERSE);
}

void cursesINDI::keyPressed( int ch )
{

   switch(ch)
   {
      case 'e':
      {
         auto it = knownElements.begin();
         while(it != knownElements.end())
         {
            if( (size_t) it->second.tableRow == m_currY) break;
            ++it;
         }
         //Error checks?

         cursStat(1);

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wprintw(w_interactWin, "set: %s.%s=", it->second.propKey.c_str(), it->second.name.c_str());
            wrefresh(w_interactWin);
         }

         bool escape = false;
         std::string newStr;
         int nch;
         while( (nch = wgetch(w_interactWin)) != '\n')
         {
            if(nch == ERR)
            {
               if( getQuitProcess())
               {
                  //If the IndiConnection has set 'quitProces' but no other shutdown
                  //has been issued then we record this as a lost connection.
                  if(!m_shutdown) m_connectionLost = true;
                  break;
               }
               else continue;
            }

            cursStat(1);

            if(nch == 27)
            {
               std::lock_guard<std::mutex> lock(m_drawMutex);
               wclear(w_interactWin);
               wrefresh(w_interactWin);
               escape = true;
               break;
            }
            if( nch == KEY_BACKSPACE )
            {
               if(newStr.size() > 0)
               {
                  std::lock_guard<std::mutex> lock(m_drawMutex);
                  newStr.erase(newStr.size()-1,1);
                  wprintw(w_interactWin, "\b \b");
                  wrefresh(w_interactWin);
               }
            }
            else if (std::isprint(nch))
            {
               std::lock_guard<std::mutex> lock(m_drawMutex);
               wprintw(w_interactWin, "%c", nch);
               wrefresh(w_interactWin);

               newStr += (char) nch;
            }
         }
         if(escape) break;

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wprintw(w_interactWin, "send: %s.%s=%s? y/n [n]", it->second.propKey.c_str(), it->second.name.c_str(), newStr.c_str());
            wrefresh(w_interactWin);
         }

         nch = 0;
         while( (nch = wgetch(w_interactWin)) == ERR)
         {
         }

         if(nch == 'y')
         {
            pcf::IndiProperty ipSend(knownProps[it->second.propKey].getType());

            ipSend.setDevice(knownProps[it->second.propKey].getDevice());
            ipSend.setName(knownProps[it->second.propKey].getName());
            ipSend.add(pcf::IndiElement(it->second.name));
            ipSend[it->second.name].setValue(newStr);
            sendNewProperty(ipSend);
         }

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wrefresh(w_interactWin);
         }

         break;
      }
      default:
         return;//break;
   }

   std::lock_guard<std::mutex> lock(m_drawMutex);
   cursStat(0);
   wrefresh(w_interactWin);

}
