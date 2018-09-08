
#include "../../INDI/libcommon/IndiClient.hpp"
#include "cursesTable.hpp"

class cursesINDI : public pcf::IndiClient, public cursesTable
{

public:
   std::vector<int> m_cx {0, 5, 20, 35, 50};

   int m_currY {0};
   int m_currX {1};

   int m_redraw {0};

   int m_cursStat {1};

   WINDOW * w_interactWin {nullptr};
   WINDOW * w_countWin {nullptr};

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

   void redrawTable();

   void tableUpdateElement( const std::string & esKey );

   void moveCurrent( int nextY,
                     int nextX
                   );

   void keyPressed( int ch );

};

cursesINDI::cursesINDI( const std::string &szName,
                        const std::string &szVersion,
                        const std::string &szProtocolVersion
                      ) : pcf::IndiClient(szName, szVersion, szProtocolVersion)
{
   m_tabY = 5;
   m_tabX = 0;
   m_tabHeight = LINES-5;
   m_tabWidth = COLS;
   
   setInterval(250);

}

cursesINDI::~cursesINDI()
{
   if( w_interactWin ) delwin(w_interactWin);
}

void cursesINDI::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
   if(!ipRecv.hasValidDevice() && !ipRecv.hasValidName()) return;

   std::pair<propMapIteratorT, bool> result;
   
   //std::lock_guard<std::mutex> lock(m_drawMutex); 
   
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
         //Otherwise we just update the element.
         tableUpdateElement(elResult.first->first);
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
   if(m_redraw > 0)
   {
      redrawTable();
   }
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
}

void cursesINDI::shutDown()
{
   quitProcess();
   deactivate();

   rows.clear();

   if(w_interactWin) delwin(w_interactWin);
   w_interactWin = nullptr;

   if(w_countWin) delwin(w_countWin);
   w_countWin = nullptr;

}

void cursesINDI::redrawTable()
{
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

   moveCurrent(m_currY, m_currX);


}

void cursesINDI::tableUpdateElement( const std::string & esKey )
{
   if(m_redraw) return; //Pending redraw, so we skip it and let that take care of it.
   
   elementSpec es = knownElements[esKey];
   
   if(es.tableRow == -1) return;

   int cx, cy;

   //std::lock_guard<std::mutex> lock(m_drawMutex); 
   
   getyx(w_interactWin, cy, cx);
   int cs = cursStat();
   cursStat(0);
   updateContents( es.tableRow, 4, knownProps[es.propKey][es.name].getValue());
   wmove(w_interactWin,cy,cx);
   cursStat(cs);
   wrefresh(w_interactWin);

}

void cursesINDI::moveCurrent( int nextY,
                              int nextX
                            )
{
   //std::lock_guard<std::mutex> lock(m_drawMutex); 

   wattroff(rows[m_currY].m_cellWin[m_currX], A_REVERSE);
   rows[m_currY].updateContents( m_currX, rows[m_currY].m_cellContents[m_currX], true);

   if(nextY >= 0 && nextY < m_tabHeight && nextX >= 1 && nextX <= 4)
   {
      m_currY = nextY;
      m_currX = nextX;
   }

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
            if(it->second.tableRow == m_currY) break;
            ++it;
         }
         //Error checks?
   
         cursStat(1);
         {
            //std::lock_guard<std::mutex> lock(m_drawMutex); 
         wclear(w_interactWin);
         wprintw(w_interactWin, "set: %s.%s=", it->second.propKey.c_str(), it->second.name.c_str());
         wrefresh(w_interactWin);
         }
         
         bool escape = false;
         std::string newStr;
         int nch;
         while( (nch = wgetch(w_interactWin)) != '\n')
         {
            cursStat(1);
   
            if(nch == 27)
            {
               //std::lock_guard<std::mutex> lock(m_drawMutex); 
               wclear(w_interactWin);
               wrefresh(w_interactWin);
               escape = true;
               break;
            }
            if( nch == KEY_BACKSPACE )
            {
               if(newStr.size() > 0)
               {
                  //std::lock_guard<std::mutex> lock(m_drawMutex); 
                  newStr.erase(newStr.size()-1,1);
                  wprintw(w_interactWin, "\b \b");
                  wrefresh(w_interactWin);
               }
            }
            else if (std::isprint(nch))
            {
               //std::lock_guard<std::mutex> lock(m_drawMutex); 
               wprintw(w_interactWin, "%c", nch);
               wrefresh(w_interactWin);
   
               newStr += (char) nch;
            }
         }
         if(escape) break;
         
         {
            //std::lock_guard<std::mutex> lock(m_drawMutex); 
         wclear(w_interactWin);
         wprintw(w_interactWin, "send: %s.%s=%s? y/n [n]", it->second.propKey.c_str(), it->second.name.c_str(), newStr.c_str());
         wrefresh(w_interactWin);
         }
         
         nch = 0;
         while( nch == 0 )
         {
            nch = wgetch(w_interactWin);
            if(nch == 'y' || nch == 'n' || nch == '\n') break;
         }
   
         if(nch == 'y')
         {
            pcf::IndiProperty ipSend;
            ipSend = knownProps[it->second.propKey];
            ipSend[it->second.name].setValue(newStr);
            sendNewProperty(ipSend);
         }
      
         {
         //std::lock_guard<std::mutex> lock(m_drawMutex); 
         wclear(w_interactWin);
         wrefresh(w_interactWin);
         }
         break;
      }
      default:
         return;//break;
   }
   
   {
      //std::lock_guard<std::mutex> lock(m_drawMutex); 
   cursStat(0);
   wrefresh(w_interactWin);

   }



}
