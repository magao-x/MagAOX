
#include <fstream>

#include "../../INDI/libcommon/IndiClient.hpp"
#include "cursesTableGrid.hpp"


///Simple utility to get the display string of a property
/**
  * \returns the properly formatted element value
  * \returns empty string on error
  */ 
std::string displayProperty( pcf::IndiProperty & ip /**< [in] the INDI property */ )
{
   std::string str = ip.getName();
   str += " [";
   
   std::string tstr = "n";
   if(ip.getType() == pcf::IndiProperty::Text) tstr = "t";
   if(ip.getType() == pcf::IndiProperty::Switch) tstr = "s";
   if(ip.getType() == pcf::IndiProperty::Light) tstr = "l";
   
   str += tstr;
   
   str += "]";
   
   
   if(ip.getState() == pcf::IndiProperty::Idle) str += "~";
   if(ip.getState() == pcf::IndiProperty::Ok) str += "-";
   if(ip.getState() == pcf::IndiProperty::Busy) str += "*";
   if(ip.getState() == pcf::IndiProperty::Alert) str += "!";
   
   return str;
}

///Simple utility to get the display value of an element
/**
  * \returns the properly formatted element value
  * \returns empty string if element is not in property
  */ 
std::string displayValue( pcf::IndiProperty & ip, ///< [in] the INDI property
                          std::string & el ///< [in] the name of the element
                        )
{
   if(!ip.find(el)) return "";
   
   if(ip.getType() == pcf::IndiProperty::Switch)
   {
      if( ip[el].switchState() == pcf::IndiElement::SwitchState::Off ) return "|O|";
      if( ip[el].switchState() == pcf::IndiElement::SwitchState::On ) return "|X|";
      return pcf::IndiElement::getSwitchStateString(ip[el].switchState());
   }
   else
   {
      return ip[el].value();
   }
}

class cursesINDI : public pcf::IndiClient, public cursesTableGrid
{

public:

   int m_redraw {0};

   int m_update {0};

   int m_cursStat {1};

   WINDOW * w_interactWin {nullptr};
   WINDOW * w_curvalWin {nullptr};
   WINDOW * w_attentionWin {nullptr};
   
   WINDOW * w_countWin {nullptr};

   bool m_shutdown {false};
   bool m_connectionLost{false};

   std::thread m_drawThread;
   std::mutex m_drawMutex;

   std::string m_msgFile {"/tmp/cursesINDI_logs.txt"};
   std::ofstream m_msgout;
   int m_msgsPrinted {0};
   int m_msgsMax {10000};
   
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
      std::string device;
      std::string propertyName;
      std::string name;
      int tableRow {-1};
   };

   typedef std::map< std::string, elementSpec> elementMapT;
   typedef elementMapT::value_type elementMapValueT;
   typedef elementMapT::iterator elementMapIteratorT;

   elementMapT knownElements;

   
   bool m_deviceSearching {false};
   std::string m_deviceTarget;
   
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

   /// Update the current-value window.
   void updateCurVal();
   
   void moveCurrent( int nextY,
                     int nextX
                   );

   void _moveCurrent( int nextY,
                      int nextX
                    );

   void keyPressed( int ch );

   /// If a key is pressed in column 1 (the device column), this function searchs for devices alphabetically.
   void deviceSearch( int ch );
   
   virtual int postDraw()
   {
      //if(fpout) *fpout << "post draw" << std::endl;
      
      if(w_countWin) 
      {
         wclear(w_countWin);
         delwin(w_countWin);
      }
      w_countWin = newwin( 1, m_minWidth, m_yTop+tabHeight()+1, m_xLeft);
      
      return postPrint();
   }

   virtual int postPrint()
   {
      if(! w_countWin) return 0;
      
      int shown = tabHeight();
      if( m_cellContents.size() - m_startRow <  (size_t) shown ) shown = m_cellContents.size() - m_startRow;
      
      wclear(w_countWin);
      wprintw(w_countWin, "%i/%zu elements shown.", shown, knownElements.size());
      wrefresh(w_countWin);
      
      return 0;
   }
   

};

cursesINDI::cursesINDI( const std::string &szName,
                        const std::string &szVersion,
                        const std::string &szProtocolVersion
                      ) : pcf::IndiClient(szName, szVersion, szProtocolVersion, "127.0.0.1", 7624)
{
   m_yTop = 6;
   colWidth({4, 19, 18, 18, 18});
   
   m_yBot = 1;
   

   m_msgout.open(m_msgFile);

}

cursesINDI::~cursesINDI()
{
   if( w_interactWin ) delwin(w_interactWin);
   if( w_curvalWin ) delwin(w_curvalWin);
   if( w_attentionWin ) delwin(w_attentionWin);
   if( w_countWin) delwin(w_countWin);
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
      result.first->second = ipRecv; //We already have it, so we're already registered
   }
   else
   {
      sendGetProperties(ipRecv); //Otherwise register for it
   }

   auto elIt = ipRecv.getElements().begin();

   while(elIt != ipRecv.getElements().end())
   {
      elementSpec es;
      es.propKey = ipRecv.createUniqueKey();
      es.device = ipRecv.getDevice();
      es.propertyName = ipRecv.getName();
      es.name = elIt->second.name();

      std::string key = es.propKey + "." + es.name;
      auto elResult = knownElements.insert( elementMapValueT(key, es));

      if(elResult.second == true)
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
   //if(fpout) *fpout << "got delete property" << std::endl;
   
   if(ipRecv.hasValidDevice())
   {
      if(!ipRecv.hasValidName())
      {
         //if(fpout) *fpout << "will delete: " << ipRecv.getDevice() << "\n";
         
         for(elementMapIteratorT elIt = knownElements.begin(); elIt != knownElements.end();)
         {
            if( elIt->second.device == ipRecv.getDevice()) elIt = knownElements.erase(elIt);
            else ++elIt;
         }
         
         for(propMapIteratorT pIt = knownProps.begin(); pIt != knownProps.end();)
         {
            if( pIt->first == ipRecv.createUniqueKey() ) pIt = knownProps.erase(pIt);
            else ++pIt;
         }
      }
      else
      {
         //if(fpout) *fpout << "will delete: " << ipRecv.createUniqueKey() << "\n";
         
         for(elementMapIteratorT elIt = knownElements.begin(); elIt != knownElements.end();)
         {
            if( elIt->second.propKey == ipRecv.createUniqueKey()) elIt = knownElements.erase(elIt);
            else ++elIt;
         }
         
         knownProps.erase(ipRecv.createUniqueKey());
         
      }
   }
   
   ++m_redraw;
   
}

void cursesINDI::handleMessage( const pcf::IndiProperty &ipRecv )
{
   tm bdt; //broken down time
   time_t tt = ipRecv.getTimeStamp().getTimeVal().tv_sec; 
   gmtime_r( &tt, &bdt);
   
   char tstr1[25];
   strftime(tstr1, sizeof(tstr1), "%H:%M:%S", &bdt);
   char tstr2[11];
   snprintf(tstr2, sizeof(tstr2), ".%06i", static_cast<int>(ipRecv.getTimeStamp().getTimeVal().tv_usec)); //casting in case we switch to int64_t
         
   
   std::string msg = ipRecv.getMessage();
   
   if(msg.size() > 4)
   {
      std::string prio = msg.substr(0,4);
      if(prio == "NOTE")
      {
         m_msgout << "\033[1m";
      }
      else if(prio == "WARN")
      {
         m_msgout << "\033[93m\033[1m";
      }
      else if(prio == "ERR ")
      {
         m_msgout << "\033[91m\033[1m";
      }
      else if(prio == "CRIT")
      {
         m_msgout << "\033[41m\033[1m";
      }
      else if(prio == "ALRT")
      {
         m_msgout << "\033[41m\033[1m";
      }
      else if(prio == "EMER")
      {
         m_msgout << "\033[41m\033[1m";
      }
   }   
   m_msgout << std::string(tstr1) << std::string(tstr2) << " [" << ipRecv.getDevice() << "] " << msg;
   
   m_msgout << "\033[0m";
   m_msgout << std::endl;
   
   
   ++m_msgsPrinted;
   if(m_msgsPrinted > m_msgsMax)
   {
      m_msgout.close();
      m_msgout.open(m_msgFile);
      m_msgsPrinted = 0;
   }
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
      w_interactWin = newwin( 1, m_minWidth, m_yTop-3, m_xLeft);
   }

   if(w_curvalWin == nullptr)
   {
      w_curvalWin = newwin( 1, m_minWidth, m_yTop-2, m_xLeft);
   }
   
   if(w_attentionWin == nullptr)
   {
      w_attentionWin = newwin( 1, m_minWidth, m_yTop-4, m_xLeft);
   }
   
   keypad(w_interactWin, TRUE);


   m_shutdown = false;
   drawThreadStart();
}

void cursesINDI::shutDown()
{
   if(getQuitProcess() && !m_shutdown) m_connectionLost = true;
   
   m_shutdown = true;

   quitProcess();
   deactivate();

   m_drawThread.join();

   m_cellContents.clear();

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
      ////if(fpout) *fpout << "draw thread . . ." << std::endl;
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

      m_cellContents.clear();
      redrawTable();
      m_shutdown = true;
   }

}

void cursesINDI::redrawTable()
{
   std::lock_guard<std::mutex> lock(m_drawMutex);

   int start_redraw = m_redraw;

   //if(fpout) *fpout << "redrawTable: " << m_redraw << std::endl; 
   
   m_cellContents.clear();

   bool fsmAlerts = false;
   
   std::set<std::string> alertDev;
   
   for( elementMapIteratorT es = knownElements.begin(); es != knownElements.end(); ++es)
   {
      //if(fpout) *fpout << knownProps[es->second.propKey].getName() << " " << knownProps[es->second.propKey].getState() << "\n";
      
      if(knownProps[es->second.propKey].getName() == "fsm" &&
            knownProps[es->second.propKey].getState() == pcf::IndiProperty::Alert) 
      {
         fsmAlerts = true;
         alertDev.insert(knownProps[es->second.propKey].getDevice());
      }
      std::vector<std::string> s;

      s.resize( m_colFraction.size() );
      
      s[0] = std::to_string(m_cellContents.size()+1);
      s[1] = knownProps[es->second.propKey].getDevice();
      s[2] = displayProperty( knownProps[es->second.propKey] );
      
      s[3] = es->second.name;
      
      s[4] = displayValue( knownProps[es->second.propKey], es->second.name);
      
      m_cellContents.push_back(s);
      es->second.tableRow = m_cellContents.size()-1;
   }

   draw();
   
   //if(fpout) *fpout << "fsmAlerts: " << fsmAlerts << "\n";
   
   int cx, cy;
   getyx(w_interactWin, cy, cx);
   int cs = cursStat();
   cursStat(0);
   
   wclear(w_attentionWin);
   if(fsmAlerts)
   {
      std::string alrt = "!! FSM alert: " + *alertDev.begin();
      if(alertDev.size() > 1) alrt += " (+" + std::to_string(alertDev.size()-1) + ")";
      
      wprintw(w_attentionWin, "%s", alrt.c_str());
   }
   wrefresh(w_attentionWin);
   
   wmove(w_interactWin,cy,cx);
   cursStat(cs);
   wrefresh(w_interactWin);
   
   m_redraw -= start_redraw;
   if(m_redraw <0) m_redraw = 0;
   
   _moveCurrent(m_currY, m_currX);
}


   
void cursesINDI::updateTable()
{
   if(m_redraw) return; //Pending redraw, so we skip it and let that take care of it.

   std::lock_guard<std::mutex> lock(m_drawMutex);

   int start_update = m_update;

   //if(fpout) *fpout << "updateTable: " << m_update << std::endl; 
   int cx, cy;

   getyx(w_interactWin, cy, cx);
   int cs = cursStat();
   
   updateCurVal();
   
   bool fsmAlerts {false};
   std::set<std::string> alertDev;
   
   for(auto it = knownElements.begin(); it != knownElements.end(); ++it)
   {
      //if(fpout) *fpout << knownProps[it->second.propKey].getName() << " " << knownProps[it->second.propKey].getState() << "\n";
      
      if(knownProps[it->second.propKey].getName() == "fsm" &&
            knownProps[it->second.propKey].getState() == pcf::IndiProperty::Alert) 
      {
         fsmAlerts = true;
         alertDev.insert(knownProps[it->second.propKey].getDevice());
      }
      
      if(it->second.tableRow == -1) continue;
      
      if(m_cellContents[it->second.tableRow][2] != displayProperty(knownProps[it->second.propKey]) )
      {
         m_cellContents[it->second.tableRow][2] = displayProperty(knownProps[it->second.propKey]) ;
         
         if(it->second.tableRow - m_startRow < (size_t) tabHeight()) //It's currently displayed
         {
            cursStat(0);
            wclear(m_gridWin[it->second.tableRow - m_startRow][2]);
            if(hasContent(it->second.tableRow,2)) wprintw(m_gridWin[it->second.tableRow - m_startRow][2], "%s", m_cellContents[it->second.tableRow][2].c_str());
            wrefresh(m_gridWin[it->second.tableRow - m_startRow][2]);
            wmove(w_interactWin,cy,cx);
            cursStat(cs);
            wrefresh(w_interactWin);
         }
      }
      
      if(m_cellContents[it->second.tableRow][4] != displayValue(knownProps[it->second.propKey], it->second.name)) //.value())
      {
         m_cellContents[it->second.tableRow][4] = displayValue(knownProps[it->second.propKey], it->second.name); //knownProps[it->second.propKey][it->second.name].value();
         
         if(it->second.tableRow - m_startRow < (size_t) tabHeight()) //It's currently displayed
         {
            cursStat(0);
            wclear(m_gridWin[it->second.tableRow - m_startRow][4]);
            if(hasContent(it->second.tableRow,4)) wprintw(m_gridWin[it->second.tableRow - m_startRow][4], "%s", m_cellContents[it->second.tableRow][4].c_str());
            wrefresh(m_gridWin[it->second.tableRow - m_startRow][4]);
            wmove(w_interactWin,cy,cx);
            cursStat(cs);
            wrefresh(w_interactWin);
         }
         
      };
      //updateContents( it->second.tableRow, 4,  knownProps[it->second.propKey][it->second.name].value());
   }

   wattron(m_gridWin[m_currY][m_currX], A_REVERSE);

   //if(fpout) *fpout << "fsmAlerts: " << fsmAlerts << "\n";
   
   getyx(w_interactWin, cy, cx);
   cs = cursStat();
   cursStat(0);
   wclear(w_attentionWin);
   if(fsmAlerts)
   {
      std::string alrt = "!! FSM alert: " + *alertDev.begin();
      if(alertDev.size() > 1) alrt += " (+" + std::to_string(alertDev.size()-1) + ")";
      
      wprintw(w_attentionWin, "%s", alrt.c_str());
   }
   wrefresh(w_attentionWin);
   wmove(w_interactWin,cy,cx);
   cursStat(cs);
   wrefresh(w_interactWin);
            
   //print();
   
//    wmove(w_interactWin,cy,cx);
//    cursStat(cs);
//    wrefresh(w_interactWin);

   m_update -= start_update;
   if(m_update <0) m_update = 0;
}

void cursesINDI::updateCurVal( )
{
   int cx, cy;
   getyx(w_interactWin, cy, cx);
   int cs = cursStat();
   
   cursStat(0);
   wclear(w_curvalWin);
      
   auto it = knownElements.begin();
   while(it != knownElements.end())
   {
      if( (size_t) it->second.tableRow == m_currY+m_startRow) break;
      ++it;
   }

   if(it == knownElements.end())
   {
      wrefresh(w_interactWin);
      cursStat(cs);
      return;
   }

   std::string cval = "> " + it->second.propKey + "." + it->second.name + " = ";
   
   cval += displayValue( knownProps[it->second.propKey], it->second.name);

   wprintw(w_curvalWin, "%s", cval.c_str());
   wrefresh(w_curvalWin);

   wmove(w_interactWin,cy,cx);
   cursStat(cs);
   wrefresh(w_interactWin);
   
   return;
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
   int currX = m_currX;
   int currY = m_currY;
   
   moveSelected(nextY, nextX);
   
   //if(fpout) *fpout << "moved: " << nextX << " " << nextY << " " << currX << "->" << m_currX << " " << currY << "->" << m_currY << std::endl; 
   
   if(m_deviceSearching && nextX != 1)
   {
      wclear(w_interactWin);
      wrefresh(w_interactWin);
      m_deviceSearching = false;
   }
      
   updateCurVal();

   if(nextX == 1)
   {
      if( currX != nextX)
      {
         wclear(w_interactWin);
         wprintw(w_interactWin, "search: ");
         wrefresh(w_interactWin);
      }
   }
   else if(currY != nextY || currX == 1)
   {
      wclear(w_interactWin);
      if(m_currY + m_startRow >= knownElements.size()) 
      {
         wrefresh(w_interactWin);
         return;
      }
      
      auto it = knownElements.begin();
      while(it != knownElements.end())
      {
         if( (size_t) it->second.tableRow == m_currY+m_startRow) break;
         ++it;
      }
      
      if(it == knownElements.end())
      {
         wrefresh(w_interactWin);
         return;
      }
      
      if( knownProps[it->second.propKey].getPerm() != pcf::IndiProperty::ReadWrite)
      {
         wrefresh(w_interactWin);
         return;
      }
      
      if( knownProps[it->second.propKey].getType() == pcf::IndiProperty::Text)
      {
         wprintw(w_interactWin, "(e)dit this text");
      }
      else if( knownProps[it->second.propKey].getType() == pcf::IndiProperty::Number)
      {
         wprintw(w_interactWin, "(e)dit this number");
      }
      else if( knownProps[it->second.propKey].getType() == pcf::IndiProperty::Switch)
      {
         wprintw(w_interactWin, "(p)ress or (t)oggle this switch");
      }
         

      wrefresh(w_interactWin);
   }
   
}


void cursesINDI::deviceSearch( int ch )
{
   bool updated = false;
   if( m_deviceSearching == true )
   {
      if( ch == KEY_BACKSPACE )
      {
         if(m_deviceTarget.size() > 0)
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            m_deviceTarget.erase(m_deviceTarget.size()-1,1);
            wprintw(w_interactWin, "\b \b");
            wrefresh(w_interactWin);
            return;
         }
      }
      else if (std::isprint(ch))
      {
         m_deviceTarget += ch;
         updated = true;
      }
      else return;
   }
   else if (std::isprint(ch))
   {
      m_deviceTarget = ch;
      updated = true;
   }
   else return;
   
   
   m_deviceSearching = true;
   
   if(updated)
   {
      std::lock_guard<std::mutex> lock(m_drawMutex);
      wprintw(w_interactWin, "%c", ch);
      wrefresh(w_interactWin);
   }
      
   //if(fpout) *fpout << "device searching: " << m_deviceTarget << std::endl; 
   if(m_deviceTarget.size() == 0) return;
   
   auto it = knownElements.lower_bound(m_deviceTarget);

   //if(fpout) *fpout << "new row: " << it->second.tableRow << " " << m_startRow << " " << it->second.tableRow-m_startRow << "\n";
   
   if(it->second.tableRow == -1) return;
   
   m_startRow = it->second.tableRow;
   
   moveCurrent( 0, 1);
   
   redrawTable();
   
   return;
            
}

void cursesINDI::keyPressed( int ch )
{
   
   //If in first column, do device selection
   if(m_currX == 1 )
   {
      deviceSearch(ch);
      return;
   }

   if(m_deviceSearching)
   {
      wclear(w_interactWin);
      wrefresh(w_interactWin);
      m_deviceSearching = false;
   }
   
   switch(ch)
   {
      case 'e':
      {
         if(m_currY + m_startRow >= knownElements.size()) break;
         auto it = knownElements.begin();
         while(it != knownElements.end())
         {
            if( (size_t) it->second.tableRow == m_currY+m_startRow) break;
            ++it;
         }
         
         if(it == knownElements.end()) break;

         //Can't edit a switch
         if( knownProps[it->second.propKey].getType() != pcf::IndiProperty::Text && knownProps[it->second.propKey].getType() != pcf::IndiProperty::Number) break;
         
         cursStat(1);

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wprintw(w_interactWin, "set %s.%s=", it->second.propKey.c_str(), it->second.name.c_str());
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
            wprintw(w_interactWin, "send %s.%s=%s? y/n [n]", it->second.propKey.c_str(), it->second.name.c_str(), newStr.c_str());
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
            //if(fpout) *fpout << "newStr: " << newStr << std::endl; 
   
            ipSend[it->second.name].value(newStr);
            sendNewProperty(ipSend);
         }

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wrefresh(w_interactWin);
         }

         break;
      } //case 'e'
      case 't':
      {
         if(m_currY + m_startRow >= knownElements.size()) break;
         auto it = knownElements.begin();
         while(it != knownElements.end())
         {
            if( (size_t) it->second.tableRow == m_currY+m_startRow) break;
            ++it;
         }
         
         if(it == knownElements.end()) break;

         if( !knownProps[it->second.propKey].find(it->second.name)) break; //Just a check.
         
         if( knownProps[it->second.propKey].getType() != pcf::IndiProperty::Switch) break;
         
         std::string toggleString;
         pcf::IndiElement::SwitchState toggleState;
         if( knownProps[it->second.propKey][it->second.name].switchState() == pcf::IndiElement::SwitchState::Off  )
         {
            toggleString = "On";
            toggleState = pcf::IndiElement::SwitchState::On;
         }
         else if(knownProps[it->second.propKey][it->second.name].switchState() == pcf::IndiElement::SwitchState::On)
         {
            toggleString = "Off";
            toggleState = pcf::IndiElement::SwitchState::Off;
         }
         else break; //would happen fo state unknown
         
         cursStat(1);

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wprintw(w_interactWin, "toggle %s.%s to %s?", it->second.propKey.c_str(), it->second.name.c_str(), toggleString.c_str());
            wrefresh(w_interactWin);
         }

         int nch = 0;
         while( (nch = wgetch(w_interactWin)) == ERR)
         {
         }

         if(nch == 'y')
         {
            pcf::IndiProperty ipSend(knownProps[it->second.propKey].getType());

            ipSend.setDevice(knownProps[it->second.propKey].getDevice());
            ipSend.setName(knownProps[it->second.propKey].getName());
            ipSend.add(pcf::IndiElement(it->second.name));
            ipSend[it->second.name].switchState(toggleState);
            sendNewProperty(ipSend);
         }

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wrefresh(w_interactWin);
         }

         break;
      } //case 't'
      case 'p':
      {
         if(m_currY + m_startRow >= knownElements.size()) break;
         auto it = knownElements.begin();
         while(it != knownElements.end())
         {
            if( (size_t) it->second.tableRow == m_currY+m_startRow) break;
            ++it;
         }
         
         if(it == knownElements.end()) break;

         if( !knownProps[it->second.propKey].find(it->second.name)) break; //Just a check.
                  
         if( knownProps[it->second.propKey].getType() != pcf::IndiProperty::Switch) break;
         cursStat(1);

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wprintw(w_interactWin, "press switch %s.%s?", it->second.propKey.c_str(), it->second.name.c_str());
            wrefresh(w_interactWin);
         }

         int nch = 0;
         while( (nch = wgetch(w_interactWin)) == ERR)
         {
         }

         if(nch == 'y')
         {
            pcf::IndiProperty ipSend(knownProps[it->second.propKey].getType());

            ipSend.setDevice(knownProps[it->second.propKey].getDevice());
            ipSend.setName(knownProps[it->second.propKey].getName());
            
            //Must add all elements
            for(auto elit = knownProps[it->second.propKey].getElements().begin(); elit != knownProps[it->second.propKey].getElements().end(); ++elit)
            {
               ipSend.add(elit->second);
               if( knownProps[it->second.propKey].getRule() != pcf::IndiProperty::AnyOfMany)
               {
                  ipSend[elit->first].switchState(pcf::IndiElement::SwitchState::Off);
               }
            }
            
            ipSend[it->second.name].switchState(pcf::IndiElement::SwitchState::On);
            sendNewProperty(ipSend);
         }

         //mutex scope
         {
            std::lock_guard<std::mutex> lock(m_drawMutex);
            wclear(w_interactWin);
            wrefresh(w_interactWin);
         }

         break;
      } //case 'p'
      default:
         return;//break;
   }

   std::lock_guard<std::mutex> lock(m_drawMutex);
   cursStat(0);
   wrefresh(w_interactWin);

}
