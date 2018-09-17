
#include "cursesINDI.hpp"



#include <fstream>

int main()
{
   cursesINDI ci("me", "1.7", "1.7");

   WINDOW * topWin;

   int ch;

   initscr();
   
   raw();    /* Line buffering disabled*/
   halfdelay(10); //We use a 1 second timeout to check for connection loss.
   keypad(stdscr, TRUE); /* We get F1, 2 etc...*/

   noecho(); /* Don't echo() while we do getch */

   ci.m_tabHeight = LINES-5-2;
   ci.m_tabX = 1;
   ci.m_tabWidth = 78;

   ci.startUp();
   ci.activate();

   pcf::IndiProperty ipSend;
   ci.sendGetProperties( ipSend );

   topWin = newwin(1, COLS, 0, 0);
   wprintw(topWin, "(e)dit a property, (q)uit");
   keypad(topWin, TRUE);
   wrefresh(topWin);

   WINDOW * boxWin;
   boxWin = newwin(ci.m_tabHeight+2, ci.m_tabWidth+2, 4,0);
   box(boxWin, 0, 0);
   wrefresh(boxWin);
   
   ci.cursStat(0);

   
   //Now main event loop
   while((ch = wgetch(ci.w_interactWin)) != 'q')
   {
      if(ch == ERR)
      {
         if( ci.getQuitProcess() || ci.m_shutdown) break;
         else continue;
      }
      
      int nextX = ci.m_currX;
      int nextY = ci.m_currY;

      switch(ch)
      {
         case KEY_LEFT:
            --nextX;
            break;
         case KEY_RIGHT:
            ++nextX;
            break;
         case KEY_UP:
            --nextY;
            break;
         case KEY_DOWN:
            ++nextY;
            break;
         default:
            ci.keyPressed(ch);
            continue;
            break;
      }

      int maxX = ci.m_cx.size()-1;
      if(nextX < 1) nextX = 1;
      if(nextX >= maxX) nextX = maxX;

      int maxY = ci.rows.size();
      if(nextY < 0) nextY = 0;
      if(nextY >= maxY) nextY = maxY - 1;

      if(nextY-ci.m_currFirstRow > ci.m_tabHeight-1)
      {
         ci.updateRowY(nextY - ci.m_tabHeight + 1);
      }
      else if( nextY < ci.m_currFirstRow)
      {
         ci.updateRowY(nextY);
      }

      ci.moveCurrent(nextY, nextX);

      ci.cursStat(0);
      
      if(ci.m_shutdown) break;
   }

   ci.shutDown();

   endwin();   /* End curses mode */

   if(ci.m_connectionLost)
   {
      std::cerr << "\ncursesINDI: lost connection to indiserver.\n\n";
   }
   
   return 0;


}
