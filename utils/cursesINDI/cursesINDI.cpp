
//#define DEBUG_TMPOUT



#include "cursesINDI.hpp"





int main()
{
   cursesINDI ci("me", "1.7", "1.7");

   std::ofstream *fpout {nullptr};
   
   #ifdef DEBUG_TMPOUT
   std::string fname = "/tmp/cursesINDI.txt";
   fpout = new std::ofstream;
   fpout->open(fname);
   
   ci.fpout = fpout;
   #endif
   
   
   WINDOW * topWin;

   int ch;

   initscr();
   
   raw();    /* Line buffering disabled*/
   halfdelay(10); //We use a 1 second timeout to check for connection loss.
   keypad(stdscr, TRUE); /* We get F1, 2 etc...*/

   noecho(); /* Don't echo() while we do getch */


   ci.startUp();
   ci.activate();

   pcf::IndiProperty ipSend;
   ci.sendGetProperties( ipSend );

   topWin = newwin(1, COLS, 0, 0);
   wprintw(topWin, "(e)dit a property, (q)uit");
   keypad(topWin, TRUE);
   wrefresh(topWin);

   
   ci.cursStat(0);
   
   
   //Now main event loop
   while((ch = wgetch(ci.w_interactWin)) != 'q')
   {
      if(ch == ERR)
      {
         //if(fpout) *fpout << "loop" << std::endl;
         if( ci.getQuitProcess() || ci.m_shutdown) break;
         else continue;
      }
      
      //Get hold downs
      int ch0 = ch;
      int npress = 1;
      
      nocbreak();
      wtimeout(ci.w_interactWin, 50);
      ch = wgetch(ci.w_interactWin);
      while(ch == ch0) 
      {
         ch = wgetch(ci.w_interactWin);
         ++npress; 
      }
      if(ch != ERR) ungetch(ch);
      halfdelay(10);   
      
      int nextX = ci.m_currX;
      int nextY = ci.m_currY;

      switch(ch0)
      {
         case KEY_LEFT:
            nextX -= npress;
            if(fpout) *fpout << "left: " << npress << std::endl;
            break;
         case KEY_RIGHT:
            nextX += npress;
            if(fpout) *fpout << "right: " << npress << std::endl;
            break;
         case KEY_UP:
            nextY -= npress;
            if(fpout) *fpout << "up: " << npress << std::endl;
            break;
         case KEY_DOWN:
            nextY += npress;
            if(fpout) *fpout << "down: " << npress << std::endl;
            break;
         case KEY_PPAGE:
            nextY -= ci.m_gridWin.size()-1;
            if(fpout) *fpout << "ppage: " << npress << std::endl;
            break;   
         case KEY_NPAGE:
            nextY += ci.m_gridWin.size()-1;
            if(fpout) *fpout << "npage: " << npress << std::endl;
            break;
         case KEY_RESIZE:
            ci.draw();
            if(fpout) *fpout << "resizes: " << npress << std::endl;
            continue;
         default:
            if(fpout) *fpout << "other: " << npress << std::endl;
            ci.keyPressed(ch0);
            continue;
            break;
      }

      int maxX = ci.m_gridWin[0].size()-1;
      if(nextX < 1) nextX = 1;
      if(nextX >= maxX) nextX = maxX;

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
