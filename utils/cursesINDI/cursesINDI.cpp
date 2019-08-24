
//#define DEBUG_TMPOUT



#include "cursesINDI.hpp"





int main()
{
   
   cursesINDI * ci;
   
   //This is for debugging
   std::ofstream *fpout {nullptr};
   
   #ifdef DEBUG_TMPOUT
   std::string fname = "/tmp/cursesINDI.txt";
   fpout = new std::ofstream;
   fpout->open(fname);
   
   #endif
   
   

   std::cout << " cursesINDI: Connecting to INDI server . . .                                            \r";

retry:
   bool notConnected = true;

   while(notConnected)
   {   
       
      ci = new cursesINDI("cursesINDI", "1.7", "1.7");


      #ifdef DEBUG_TMPOUT
      ci->fpout = fpout;
      #endif
   
      ci->activate();
   
      pcf::IndiProperty ipSend;
      ci->sendGetProperties( ipSend );

      sleep(2);
      if(ci->getQuitProcess())
      {
         ci->quitProcess();
         ci->deactivate();
         delete ci;
         std::cout << "\r cursesINDI: Connection to INDI server failed.  Will retry in 5...";
         sleep(1);
         for(int i=4; i > 0; --i)
         {
            std::cout << i << "...";
            sleep(1);
         }
         std::cout << "\r cursesINDI: Retrying connection to INDI server . . .                                            \r";
      }
      else
      {
         notConnected = false;
      }
   }
   
   if(notConnected) return 0;
   //sleep(1);
   
   
   //return 0;
   
   WINDOW * topWin;

   int ch;

   initscr();
   
   raw();    /* Line buffering disabled*/
   halfdelay(10); //We use a 1 second timeout to check for connection loss.
   keypad(stdscr, TRUE); /* We get F1, 2 etc...*/

   noecho(); /* Don't echo() while we do getch */

   ci->startUp();
   
   


   topWin = newwin(1, COLS, 0, 0);
   wprintw(topWin, "cursesINDI (arrow keys move, ctrl-c to quit)");
   keypad(topWin, TRUE);
   wrefresh(topWin);

   
   ci->cursStat(0);
   
   
   //Now main event loop
   while((ch = wgetch(ci->w_interactWin))) // != 'q')
   {
      if(ch == ERR)
      {
         //if(fpout) *fpout << "loop" << std::endl;
         if( ci->getQuitProcess() || ci->m_shutdown) break;
         else continue;
      }
      
      //Get hold downs
      int ch0 = ch;
      int npress = 1;
      
      nocbreak();
      wtimeout(ci->w_interactWin, 50);
      ch = wgetch(ci->w_interactWin);
      while(ch == ch0) 
      {
         ch = wgetch(ci->w_interactWin);
         ++npress; 
      }
      if(ch != ERR) ungetch(ch);
      halfdelay(10);   
      
      int nextX = ci->m_currX;
      int nextY = ci->m_currY;

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
            nextY -= ci->m_gridWin.size()-1;
            if(fpout) *fpout << "ppage: " << npress << std::endl;
            break;   
         case KEY_NPAGE:
            nextY += ci->m_gridWin.size()-1;
            if(fpout) *fpout << "npage: " << npress << std::endl;
            break;
         case KEY_RESIZE:
            ci->draw();
            if(fpout) *fpout << "resizes: " << npress << std::endl;
            continue;
         default:
            if(fpout) *fpout << "other: " << npress << std::endl;
            ci->keyPressed(ch0);
            continue;
            break;
      }

      int maxX = ci->m_gridWin[0].size()-1;
      if(nextX < 1) nextX = 1;
      if(nextX >= maxX) nextX = maxX;

      ci->moveCurrent(nextY, nextX);

      ci->cursStat(0);
      
      if(ci->m_shutdown) break;
   }

   ci->shutDown();

   endwin();   /* End curses mode */

   sleep(1);
   if(ci->m_connectionLost)
   {
      delete ci;

      std::cout << "\r cursesINDI: lost connection to indiserver.  Will retry in 5...";
      
      sleep(1);
      for(int i=4; i > 0; --i)
      {
         std::cout << i << "...";
         sleep(1);
      }
      std::cout << "\r cursesINDI: Retrying connection to INDI server . . .                                            \r";
      goto retry;
   }
   
   std::cout << "\r cursesINDI: Disconnected from INDI server.                                                              \n";
   return 0;


}
