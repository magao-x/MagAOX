
#ifndef cursesTableGrid_hpp
#define cursesTableGrid_hpp

#include <fstream>

#include <ncurses.h>

class cursesTableGrid
{
public:
   
   std::ofstream * fpout {nullptr};
   
   int m_xLeft {1};
   int m_yTop {1};
   int m_xRight {-1};
   int m_yBot {-1};
   
   int m_minWidth {78};
   
   size_t m_currX {0};
   size_t m_currY {0};
   
   size_t m_startRow {0};
   
   std::vector<std::vector<std::string>> m_cellContents;
   
   std::vector<float> m_colFraction;
   
   
   WINDOW * m_boxWin {nullptr};
   std::vector<std::vector<WINDOW *>> m_gridWin; 
   
   int minWidth( int mw );
   
   int colWidth( const std::vector<int> & cw );
   
   int colFraction( const std::vector<float> & cf );
   
   int tabHeight()
   {
      return m_gridWin.size();
   }
   
   bool hasContent( size_t y, 
                    size_t x
                  );
   
   int draw();
   virtual int postDraw()
   {
      return 0;
   }
   
   int print();
   
   virtual int postPrint()
   {
      return 0;
   }
   
   int moveSelected( int nextY, 
                     int nextX
                   );
   
   //Call after setting fraction.
   int testContents(int nrows);
};

inline
int cursesTableGrid::minWidth( int mw )
{
   m_minWidth = mw;
   return 0;
}

inline
int cursesTableGrid::colWidth( const std::vector<int> & cw )
{
   
   int mw = 0;
   for(size_t i=0;i<cw.size();++i) mw += cw[i];

   if( minWidth(mw) < 0)
   {
      return -1;
   }

   std::vector<float> cf(cw.size());

   for(size_t i=0;i<cf.size();++i)
   {
      cf[i] = cw[i];
      cf[i] /= m_minWidth;
   }
   
   return colFraction(cf);
}

inline 
int cursesTableGrid::colFraction( const std::vector<float> & cf )
{
   m_colFraction = cf;
   
   return 0;
}

inline 
bool cursesTableGrid::hasContent( size_t y, 
                                  size_t x
                                )
{
   bool hascontent = false;
   if(y < m_cellContents.size())
   {
      if(x < m_cellContents[y].size()) hascontent = true;
   }
         
   return hascontent;
}

inline 
int cursesTableGrid::draw()
{
   //Todo: box on/off selectable
   //Todo: if box on, must be y0>=1, x0>=1
   
   int tabHeight = LINES - m_yTop - 1;
   if(m_yBot > 0 ) tabHeight -= m_yBot; //else  -= 1.
   
   int tabWidth = COLS - m_xLeft - 1;
   if(m_xRight > 0 ) tabWidth -= m_xRight + 1;
   
   if(m_boxWin) 
   {  werase(m_boxWin);
      wrefresh(m_boxWin);
      delwin(m_boxWin);
   }
   
   m_boxWin = newwin(tabHeight+2, tabWidth+2, m_yTop-1,m_xLeft-1);
   box(m_boxWin, 0, 0);
   wrefresh(m_boxWin);
   
   if(m_gridWin.size() > 0)
   {
      for(size_t y =0; y < m_gridWin.size(); ++y)
      {
         for(size_t x=0; x < m_gridWin[y].size(); ++x)
         {
            delwin(m_gridWin[y][x]);
         }
      }
   }
   
   int nDispRow = tabHeight;
   
   int workingWidth = tabWidth;
   if(workingWidth < m_minWidth) workingWidth = m_minWidth;
   
   int spos = 0;
   std::vector<int> dispCol_start;
   std::vector<int> dispCol_width;
   for(size_t i=0; i< m_colFraction.size(); ++i)
   {
      int cw = m_colFraction[i] * workingWidth + 0.5; //rounded
      
      if(spos + cw > tabWidth) cw = tabWidth-spos;
      
      if(cw == 0 ) break;
      
      dispCol_start.push_back(spos);
      dispCol_width.push_back(cw);
      
      spos += cw;
      
      if(spos >= tabWidth) break;
   }
   
   m_gridWin.resize(nDispRow);
   
   std::string ch;
   
   for(size_t y=0; y< m_gridWin.size(); ++y)
   {
      m_gridWin[y].resize(dispCol_start.size());
      
      if(fpout) *fpout << y; 
      for(size_t x=0; x< m_gridWin[y].size(); ++x)
      {
         m_gridWin[y][x] = newwin(1, dispCol_width[x], m_yTop+y,m_xLeft + dispCol_start[x]);
         
         if(fpout) *fpout << ": " << x << " (" <<  dispCol_start[x] << "," << dispCol_width[x] << ")";
         
         if(hasContent(m_startRow+y,x)) wprintw(m_gridWin[y][x], "%s", m_cellContents[m_startRow + y][x].c_str());
         //else wprintw(m_gridWin[y][x], ".");
         wrefresh(m_gridWin[y][x]);
      }
      if(fpout) *fpout << std::endl; 
   }
   
   //Move cursor if needed.
   if(m_currY >= m_gridWin.size()) m_currY = m_gridWin.size()-1;
   if(m_currX >= m_gridWin[m_currY].size()) m_currX = m_gridWin[m_currY].size()-1;
   
   wattron(m_gridWin[m_currY][m_currX], A_REVERSE);
   wclear(m_gridWin[m_currY][m_currX]);
   if(hasContent(m_startRow + m_currY, m_currX))  wprintw(m_gridWin[m_currY][m_currX], "%s", m_cellContents[m_startRow+m_currY][m_currX].c_str());
   wrefresh(m_gridWin[m_currY][m_currX]);
   
   
   if(fpout) *fpout << "drawn" << std::endl;
   
   return postDraw();
}

inline
int cursesTableGrid::print()
{
   for(size_t y=0; y< m_gridWin.size(); ++y)
   {
      for(size_t x=0; x< m_gridWin[y].size(); ++x)
      {
         wclear(m_gridWin[y][x]);
         if(hasContent(m_startRow+y,x)) wprintw(m_gridWin[y][x], "%s", m_cellContents[m_startRow + y][x].c_str());
         wrefresh(m_gridWin[y][x]);
      }
   }
 
   return postPrint();
}
   
inline
int cursesTableGrid::testContents(int nrows)
{
   m_cellContents.resize(nrows);
   
   for(size_t y=0;y<m_cellContents.size(); ++y)
   {
      m_cellContents[y].resize(m_colFraction.size());
   }
   
   std::string ch;
   for(size_t y=0;y<m_cellContents.size(); ++y)
   {
      for(size_t x=0; x< m_colFraction.size(); ++x)
      {
         size_t cw = m_colFraction[x] * m_minWidth + 0.5;
         
         ch=std::to_string(y) + "," + std::to_string(x);
         if( cw > ch.size()+1) ch += std::string( cw-ch.size()-1,'*');
         ch += "|";
         
         if(ch.size() > cw) ch.erase(cw);
         
         m_cellContents[y][x] = ch;
      }
   }
   
   return 0;
}


inline
int cursesTableGrid::moveSelected( int nextY,
                                   int nextX
                                 )
{
   //Do some bounds checks
   if(m_cellContents.size() == 0 || m_currY >= m_gridWin.size()) return 0;
   if(m_currX >= m_gridWin[m_currY].size()) return 0;

   //Now turn off the reverse
   wattroff(m_gridWin[m_currY][m_currX], A_REVERSE);
   
   wclear(m_gridWin[m_currY][m_currX]);
   if(hasContent(m_startRow + m_currY, m_currX))  wprintw(m_gridWin[m_currY][m_currX], "%s", m_cellContents[m_startRow+m_currY][m_currX].c_str());
   wrefresh(m_gridWin[m_currY][m_currX]);
            
   if(fpout) *fpout << "nextY: " << nextY << " m_gridWin.size(): " <<  m_gridWin.size() << std::endl;
   
   bool reprint = false;
   if(nextY >= (int64_t) m_gridWin.size())
   {
      size_t oldSR = m_startRow;
      m_startRow += nextY - m_gridWin.size() + 1;
   
      if(m_startRow >= m_cellContents.size()) m_startRow = m_cellContents.size()-1;
      
      if( oldSR != m_startRow) reprint = true;
      
      if(fpout) *fpout << "+m_startRow: " << m_startRow << " " << reprint << std::endl;
      
      nextY = m_gridWin.size()-1;
   }
   
   if(nextY < 0)
   {
      size_t oldSR = m_startRow;
      
      if(m_startRow > 0) 
      {
         if(m_startRow < (size_t) (-nextY)) m_startRow = 0;
         else m_startRow += nextY;
      }
      
      if( oldSR != m_startRow) reprint = true;
      
      if(fpout) *fpout << "-m_startRow: " << m_startRow << " " << reprint << std::endl;
      
      nextY = 0;
   }
   
   if(reprint) print();
   
   //Move the cursor position
   if(nextY >= 0 &&  nextY < (int64_t) m_gridWin.size() && nextX >= 0 && nextX < (int64_t) m_gridWin[0].size())
   {
      m_currY = nextY;
      m_currX = nextX;
   }

   //Turn it back on
   wattron(m_gridWin[m_currY][m_currX], A_REVERSE);
   
   wclear(m_gridWin[m_currY][m_currX]);
   if(hasContent(m_startRow + m_currY, m_currX))  wprintw(m_gridWin[m_currY][m_currX], "%s", m_cellContents[m_startRow+m_currY][m_currX].c_str());
   else wprintw(m_gridWin[m_currY][m_currX], " ");
   wrefresh(m_gridWin[m_currY][m_currX]);
   
   wattroff(m_gridWin[m_currY][m_currX], A_REVERSE);
   
   return 0;
}


#endif //cursesTableGrid_hpp
