#ifndef cursesTableRow_hpp
#define cursesTableRow_hpp

#ifdef DEBUG_TMPOUT
#include <fstream>
#endif


class cursesTableRow
{
public:

   int m_rowY {0};
   int m_rowX {0};
   int m_rowHeight {1};
   int m_rowWidth {-1};

   std::vector<int> m_cellX;


   std::vector<std::string> m_cellContents;

private:
   bool own {false}; //Need this for move semantics

public:
   std::vector<WINDOW *> m_cellWin;


   cursesTableRow()
   {
   }

   /// Move constructor
   /** Allows safe copy-construction for vector-resize
     */
   cursesTableRow( cursesTableRow && ctr ) : m_rowY{ctr.m_rowY}, 
                                             m_rowX{ctr.m_rowX}, 
                                             m_rowHeight{ctr.m_rowHeight}, 
                                             m_rowWidth{ctr.m_rowWidth},
                                             m_cellX{ctr.m_cellX},
                                             m_cellContents{ctr.m_cellContents},
                                             m_cellWin{ctr.m_cellWin}
   {
      ctr.own=false;
   }

   ~cursesTableRow();

   void deleteRowWins();

   void deleteRow();

   void createRow( int rowH,
                   int rowW,
                   int rowY,
                   int rowX,
                   std::vector<int> cellX
                 );

   void updateContents( const std::vector<std::string> & cellContents,
                        bool force = false,
                        bool display = true
                      );

   void updateContents( size_t cellNo,
                        const std::string & cellContents,
                        bool force = false,
                        bool display = true
                      );

   void recreate( bool display );

   int cellY( int cellNo );

   int cellX( int cellNo );

   void keyPressed( int cellNo,
                    int ch
                  );
   
   #ifdef DEBUG_TMPOUT
public:
   std::ofstream * fout {0};
   #endif
};

cursesTableRow::~cursesTableRow()
{
   deleteRowWins();
}

void cursesTableRow::deleteRowWins()
{
   for(size_t i=0; i<m_cellWin.size(); ++i)
   {
      if(m_cellWin[i] && own) delwin(m_cellWin[i]);
   }
}

void cursesTableRow::deleteRow()
{
   deleteRowWins();
   m_cellContents.clear();
   m_cellX.clear();
}

void cursesTableRow::createRow( int rowH,
                                int rowW,
                                int rowY,
                                int rowX,
                                std::vector<int> cellX
                              )
{
   if(cellX.size() != m_cellX.size())
   {
      //A reset is called for.
      deleteRow();

      m_cellX = cellX;
      m_cellContents.resize(m_cellX.size());
      m_cellWin.resize(m_cellX.size(), 0);
   }
   else
   {
      //Check if we actually do anyting
      bool cellsSame = true;
      for(size_t i=0; i<m_cellX.size(); ++i)
      {
         if(cellX[i] != m_cellX[i])
         {
            cellsSame = false;
            break;
         }
      }

      //If no changes then we're out.
      if( cellsSame && rowH == m_rowHeight && rowW == m_rowWidth && rowY == m_rowY && rowX == m_rowX ) return;

      deleteRowWins();
   }

   m_rowHeight = rowH;
   m_rowWidth = rowW;
   m_rowY = rowY;
   m_rowX = rowX;

   own = true;
   for(size_t i=0; i< m_cellWin.size()-1; ++i)
   {
      m_cellWin[i] = newwin(m_rowHeight, m_cellX[i+1]-m_cellX[i], m_rowY, m_rowX + m_cellX[i]);
      keypad(m_cellWin[i], TRUE);
   }

   //Do last one outside loop to handle max-width case
   int l = m_cellWin.size()-1;
   int w = m_rowWidth;
   if(w < 0) w = COLS;
   m_cellWin[l] = newwin(m_rowHeight, w-m_cellX[l], m_rowY, m_rowX + m_cellX[l]);
   keypad(m_cellWin[l], TRUE);

   //updateContents(m_cellContents, true);

}

void cursesTableRow::updateContents( const std::vector<std::string> & cellContents,
                                     bool force,
                                     bool display
                                   )
{
   if(cellContents.size() != m_cellContents.size())
   {
      std::cout << "Wrong cell contents size" << std::endl;
      return;
   }

   for(size_t i=0; i< m_cellContents.size(); ++i)
   {
      if(cellContents[i] != m_cellContents[i] || force)
      {
         m_cellContents[i] = cellContents[i];
         if(display)
         {
            wclear(m_cellWin[i]);
            wprintw(m_cellWin[i], m_cellContents[i].c_str());
            wrefresh(m_cellWin[i]);
         }
      }
   }
}

void cursesTableRow::updateContents( size_t cellNo,
                                     const std::string & cellContents,
                                     bool force,
                                     bool display
                                   )
{
   if(cellNo >= m_cellContents.size()) return;

   if(cellContents != m_cellContents[cellNo] || force)
   {
      m_cellContents[cellNo] = cellContents;
      if(display)
      {
         wclear(m_cellWin[cellNo]);
         wprintw(m_cellWin[cellNo], m_cellContents[cellNo].c_str());
         wrefresh(m_cellWin[cellNo]);
      }
   }
}

void cursesTableRow::recreate( bool display )
{
   own = true;
   for(size_t i=0; i< m_cellWin.size()-1; ++i)
   {
      #ifdef DEBUG_TMPOUT
      if(fout) *fout << __FILE__ << " " << __LINE__<< " " << i << " " << m_cellWin.size() << std::endl;
      #endif
      delwin(m_cellWin[i]);
      m_cellWin[i] = newwin(m_rowHeight, m_cellX[i+1]-m_cellX[i], m_rowY, m_rowX + m_cellX[i]);
      keypad(m_cellWin[i], TRUE);
   }

   
      
   //Do last one outside loop to handle max-width case
   int l = m_cellWin.size()-1;
   int w = m_rowWidth;
   if(w < 0) w = COLS;

   #ifdef DEBUG_TMPOUT
   if(fout)  *fout << __FILE__ << " " << __LINE__<< " " << l << " " << m_cellWin.size() << std::endl;
   #endif
   delwin(m_cellWin[l]);
   m_cellWin[l] = newwin(m_rowHeight, w-m_cellX[l], m_rowY, m_rowX + m_cellX[l]);
   keypad(m_cellWin[l], TRUE);

   updateContents(m_cellContents, true, display);

}

int cursesTableRow::cellY( int cellNo )
{
   static_cast<void>(cellNo);

   return m_rowY;
}

int cursesTableRow::cellX( int cellNo )
{
   return m_rowX + m_cellX[cellNo];
}

void cursesTableRow::keyPressed( int cellNo,
                                 int ch
                               )
{
   std::string newStr;
   int nch;

   switch(ch)
   {
      case 'e':
         while( (nch = wgetch(m_cellWin[cellNo])) != '\n')
         {
            if(nch == 27)
            {
               newStr = m_cellContents[cellNo];
               break;
            }
            wprintw(m_cellWin[cellNo], "%c", nch);
            wrefresh(m_cellWin[cellNo]);

            newStr += (char) nch;
         }
         updateContents(cellNo, newStr, true);
         break;
      default:
         break;
   }
}

#endif //cursesTableRow_hpp
