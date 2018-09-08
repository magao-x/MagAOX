#ifndef cursesTable_hpp
#define cursesTable_hpp

#include <vector>
#include <string>
   
#include <unistd.h>
#include <ncurses.h>

#include "cursesTableRow.hpp"

class cursesTable 
{
public:
   int m_tabY {0};
   int m_tabX {0};
   int m_tabHeight {0};
   int m_tabWidth {0};
   
   int m_currFirstRow {0};
   
   std::vector<cursesTableRow> rows;
   
   int addRow( const std::vector<int> & cellX );
   
   void updateContents( int rowNo,
                        const std::vector<std::string> & contents 
                      );

   void updateContents( int rowNo,
                        int cellNo,
                        const std::string & contents 
                      );
   
   /// Update the Y position of all rows, for scrolling
   void updateRowY( int firstRow );

};

int cursesTable::addRow( const std::vector<int> & cellX )
{
   rows.emplace_back();
   rows.back().createRow(1, m_tabWidth, m_tabY + rows.size()-1, m_tabX, cellX);
   
   return rows.size()-1;
}

void cursesTable::updateContents( int rowNo,
                                  const std::vector<std::string> & contents 
                                )
{
   bool display = false;
   if(rows[rowNo].m_rowY >= m_tabY && rows[rowNo].m_rowY < m_tabY + m_tabHeight) display = true;
         
   rows[rowNo].updateContents(contents, false, display);
   
}

void cursesTable::updateContents( int rowNo,
                                  int cellNo,
                                  const std::string & contents 
                                )
{
   if(rowNo < 0  || (size_t) rowNo >= rows.size()) return;
   
   bool display = false;
   if(rows[rowNo].m_rowY >= m_tabY && rows[rowNo].m_rowY < m_tabY + m_tabHeight) display = true;
         
   rows[rowNo].updateContents(cellNo, contents, false, display);
   
}

void cursesTable::updateRowY( int firstRow )
{
   if( firstRow == m_currFirstRow) return;
   
   m_currFirstRow = firstRow;
   
   for(size_t i=0; i< rows.size(); ++i)
   {
      rows[i].m_rowY = m_tabY + (i - m_currFirstRow);
      
      bool display = false;
      
      if(rows[i].m_rowY >= m_tabY && rows[i].m_rowY < m_tabY + m_tabHeight)
      {
         display = true;
      }
      
      rows[i].recreate(display); //updateContents(rows[i].m_cellContents, true);
   }
}

#endif //cursesTable_hpp
