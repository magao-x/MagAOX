/** \file logsurgeon.hpp
  * \brief A utility to fix corrupted MagAO-X binary logs.
  *
  * \ingroup logsurgeon_files
  */

#ifndef logsurgeon_hpp
#define logsurgeon_hpp

#include <iostream>
#include <cstring>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
using namespace MagAOX::logger;

using namespace flatlogs;


/** \defgroup logsurgeon logsurgeon: MagAO-X Log Corrector
  * \brief Read a MagAO-X binary log file and remove corrupted bytes.
  *
  * <a href="../handbook/utils/logsurgeon.html">Utility Documentation</a>
  *
  * \ingroup utils
  *
  */

/** \defgroup logsurgeon_files logsurgeon Files
  * \ingroup logsurgeon
  */

/// An application to fix corrupted MagAo-X binary logs.
/** \todo document this
  *
  * \ingroup logsurgeon
  */
class logsurgeon : public mx::app::application
{
protected:

   std::string m_fname;

public:
   virtual void setupConfig();

   virtual void loadConfig();

   virtual int execute();
};

void logsurgeon::setupConfig()
{
   config.add("file","F", "file" , argType::Required, "", "file", true,  "string", "The single file to process.  If no / are found in name it will look in the specified directory (or MagAO-X default).");
}

void logsurgeon::loadConfig()
{
   config(m_fname, "file");
}

int logsurgeon::execute()
{
   if(m_fname == "")
   {
      std::cerr << "Must specify filename with -F option.\n";
      return EXIT_FAILURE;
   }


   FILE * fin ;
   fin = fopen(m_fname.c_str(), "rb");

   if(!fin)
   {
      std::cerr << "Error opening file " << m_fname << "\n";
      return EXIT_FAILURE;
   }

   ssize_t fsz = mx::ioutils::fileSize(fin);


   char * buff = new char[fsz];
   char * gbuff = new char[fsz];

   ssize_t nrd = fread(buff, 1, fsz, fin); 
   if(nrd != fsz)
   {
      std::cerr << __FILE__ << " " << __LINE__ << " did not read complete file.\n";
      return -1;
   }
   fclose(fin);

   ssize_t gcurr = 0;
   bool inbad = false;
   ssize_t lastGoodSt = 0;
   ssize_t lastGoodSz = 0;

   ssize_t totBad = 0;
   ssize_t badSt = 0;
   ssize_t kpt = sizeof(logPrioT);
   
   //Now check each byte to see if it is a potential start of a valid log
   while(kpt < fsz)
   {
      eventCodeT ec = * ( (eventCodeT*) (&buff[kpt]));

      if( logCodeValid(ec) )
      {
         char * buffst = &buff[kpt- sizeof(logPrioT)];
         
         msgLenT len = logHeader::msgLen(buffst);
         msgLenT totLen = len + logHeader::headerSize(buffst);

         //Basic check if size isn't too big
         if( kpt - (ssize_t) sizeof(logPrioT) + (ssize_t) totLen < (ssize_t) fsz)
         {
            //Now we use the flatlogs verifier.
            char * nbuff = new char[totLen];
            
            memcpy(nbuff, buffst, totLen);

            bufferPtrT buffPtr = bufferPtrT(nbuff);

            if(logVerify(ec, buffPtr, len))
            {
               if(inbad)
               {
                  inbad = false;

                  char * lastGBuff = new char[lastGoodSz];
                  memcpy(lastGBuff, &buff[lastGoodSt], lastGoodSz);
                  bufferPtrT lgBuffPtr = bufferPtrT(lastGBuff);

                  std::cerr << "Found corrupt section: \n";
                  std::cerr << "   Before: ";
                  logStdFormat( std::cerr, lgBuffPtr);
                  std::cerr << "\n";

                  //printLogBuff(lglvl, lgec, logHeader::msgLen(lastGBuff), lgBuffPtr);

                  std::cerr << "   Corrupt: " << badSt << " - " << kpt << " (" << kpt-badSt << " bytes)\n";
                  totBad += kpt-badSt;

                  std::cerr << "   After:  ";
                  logStdFormat( std::cerr, buffPtr);
                  std::cerr << "\n";
               }

               memcpy(&gbuff[gcurr], &buff[kpt-sizeof(logPrioT)], totLen);

               lastGoodSt = kpt-sizeof(logPrioT);
               lastGoodSz = totLen;

               gcurr += totLen;
               kpt += totLen;

               continue;
            }

         }
      }
      if(inbad == false)
      {
         badSt = kpt;
      }

      inbad = true;
      ++kpt;
   }

   std::cerr << "--------------------------------------------------------\n";
   std::cerr << "Found " << totBad << " bad bytes ( " << (100.0*totBad)/fsz << "% bad) \n";
   std::cerr << "Found " << gcurr << " good bytes ( " << (100.0*gcurr) / fsz  <<  "% good)\n";

   if(totBad == 0)
   {
      std::cerr << "Taking no action on good file.\n";
   }
   else
   {
      std::string bupPath = m_fname + ".corrupted";

      FILE * fout;
      fout = fopen(bupPath.c_str(), "wb");

      if(!fout)
      {
         std::cerr << "Error opening corrupted file for writing (" __FILE__ << " " << __LINE__ << ")\n";
         std::cerr << "No further action taken\n";
         return EXIT_FAILURE;
      }

      ssize_t fwr = fwrite(buff, sizeof(char), fsz, fout);

      int fcst = fclose(fout);

      if(fwr != fsz)
      {
         std::cerr << "Error writing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
         std::cerr << "No further action taken\n";
         return EXIT_FAILURE;
      }

      if(fcst != 0)
      {
         std::cerr << "Error closing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
         std::cerr << "No further action taken\n";
         return EXIT_FAILURE;
      }

      std::cerr << "Wrote original file to: " << bupPath << "\n";

      fout = fopen(m_fname.c_str(), "wb");

      if(!fout)
      {
         std::cerr << "Error opening existing file for writing (" __FILE__ << " " << __LINE__ << ")\n";
         std::cerr << "No further action taken\n";
         return EXIT_FAILURE;
      }

      fwr = fwrite(gbuff, sizeof(char), gcurr, fout);

      fcst = fclose(fout);

      if(fwr != gcurr)
      {
         std::cerr << "Error writing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
         return EXIT_FAILURE;
      }
   
      if(fcst != 0)
      {
         std::cerr << "Error closing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
         return EXIT_FAILURE;
      }

      std::cerr << "Wrote corrected file to: " << m_fname << "\n";

      std::cerr << "Surgery Complete\n";
   }
   delete[] buff;
   delete[] gbuff;

   return 0;
}

#endif //logsurgeon_hpp
