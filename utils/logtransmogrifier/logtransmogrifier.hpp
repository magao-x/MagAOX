/** \file logtransmogrifier.hpp
  * \brief A utility to fix corrupted MagAO-X binary logs.
  *
  * \ingroup logtransmogrifier_files
  */

#ifndef logtransmogrifier_hpp
#define logtransmogrifier_hpp

#include <iostream>
#include <cstring>

#include <mx/ioutils/fileUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../libMagAOX/logger/types/old_telem_fxngen.hpp"
#include "../../libMagAOX/logger/types/old_telem_fxngen_generated.h"
#include "../../libMagAOX/logger/types/generated/telem_fxngen_generated.h"
using namespace MagAOX::logger;

using namespace flatlogs;


/** \defgroup logtransmogrifier logtransmogrifier: MagAO-X Log Corrector
  * \brief Read a MagAO-X binary log file with the old function generator telemetry schema and write with the new
  *
  * <a href="../handbook/utils/logtransmogrifier.html">Utility Documentation</a>
  *
  * \ingroup utils
  *
  */

/** \defgroup logtransmogrifier_files logtransmogrifier Files
  * \ingroup logtransmogrifier
  */

/// An application to fix corrupted MagAo-X binary logs.
/** \todo document this
  *
  * \ingroup logtransmogrifier
  */
class logtransmogrifier : public mx::app::application
{
protected:

   std::string m_fname;

public:
   virtual void setupConfig();

   virtual void loadConfig();

   virtual int execute();
};

void logtransmogrifier::setupConfig()
{
   config.add("file","F", "file" , argType::Required, "", "file", true,  "string", "The single file to process.  If no / are found in name it will look in the specified directory (or MagAO-X default).");
}

void logtransmogrifier::loadConfig()
{
   config(m_fname, "file");
}

int logtransmogrifier::execute()
{
   if(m_fname == "")
   {
      std::cerr << "Must specify filename with -F option.\n";
      return EXIT_FAILURE;
   }


   FILE * fin ;
   fin = fopen(m_fname.c_str(), "rb");
   FILE * fout;
   std::string outfn = m_fname + ".updated";
   // fout = fopen(outfn.c_str(), "wb");

   if(!fin)
   {
      std::cerr << "Error opening file " << m_fname << "\n";
      std::cerr << "errno says: " << strerror(errno) << "\n";
      return EXIT_FAILURE;
   }

   // if(!fout)
   // {
   //    std::cerr << "Error opening file " << outfn << "\n";
   //    std::cerr << "errno says: " << strerror(errno) << "\n";
   //    return EXIT_FAILURE;
   // }

   ssize_t fsz = mx::ioutils::fileSize(fin);

   char * buff = new char[fsz];

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
      char * buffst = &buff[kpt- sizeof(logPrioT)];

      msgLenT len = logHeader::msgLen(buffst);
      msgLenT totLen = len + logHeader::headerSize(buffst);

      if (ec == MagAOX::logger::eventCodes::TELEM_FXNGEN) {
         //Now we use the flatlogs verifier.
         char * nbuff = new char[totLen];
         
         memcpy(nbuff, buffst, totLen);

         bufferPtrT buffPtr = bufferPtrT(nbuff);
         logPrioT lvl = logHeader::logLevel(buffPtr);
         eventCodeT ec = logHeader::eventCode(buffPtr);
         msgLenT len = logHeader::msgLen(buffPtr);
         timespecX ts = logHeader::timespec(buffPtr);
         auto fbs = GetOldTelem_fxngen_fb((uint8_t*) flatlogs::logHeader::messageBuffer(buffPtr));
         auto verifier = flatbuffers::Verifier( (uint8_t*) flatlogs::logHeader::messageBuffer(buffPtr), static_cast<size_t>(len));
         if(!VerifyOldTelem_fxngen_fbBuffer(verifier)) {
            std::cerr << "heck\n";
            return 1;
         } else {
            std::string oldMsgString = MagAOX::logger::old_telem_fxngen::msgString((uint8_t*) flatlogs::logHeader::messageBuffer(buffPtr), totLen);
            std::cout << "old: " << oldMsgString << "\n";
            std::cout << "old: " << MagAOX::logger::old_telem_fxngen::makeJSON((uint8_t*) flatlogs::logHeader::messageBuffer(buffPtr), totLen, OldTelem_fxngen_fbTypeTable()) << "\n";

            double widthMissingSentinel = 0.0;
            uint8_t C1outp = fbs->C1outp();
            double C1freq = fbs->C1freq();
            double C1vpp = fbs->C1vpp();
            double C1ofst = fbs->C1ofst();
            double C1phse = fbs->C1phse();
            auto wvtpNum = fbs->C1wvtp();
            std::string C1wvtp;
            if (wvtpNum == 0) {
               C1wvtp = "DC";
            } else if (wvtpNum == 1) {
               C1wvtp = "SINE";
            } else if (wvtpNum == 2) {
               C1wvtp = "PULSE";
            } else {
               C1wvtp = "UNK";
            }
            uint8_t C2outp = fbs->C2outp();
            double C2freq = fbs->C2freq();
            double C2vpp = fbs->C2vpp();
            double C2ofst = fbs->C2ofst();
            double C2phse = fbs->C2phse();
            wvtpNum = fbs->C2wvtp();
            std::string C2wvtp;
            if (wvtpNum == 0) {
               C2wvtp = "DC";
            } else if (wvtpNum == 1) {
               C2wvtp = "SINE";
            } else if (wvtpNum == 2) {
               C2wvtp = "PULSE";
            } else {
               C2wvtp = "UNK";
            }
            uint8_t C1sync = fbs->C1sync();
            uint8_t C2sync = fbs->C2sync();

            double C1wdth = fbs->C1wdth();
            double C2wdth = fbs->C2wdth();

            telem_fxngen::messageT newMessage = telem_fxngen::messageT(
               C1outp,
               C1freq,
               C1vpp,
               C1ofst,
               C1phse,
               C1wdth,
               C1wvtp,
               C2outp,
               C2freq,
               C2vpp,
               C2ofst,
               C2phse,
               C2wdth,
               C2wvtp,
               C1sync,
               C2sync
            );

            bufferPtrT newLogBuffer;
            logHeader::createLog<MagAOX::logger::telem_fxngen>(newLogBuffer, ts, newMessage, lvl);
            std::string newMsgString = MagAOX::logger::telem_fxngen::msgString((uint8_t*) flatlogs::logHeader::messageBuffer(newLogBuffer), totLen);

            std::cout << "new: " << newMsgString << "\n";
            std::cout << "new: " << MagAOX::logger::telem_fxngen::makeJSON((uint8_t*) flatlogs::logHeader::messageBuffer(newLogBuffer), len, Telem_fxngen_fbTypeTable()) << "\n";

            if(newMsgString != oldMsgString) {
               std::cerr << "heck\n";
               return 0;
            }

            size_t N = flatlogs::logHeader::totalSize(newLogBuffer);

            // size_t nwr = fwrite( newLogBuffer.get(), sizeof(char), N, fout);

            // if(nwr != N*sizeof(char))
            // {
            //    std::cerr << "Error by fwrite.  At: " << __FILE__ << " " << __LINE__ << "\n";
            //    std::cerr << "errno says: " << strerror(errno) << "\n";
            //    return -1;
            // }
         }

         // logHeader::createLog<old_telem_fxngen>(logBuffer, ts, msg, level);

         kpt += totLen;

         continue;
      }

   }

   // std::cerr << "--------------------------------------------------------\n";
   // std::cerr << "Found " << totBad << " bad bytes ( " << (100.0*totBad)/fsz << "% bad) \n";
   // std::cerr << "Found " << gcurr << " good bytes ( " << (100.0*gcurr) / fsz  <<  "% good)\n";

   // if(totBad == 0)
   // {
   //    std::cerr << "Taking no action on good file.\n";
   // }
   // else if (m_checkOnly)
   // {
   //    std::cerr << "Check-only mode set, exiting with error status to indicate failed verification\n";
   //    return EXIT_FAILURE;
   // }
   // else
   // {
   //    std::string bupPath = m_fname + ".corrupted";

   //    FILE * fout;
   //    fout = fopen(bupPath.c_str(), "wb");

   //    if(!fout)
   //    {
   //       std::cerr << "Error opening corrupted file for writing (" __FILE__ << " " << __LINE__ << ")\n";
   //       std::cerr << "No further action taken\n";
   //       return EXIT_FAILURE;
   //    }

   //    ssize_t fwr = fwrite(buff, sizeof(char), fsz, fout);

   //    int fcst = fclose(fout);

   //    if(fwr != fsz)
   //    {
   //       std::cerr << "Error writing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
   //       std::cerr << "No further action taken\n";
   //       return EXIT_FAILURE;
   //    }

   //    if(fcst != 0)
   //    {
   //       std::cerr << "Error closing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
   //       std::cerr << "No further action taken\n";
   //       return EXIT_FAILURE;
   //    }

   //    std::cerr << "Wrote original file to: " << bupPath << "\n";

   //    fout = fopen(m_fname.c_str(), "wb");

   //    if(!fout)
   //    {
   //       std::cerr << "Error opening existing file for writing (" __FILE__ << " " << __LINE__ << ")\n";
   //       std::cerr << "No further action taken\n";
   //       return EXIT_FAILURE;
   //    }

   //    fwr = fwrite(gbuff, sizeof(char), gcurr, fout);

   //    fcst = fclose(fout);

   //    if(fwr != gcurr)
   //    {
   //       std::cerr << "Error writing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
   //       return EXIT_FAILURE;
   //    }
   
   //    if(fcst != 0)
   //    {
   //       std::cerr << "Error closing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
   //       return EXIT_FAILURE;
   //    }

   //    std::cerr << "Wrote corrected file to: " << m_fname << "\n";

   //    std::cerr << "Surgery Complete\n";
   // }
   // delete[] buff;
   // delete[] gbuff;

   return 0;
}

#endif //logtransmogrifier_hpp
