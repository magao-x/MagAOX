/** \file flatlogcodes.cpp
  * \brief Program to parse a log type to code file.
  * \author Jared R. Males (jaredmales@gmail.com)
  *
  * \ingroup flatlogcodes
  * 
  * History:
  * - 2018-08-18 created by JRM
  */

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

#include "../include/flatlogs/logDefs.hpp"
using namespace flatlogs;

///Read in an event code file line by line, parsing for log type and code.
/** Looks for entries of the form:
  * \verbatim log_type 12349876 \endverbatim
  * Each log type must be on its own line.  The log type name and its code
  * are deliminated by whitespace.  On any line, all input after a \# char are ignored (allowing comments).
  *
  * In addition to parsing errors, duplicate log codes result in an error.  Errors are reported with
  * the line number of the input file.
  *
  * \returns 0 on success
  * \returns -1 on error
  * 
  */ 
int readCodeFile( std::map<eventCodeT, std::string> & codeMap, ///< [out] The map of codes to log types
                  const std::string & fileName ///< [in] the file to parse
                )
{
   typedef std::map<eventCodeT, std::string> codeMapT;
   
   std::fstream fin;

   fin.open(fileName);

   int lineNo = 0;
   while(1)
   {
      if(!fin.good()) break;

      std::string line;

      ++lineNo;
      try
      {
         getline(fin, line);
      }
      catch(std::exception & e)
      {
         std::cerr << fileName << " line " << lineNo << ": Exception: " << e.what() << "\n";
         return -1;
      }
      catch(...)
      {
         std::cerr << fileName << " line " << lineNo << ": unknown exception.\n";
         return -1;
      }

      
      size_t com = line.find('#', 0);

      if(com != std::string::npos) line.erase(com);

      if(line.size() == 0) continue;
      
      std::stringstream sstr(line, std::ios_base::in);

      std::string logType, logCodeStr;
      eventCodeT logCode;

      sstr >> logType;
      sstr >> logCodeStr;
      
      if(logCodeStr.size() == 0)
      {
         std::cerr << fileName << " line " << lineNo << ": no log code found.\n";
         return -1;
      }
      
      if(!isdigit(logCodeStr[0]))
      {
         std::cerr << fileName << " line " << lineNo << ": log code must be numeric.\n";
         return -1;
      }

      std::stringstream sstr2(logCodeStr, std::ios_base::in);
      sstr2 >> logCode;
      
      std::pair<codeMapT::iterator, bool> res;
      try
      {
          res = codeMap.insert( codeMapT::value_type(logCode, logType) );
      }
      catch(std::exception & e)
      {
         std::cerr << fileName << " line " << lineNo << ": Exception on map insertion: " << e.what() << "\n";
         return -1;
      }
      catch(...)
      {
         std::cerr << fileName << " line " << lineNo << ": unknown exception on map insertion.\n";
         return -1;
      }
      
      if(res.second == false)
      {
         std::cerr << fileName << " line " << lineNo << ": Duplicate log code.\n";
         std::cerr << "Original:    " << res.first->second << " " << res.first->first << "\n";
         std::cerr << "New attempt: " << logType << " " << logCode << "\n\n";
         return -1;
      }
   }
   
   return 0;
}

/// Write the logCodes.hpp header
int emitLogCodes( const std::string & fileName,
                  std::map<uint16_t, std::string> & logCodes 
                )
{
   typedef std::map<uint16_t, std::string> mapT;
   
   std::ofstream fout;
   fout.open(fileName);
   
   fout << "#ifndef logger_logCodes_hpp\n";
   fout << "#define logger_logCodes_hpp\n";
   fout << "#include <flatlogs/flatlogs.hpp>\n";
   fout << "namespace MagAOX\n";
   fout << "{\n";
   fout << "namespace logger\n";
   fout << "{\n";
   fout << "namespace eventCodes\n";
   fout << "{\n";
      mapT::iterator it = logCodes.begin();
      for(; it!=logCodes.end(); ++it)
      {
         std::string name = it->second;
         for(size_t i=0;i<name.size();++i) name[i] = toupper(name[i]);
         fout << "   constexpr static flatlogs::eventCodeT " << name << " = " << it->first <<";\n";
      }
   fout << "}\n";
   fout << "}\n";
   fout << "}\n";
   fout << "#endif\n";
   
   fout.close();
   return 0;
}

///Write the logStdFormat.hpp header.
int emitStdFormatHeader( const std::string & fileName,
                         std::map<uint16_t, std::string> & logCodes 
                       )
{
   typedef std::map<uint16_t, std::string> mapT;
   
   mapT::iterator it = logCodes.begin();
   
   std::ofstream fout;
   fout.open(fileName);

   
   fout << "#ifndef logger_logStdFormat_hpp\n";
   fout << "#define logger_logStdFormat_hpp\n";

   fout << "#include <flatlogs/flatlogs.hpp>\n";

   fout << "#include \"logTypes.hpp\"\n";

   ///\todo Need to allow specification of the namespaces
   fout << "namespace MagAOX\n";
   fout << "{\n";
   fout << "namespace logger\n";
   fout << "{\n";

   fout << "inline\n";
   fout << "void logStdFormat(flatlogs::bufferPtrT & buffer )\n";
   fout << "{\n";
   fout << "   flatlogs::eventCodeT ec;\n";
   fout << "   ec = flatlogs::logHeader::eventCode(buffer);\n";
   
   fout << "   switch(ec)\n";
   fout << "   {\n";
   for(; it!=logCodes.end(); ++it)
   {
      fout << "      case " << it->first << ":\n";
      fout << "         return flatlogs::stdFormat<" << it->second << ">(buffer);\n";
   }
      fout << "      default:\n";
      fout << "         std::cerr << \"Unknown log type: \" << ec << \"\\n\";\n";
   fout << "   }\n";
   fout << "}\n";

   fout << "}\n"; //namespace logger
   fout << "}\n"; //namespace MagAOX

   fout << "#endif\n"; //logger_logStdFormat_hpp

   fout.close();
   
   return 0;
}

/// Write the logTypes.hpp header
int emitLogTypes( const std::string & fileName,
                  std::map<uint16_t, std::string> & logCodes 
                )
{
   typedef std::map<uint16_t, std::string> mapT;
   
   mapT::iterator it = logCodes.begin();
   
   std::ofstream fout;
   fout.open(fileName);

   fout << "#ifndef logger_logTypes_hpp\n";
   fout << "#define logger_logTypes_hpp\n";
   fout << "#include \"logCodes.hpp\"\n";
   for(; it!=logCodes.end(); ++it)
   {
      fout << "#include \"../types/" << it->second << ".hpp\"\n";
   }   
   fout << "#endif\n";
   fout.close();
}

int main()
{
   typedef std::map<uint16_t, std::string> mapT;
   
   
   
   std::string inputFile = "logCodes.dat";
   std::string stdFormatHeader = "generated/logStdFormat.hpp";
   std::string logCodesHeader = "generated/logCodes.hpp";
   std::string logTypesHeader = "generated/logTypes.hpp";
   
   mapT logCodes;
   
   if( readCodeFile(logCodes, inputFile) < 0 )
   {
      std::cerr << "Error reading code file.\n";
      return -1;
   }
   
   emitStdFormatHeader(stdFormatHeader, logCodes );
   emitLogCodes( logCodesHeader, logCodes );
   emitLogTypes( logTypesHeader, logCodes );
   return 0;
}

