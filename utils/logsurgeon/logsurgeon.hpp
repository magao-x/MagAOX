/** \file logsurgeon.hpp
 * \brief A utility to fix corrupted MagAO-X binary logs.
 *
 * \ingroup files
 */

#ifndef hpp
#define hpp

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

/** \defgroup files logsurgeon Files
 * \ingroup logsurgeon
 */

/// An application to fix corrupted MagAO-X binary logs.
/** \todo document this
 *
 * \ingroup logsurgeon
 */
class logsurgeon : public mx::app::application
{
  protected:
    std::string m_fname; ///< The full path to the file to check

    bool m_checkOnly{ false }; /** If true then no modification to files on disk occurs, exit code
                                   0 indicates successful verification.  Default is false.*/

  public:
    enum returnVals
    {
        noerror            = 0, ///< no errors occurred
        file_not_specified = -1, ///< no file wa specified
        file_not_found     = -2, ///< the file was found (or an error occurred opening it)
        errors_found       = -100, ///< errors were found in the file in checkOnly mode
        error              = -9999 ///< other errors were found
    };

    virtual void setupConfig();

    virtual void loadConfig();

    virtual int execute();

    const std::string &fname();

    bool checkOnly();
};

void logsurgeon::setupConfig()
{
    config.add( "file",
                "F",
                "file",
                argType::Required,
                "",
                "file",
                true,
                "string",
                "The single file to process.  If no / are found in name it will look in the specified "
                "directory (or MagAO-X default)." );

    config.add( "check",
                "",
                "",
                argType::Required,
                "",
                "check-only",
                false,
                "bool",
                "Check-only mode config file setting. If true then no modification to files on disk occurs, "
                "exit code 0 indicates successful verification.  Default is false." );

    config.add( "checkCL",
                "C",
                "check-only",
                argType::True,
                "",
                "",
                false,
                "bool",
                "Check-only mode command-line flag. If true then no modification to files on disk occurs, "
                "exit code 0 indicates successful verification. Overrides config file.  Default is false." );
}

void logsurgeon::loadConfig()
{
    config( m_fname, "file" );
    config( m_checkOnly, "check" );

    // Command line always wins
    if( config.isSet( "checkCL" ) )
    {
        m_checkOnly = true;
    }
}

int logsurgeon::execute()
{
    if( m_fname == "" )
    {
        std::cerr << "Must specify filename with -F option.\n";
        return file_not_specified;
    }

    FILE *fin;
    fin = fopen( m_fname.c_str(), "rb" );

    if( !fin )
    {
        std::cerr << "Error opening file " << m_fname << "\n";
        return file_not_found;
    }

    ssize_t fsz = mx::ioutils::fileSize( fin );

    char *buff = new char[fsz];

    ssize_t nrd = fread( buff, 1, fsz, fin );
    fclose( fin );

    if( nrd != fsz )
    {
        std::cerr << __FILE__ << " " << __LINE__ << " did not read complete file.\n";
        delete[] buff;

        return error;
    }

    ssize_t gcurr      = 0;
    bool    inbad      = false;
    ssize_t lastGoodSt = 0;
    ssize_t lastGoodSz = 0;

    ssize_t totBad = 0;
    ssize_t badSt  = 0;
    ssize_t kpt    = sizeof( logPrioT );

    char *gbuff = new char[fsz];

    // Now check each byte to see if it is a valid eventCode,
    // which makes it a potential start of a valid log
    while( kpt < fsz )
    {
        eventCodeT ec = *( (eventCodeT *)( &buff[kpt] ) );

        if( logCodeValid( ec ) )
        {
            char *buffst = &buff[kpt - sizeof( logPrioT )];

            msgLenT len    = logHeader::msgLen( buffst );
            msgLenT totLen = len + logHeader::headerSize( buffst );

            // Basic check if size isn't too big (i.e. would extend past end of file)
            ssize_t endpt = kpt - static_cast<ssize_t>( sizeof( logPrioT ) ) + static_cast<ssize_t>( totLen );
            if( endpt < static_cast<ssize_t>( fsz ) )
            {
                // Now we use the flatlogs verifier.
                char *nbuff = (char *)::operator new( totLen * sizeof( char ) );

                memcpy( nbuff, buffst, totLen );

                bufferPtrT buffPtr = bufferPtrT( nbuff );

                // true means good
                if( logVerify( ec, buffPtr, len ) )
                {
                    // if we pass we check if we're currently in a bad section
                    if( inbad )
                    {
                        // if we were in a bad section we record the end of the bad section
                        inbad = false;

                        char *lastGBuff = (char *)::operator new( lastGoodSz * sizeof( char ) );

                        memcpy( lastGBuff, &buff[lastGoodSt], lastGoodSz );
                        bufferPtrT lgBuffPtr = bufferPtrT( lastGBuff );

                        std::cerr << "Found corrupt section: \n";
                        std::cerr << "   Before: ";
                        logStdFormat( std::cerr, lgBuffPtr );
                        std::cerr << "\n";

                        // printLogBuff(lglvl, lgec, logHeader::msgLen(lastGBuff), lgBuffPtr);

                        std::cerr << "   Corrupt: " << badSt << " - " << kpt << " (" << kpt - badSt << " bytes)\n";
                        totBad += kpt - badSt;

                        std::cerr << "   After:  ";
                        logStdFormat( std::cerr, buffPtr );
                        std::cerr << "\n";
                    }

                    // It's good so we write it to the good buffer
                    memcpy( &gbuff[gcurr], &buff[kpt - sizeof( logPrioT )], totLen );

                    lastGoodSt = kpt - sizeof( logPrioT );
                    lastGoodSz = totLen;

                    gcurr += totLen;
                    kpt += totLen;

                    continue;
                }
            }
        }

        // If here the one of the checks has failed
        if( inbad == false )
        {
            // a new bad section has started
            badSt = kpt;
            inbad = true;
        }

        ++kpt;
    }

    std::cerr << "--------------------------------------------------------\n";
    std::cerr << "Found " << totBad << " bad bytes ( " << ( 100.0 * totBad ) / fsz << "% bad) \n";
    std::cerr << "Found " << gcurr << " good bytes ( " << ( 100.0 * gcurr ) / fsz << "% good)\n";

    if( totBad == 0 )
    {
        std::cerr << "Taking no action on good file.\n";
    }
    else if( m_checkOnly )
    {
        std::cerr << "Check-only mode set, exiting with error status to indicate failed verification\n";
        delete[] buff;
        delete[] gbuff;
        return errors_found;
    }
    else
    {
        std::string bupPath = m_fname + ".corrupted";

        FILE *fout;
        fout = fopen( bupPath.c_str(), "wb" );

        if( !fout )
        {
            std::cerr << "Error opening corrupted file for writing (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        ssize_t fwr = fwrite( buff, sizeof( char ), fsz, fout );

        int fcst = fclose( fout );

        if( fwr != fsz )
        {
            std::cerr << "Error writing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        if( fcst != 0 )
        {
            std::cerr << "Error closing backup corrupted file (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        std::cerr << "Wrote original file to: " << bupPath << "\n";

        fout = fopen( m_fname.c_str(), "wb" );

        if( !fout )
        {
            std::cerr << "Error opening existing file for writing (" __FILE__ << " " << __LINE__ << ")\n";
            std::cerr << "No further action taken\n";

            delete[] buff;
            delete[] gbuff;
            return error;
        }

        fwr = fwrite( gbuff, sizeof( char ), gcurr, fout );

        fcst = fclose( fout );

        if( fwr != gcurr )
        {
            std::cerr << "Error writing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        if( fcst != 0 )
        {
            std::cerr << "Error closing corrected file (" __FILE__ << " " << __LINE__ << ")\n";
            delete[] buff;
            delete[] gbuff;
            return error;
        }

        std::cerr << "Wrote corrected file to: " << m_fname << "\n";

        std::cerr << "Surgery Complete\n";
    }
    delete[] buff;
    delete[] gbuff;

    return noerror;
}

const std::string &logsurgeon::fname()
{
    return m_fname;
}

bool logsurgeon::checkOnly()
{
    return m_checkOnly;
}

#endif // hpp
